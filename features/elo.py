"""
elo.py — Dynamic Elo rating system for cricket teams.

Computes per-format Elo ratings chronologically, with:
  - K-factor scaling (32 normal, 48 for finals/knockouts)
  - 180-day inactivity decay (regress 10% toward 1500)
  - Leakage-safe: stores pre-match ratings only

Usage:
    from features.elo import EloSystem
    elo = EloSystem(k=32, initial=1500)
    df = elo.compute_elo_features(df)
    # adds columns: elo_team1, elo_team2
"""

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class EloSystem:
    """Elo rating tracker with per-format separation and inactivity decay."""

    def __init__(self, k=32, k_final=48, initial=1500, decay_days=180,
                 decay_factor=0.10):
        """
        Parameters
        ----------
        k : int
            K-factor for normal matches.
        k_final : int
            K-factor for finals / knockout matches.
        initial : float
            Starting Elo for unseen teams.
        decay_days : int
            Days of inactivity before decay kicks in.
        decay_factor : float
            Fraction of the gap (rating - initial) to decay toward initial.
        """
        self.k = k
        self.k_final = k_final
        self.initial = initial
        self.decay_days = decay_days
        self.decay_factor = decay_factor

        # (team, format) → current rating
        self._ratings = defaultdict(lambda: self.initial)
        # (team, format) → last match date
        self._last_seen = {}

    # ── Core Elo math ─────────────────────────────────────────────────────

    def get_rating(self, team, fmt):
        """Return current Elo for (team, format)."""
        return self._ratings[(team, fmt)]

    def _expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _apply_decay(self, team, fmt, match_date):
        """Regress Elo toward initial if team inactive for decay_days+."""
        key = (team, fmt)
        last = self._last_seen.get(key)
        if last is not None and match_date is not None:
            try:
                gap = (match_date - last).days
            except (TypeError, AttributeError):
                return
            if gap >= self.decay_days:
                old = self._ratings[key]
                self._ratings[key] = old - self.decay_factor * (old - self.initial)

    def update(self, winner, loser, fmt, match_date=None, is_final=False):
        """
        Update ratings after a match result.

        Parameters
        ----------
        winner, loser : str  — team names
        fmt : str            — match format (T20, ODI, Test, ...)
        match_date : datetime or None
        is_final : bool      — use higher K-factor for finals
        """
        # Apply decay before updating
        self._apply_decay(winner, fmt, match_date)
        self._apply_decay(loser, fmt, match_date)

        k = self.k_final if is_final else self.k
        ra = self._ratings[(winner, fmt)]
        rb = self._ratings[(loser, fmt)]

        ea = self._expected_score(ra, rb)
        eb = 1.0 - ea

        self._ratings[(winner, fmt)] = ra + k * (1.0 - ea)
        self._ratings[(loser, fmt)] = rb + k * (0.0 - eb)

        # Track last-seen date
        if match_date is not None:
            self._last_seen[(winner, fmt)] = match_date
            self._last_seen[(loser, fmt)] = match_date

    # ── Batch computation on DataFrame ────────────────────────────────────

    def compute_elo_features(self, df):
        """
        Add elo_team1 and elo_team2 columns to df (leakage-safe).

        The DataFrame MUST be sorted by date before calling this.
        Required columns: date, team1, team2, winner, match_type

        Returns a copy with two new columns.
        """
        df = df.copy()
        df["elo_team1"] = np.float32(self.initial)
        df["elo_team2"] = np.float32(self.initial)

        for idx in df.index:
            team1 = str(df.at[idx, "team1"]).strip()
            team2 = str(df.at[idx, "team2"]).strip()
            fmt = str(df.at[idx, "match_type"]).strip()
            winner = str(df.at[idx, "winner"]).strip()
            match_date = df.at[idx, "date"]

            if pd.isna(match_date):
                match_date = None

            # Apply decay before reading
            self._apply_decay(team1, fmt, match_date)
            self._apply_decay(team2, fmt, match_date)

            # Store PRE-MATCH Elo (leakage-safe)
            df.at[idx, "elo_team1"] = self._ratings[(team1, fmt)]
            df.at[idx, "elo_team2"] = self._ratings[(team2, fmt)]

            # Determine winner/loser and update
            w_lower = winner.lower()
            if w_lower == team1.lower():
                self.update(team1, team2, fmt, match_date)
            elif w_lower == team2.lower():
                self.update(team2, team1, fmt, match_date)
            # else: draw / no result — skip update

        return df

    def get_all_ratings(self):
        """Return a dict of all current ratings."""
        return dict(self._ratings)

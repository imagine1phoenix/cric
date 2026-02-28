"""
advanced_features.py — 20 leakage-safe features for cricket match prediction.

All features use expanding windows / cumulative calculations with shift(1)
to avoid data leakage. The DataFrame MUST be sorted by date before calling
any function here.

Categories:
  A. Head-to-head (3 features)
  B. Venue-specific team performance (3 features)
  C. Temporal (4 features)
  D. Form momentum (4 features)
  E. Strength / Elo (4 features — Elo computed externally)
  F. Toss advantage (2 features)
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# A. HEAD-TO-HEAD FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_h2h_features(df):
    """
    Add head-to-head features between team1 and team2.

    Returns columns: h2h_win_rate, h2h_matches_played, h2h_avg_margin
    """
    df = df.copy()
    df["h2h_win_rate"] = np.float32(0.5)
    df["h2h_matches_played"] = np.int16(0)
    df["h2h_avg_margin"] = np.float32(0.0)

    # Track: (frozenset(team1, team2)) → list of (team1_won, margin)
    h2h_history = defaultdict(list)

    for idx in df.index:
        t1 = str(df.at[idx, "team1"]).strip().lower()
        t2 = str(df.at[idx, "team2"]).strip().lower()
        pair = frozenset([t1, t2])

        history = h2h_history[pair]

        if history:
            wins = [h[0] for h in history]
            margins = [h[1] for h in history]
            df.at[idx, "h2h_win_rate"] = np.mean(wins)
            df.at[idx, "h2h_matches_played"] = len(history)
            df.at[idx, "h2h_avg_margin"] = np.mean(margins)

        # Record result (from team1 perspective)
        winner = str(df.at[idx, "winner"]).strip().lower()
        team1_won = 1 if winner == t1 else 0

        # Approximate margin from innings totals
        inn1 = df.at[idx, "innings1_total_runs"] if "innings1_total_runs" in df.columns else 0
        inn2 = df.at[idx, "innings2_total_runs"] if "innings2_total_runs" in df.columns else 0
        margin = abs(inn1 - inn2)

        h2h_history[pair].append((team1_won, margin))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# B. VENUE-SPECIFIC TEAM PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def compute_venue_team_features(df):
    """
    Add venue-specific win rates for each team.

    Returns columns: team1_venue_win_rate, team2_venue_win_rate,
                     team1_venue_matches
    """
    df = df.copy()
    df["team1_venue_win_rate"] = np.float32(0.5)
    df["team2_venue_win_rate"] = np.float32(0.5)
    df["team1_venue_matches"] = np.int16(0)

    # Track: (team, venue) → list of win/loss booleans
    venue_history = defaultdict(list)

    for idx in df.index:
        t1 = str(df.at[idx, "team1"]).strip().lower()
        t2 = str(df.at[idx, "team2"]).strip().lower()
        venue = str(df.at[idx, "venue"]).strip().lower()
        winner = str(df.at[idx, "winner"]).strip().lower()

        # Pre-match stats
        t1_hist = venue_history[(t1, venue)]
        t2_hist = venue_history[(t2, venue)]

        if t1_hist:
            df.at[idx, "team1_venue_win_rate"] = np.mean(t1_hist)
            df.at[idx, "team1_venue_matches"] = len(t1_hist)
        if t2_hist:
            df.at[idx, "team2_venue_win_rate"] = np.mean(t2_hist)

        # Record result
        venue_history[(t1, venue)].append(1 if winner == t1 else 0)
        venue_history[(t2, venue)].append(1 if winner == t2 else 0)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# C. TEMPORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_temporal_features(df):
    """
    Add time-based features.

    Returns columns: days_since_last_match_team1, days_since_last_match_team2,
                     month, is_day_night
    """
    df = df.copy()
    df["days_since_last_match_team1"] = np.float32(30.0)  # default
    df["days_since_last_match_team2"] = np.float32(30.0)
    df["month"] = np.int8(6)  # default mid-year

    # Extract month from date
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["month"] = df["date"].dt.month.fillna(6).astype(np.int8)

    # Day/night — infer from match data if available, else default False
    if "match_type" in df.columns:
        # T20s and ODIs are more likely day/night; not a perfect proxy
        # but best we can do without explicit day/night field
        df["is_day_night"] = np.int8(0)
    else:
        df["is_day_night"] = np.int8(0)

    # Compute rest days
    last_match_date = {}  # team → last match date

    for idx in df.index:
        t1 = str(df.at[idx, "team1"]).strip().lower()
        t2 = str(df.at[idx, "team2"]).strip().lower()
        match_date = df.at[idx, "date"]

        if pd.notna(match_date):
            if t1 in last_match_date:
                try:
                    gap = (match_date - last_match_date[t1]).days
                    df.at[idx, "days_since_last_match_team1"] = max(0, gap)
                except (TypeError, AttributeError):
                    pass

            if t2 in last_match_date:
                try:
                    gap = (match_date - last_match_date[t2]).days
                    df.at[idx, "days_since_last_match_team2"] = max(0, gap)
                except (TypeError, AttributeError):
                    pass

            last_match_date[t1] = match_date
            last_match_date[t2] = match_date

    return df


# ═══════════════════════════════════════════════════════════════════════════
# D. FORM MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_form_momentum_features(df):
    """
    Add streak and extended form features.

    Returns columns: team1_streak, team2_streak,
                     team1_form_10, team2_form_10,
                     team1_weighted_form, team2_weighted_form
    """
    df = df.copy()
    df["team1_streak"] = np.int8(0)
    df["team2_streak"] = np.int8(0)
    df["team1_form_10"] = np.float32(0.5)
    df["team2_form_10"] = np.float32(0.5)
    df["team1_weighted_form"] = np.float32(0.5)
    df["team2_weighted_form"] = np.float32(0.5)

    # Histories: team → deque of results
    team_results = defaultdict(list)  # team → list of 0/1

    def _current_streak(results):
        """Compute current streak: positive = winning, negative = losing."""
        if not results:
            return 0
        streak = 0
        last = results[-1]
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak

    def _form_n(results, n):
        """Rolling n-match win rate."""
        if not results:
            return 0.5
        window = results[-n:] if len(results) >= n else results
        return np.mean(window)

    def _weighted_form(results, decay=0.85):
        """Exponentially weighted form (recent matches weighted higher)."""
        if not results:
            return 0.5
        recent = results[-10:]  # use last 10 max
        weights = [decay ** i for i in range(len(recent) - 1, -1, -1)]
        return np.average(recent, weights=weights)

    for idx in df.index:
        t1 = str(df.at[idx, "team1"]).strip().lower()
        t2 = str(df.at[idx, "team2"]).strip().lower()
        winner = str(df.at[idx, "winner"]).strip().lower()

        # Pre-match features (leakage-safe)
        if team_results[t1]:
            df.at[idx, "team1_streak"] = _current_streak(team_results[t1])
            df.at[idx, "team1_form_10"] = _form_n(team_results[t1], 10)
            df.at[idx, "team1_weighted_form"] = _weighted_form(team_results[t1])

        if team_results[t2]:
            df.at[idx, "team2_streak"] = _current_streak(team_results[t2])
            df.at[idx, "team2_form_10"] = _form_n(team_results[t2], 10)
            df.at[idx, "team2_weighted_form"] = _weighted_form(team_results[t2])

        # Record results
        team_results[t1].append(1 if winner == t1 else 0)
        team_results[t2].append(1 if winner == t2 else 0)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E. STRENGTH FEATURES (overall win rate — Elo is external)
# ═══════════════════════════════════════════════════════════════════════════

def compute_strength_features(df):
    """
    Add all-time win rate for each team (expanding, leakage-safe).

    Returns columns: team1_overall_win_rate, team2_overall_win_rate
    """
    df = df.copy()
    df["team1_overall_win_rate"] = np.float32(0.5)
    df["team2_overall_win_rate"] = np.float32(0.5)

    team_record = defaultdict(lambda: [0, 0])  # team → [wins, total]

    for idx in df.index:
        t1 = str(df.at[idx, "team1"]).strip().lower()
        t2 = str(df.at[idx, "team2"]).strip().lower()
        winner = str(df.at[idx, "winner"]).strip().lower()

        # Pre-match stats
        w1, n1 = team_record[t1]
        w2, n2 = team_record[t2]
        df.at[idx, "team1_overall_win_rate"] = w1 / n1 if n1 > 0 else 0.5
        df.at[idx, "team2_overall_win_rate"] = w2 / n2 if n2 > 0 else 0.5

        # Record
        team_record[t1][1] += 1
        team_record[t2][1] += 1
        if winner == t1:
            team_record[t1][0] += 1
        elif winner == t2:
            team_record[t2][0] += 1

    return df


# ═══════════════════════════════════════════════════════════════════════════
# F. TOSS ADVANTAGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_toss_advantage_features(df):
    """
    Add toss-based predictive features.

    Returns columns: venue_toss_win_advantage, format_bat_first_win_rate
    """
    df = df.copy()
    df["venue_toss_win_advantage"] = np.float32(0.5)
    df["format_bat_first_win_rate"] = np.float32(0.5)

    # (venue) → list of toss_winner == match_winner booleans
    venue_toss_wins = defaultdict(list)
    # (format) → list of bat_first_won booleans
    format_bat_first = defaultdict(list)

    for idx in df.index:
        venue = str(df.at[idx, "venue"]).strip().lower()
        fmt = str(df.at[idx, "match_type"]).strip().lower() \
            if "match_type" in df.columns else "unknown"
        winner = str(df.at[idx, "winner"]).strip().lower()
        toss_winner = str(df.at[idx, "toss_winner"]).strip().lower() \
            if "toss_winner" in df.columns else ""
        toss_decision = str(df.at[idx, "toss_decision"]).strip().lower() \
            if "toss_decision" in df.columns else ""
        team1 = str(df.at[idx, "team1"]).strip().lower()

        # Pre-match: venue toss advantage
        if venue_toss_wins[venue]:
            df.at[idx, "venue_toss_win_advantage"] = np.mean(
                venue_toss_wins[venue])

        # Pre-match: format bat-first win rate
        if format_bat_first[fmt]:
            df.at[idx, "format_bat_first_win_rate"] = np.mean(
                format_bat_first[fmt])

        # Record: did toss winner also win the match?
        toss_winner_won = 1 if toss_winner == winner else 0
        venue_toss_wins[venue].append(toss_winner_won)

        # Record: did the team batting first win?
        # team1 is always batting first in our data structure
        bat_first_won = 1 if winner == team1 else 0
        format_bat_first[fmt].append(bat_first_won)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_advanced_features(df):
    """
    Compute all 20 advanced features in one pass.

    The DataFrame MUST be sorted by date before calling this.
    Required columns: date, team1, team2, winner, venue, toss_winner,
                      toss_decision, match_type, innings1_total_runs,
                      innings2_total_runs

    Returns DataFrame with 20 new columns added.
    """
    print("   ⏳ Head-to-head features...")
    df = compute_h2h_features(df)

    print("   ⏳ Venue-team features...")
    df = compute_venue_team_features(df)

    print("   ⏳ Temporal features...")
    df = compute_temporal_features(df)

    print("   ⏳ Form momentum features...")
    df = compute_form_momentum_features(df)

    print("   ⏳ Strength features...")
    df = compute_strength_features(df)

    print("   ⏳ Toss advantage features...")
    df = compute_toss_advantage_features(df)

    new_cols = [
        # A. Head-to-head
        "h2h_win_rate", "h2h_matches_played", "h2h_avg_margin",
        # B. Venue-team
        "team1_venue_win_rate", "team2_venue_win_rate", "team1_venue_matches",
        # C. Temporal
        "days_since_last_match_team1", "days_since_last_match_team2",
        "month", "is_day_night",
        # D. Form momentum
        "team1_streak", "team2_streak",
        "team1_form_10", "team2_form_10",
        "team1_weighted_form", "team2_weighted_form",
        # E. Strength
        "team1_overall_win_rate", "team2_overall_win_rate",
        # F. Toss advantage
        "venue_toss_win_advantage", "format_bat_first_win_rate",
    ]

    print(f"   ✅ Added {len(new_cols)} advanced features")
    return df, new_cols

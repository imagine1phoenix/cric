"""
player_features.py — Player-level feature computation from ball-by-ball data.

Extracts batter/bowler stats from JSON match data and aggregates to
team-level features. All metrics are expanding (chronological, leakage-safe).

Usage:
    from features.player_features import PlayerFeatureEngine
    engine = PlayerFeatureEngine()
    engine.process_match(match_json)
    team_features = engine.get_team_features("Mumbai Indians")
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("criccric")


class PlayerFeatureEngine:
    """Compute and track player-level statistics from ball-by-ball data."""

    def __init__(self):
        # Batter stats: {player_name: {runs, balls, dismissals, boundaries, innings}}
        self._batter_stats = defaultdict(lambda: {
            "runs": 0, "balls": 0, "dismissals": 0,
            "fours": 0, "sixes": 0, "innings": 0,
        })
        # Bowler stats: {player_name: {runs_conceded, balls, wickets, dots, innings}}
        self._bowler_stats = defaultdict(lambda: {
            "runs_conceded": 0, "balls": 0, "wickets": 0,
            "dots": 0, "innings": 0, "matches": 0,
        })
        # Team rosters: {team_name: set of recent player names}
        self._team_rosters = defaultdict(set)
        # Player match counts
        self._player_matches = defaultdict(int)

    def process_match(self, match_json):
        """
        Process a single match JSON to update player stats.

        Parameters
        ----------
        match_json : dict — full match JSON from Cricsheet
        """
        info = match_json.get("info", {})
        innings_data = match_json.get("innings", [])

        if not innings_data:
            return

        for innings in innings_data:
            team = innings.get("team", "")
            bowlers_seen = set()
            batters_seen = set()

            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    batter = delivery.get("batter", "")
                    bowler = delivery.get("bowler", "")
                    runs = delivery.get("runs", {})
                    batter_runs = runs.get("batter", 0)
                    total_runs = runs.get("total", 0)

                    if batter:
                        self._batter_stats[batter]["runs"] += batter_runs
                        self._batter_stats[batter]["balls"] += 1
                        if batter_runs == 4:
                            self._batter_stats[batter]["fours"] += 1
                        elif batter_runs == 6:
                            self._batter_stats[batter]["sixes"] += 1
                        batters_seen.add(batter)
                        self._team_rosters[team].add(batter)

                    if bowler:
                        self._bowler_stats[bowler]["runs_conceded"] += total_runs
                        self._bowler_stats[bowler]["balls"] += 1
                        if total_runs == 0:
                            self._bowler_stats[bowler]["dots"] += 1
                        bowlers_seen.add(bowler)

                    # Wickets
                    for wicket in delivery.get("wickets", []):
                        player_out = wicket.get("player_out", "")
                        if player_out:
                            self._batter_stats[player_out]["dismissals"] += 1
                        if bowler:
                            kind = wicket.get("kind", "")
                            if kind not in ("run out", "retired hurt", "obstructing the field"):
                                self._bowler_stats[bowler]["wickets"] += 1

            # Update innings counts
            for b in batters_seen:
                self._batter_stats[b]["innings"] += 1
                self._player_matches[b] += 1
            for bw in bowlers_seen:
                self._bowler_stats[bw]["innings"] += 1
                self._bowler_stats[bw]["matches"] += 1

    def get_batter_rating(self, player_name):
        """Compute a single batter rating (0-100 scale)."""
        stats = self._batter_stats.get(player_name)
        if not stats or stats["balls"] == 0:
            return 30.0  # default low rating

        avg = stats["runs"] / max(stats["dismissals"], 1)
        sr = (stats["runs"] / stats["balls"]) * 100
        boundary_pct = (stats["fours"] + stats["sixes"]) / max(stats["balls"], 1)

        # Weighted rating
        rating = min(100, (avg * 0.4 + sr * 0.3 + boundary_pct * 10 * 0.15
                          + min(stats["innings"], 50) * 0.3))
        return round(rating, 1)

    def get_bowler_rating(self, player_name):
        """Compute a single bowler rating (0-100 scale)."""
        stats = self._bowler_stats.get(player_name)
        if not stats or stats["balls"] == 0:
            return 30.0

        econ = (stats["runs_conceded"] / stats["balls"]) * 6
        avg = stats["runs_conceded"] / max(stats["wickets"], 1)
        dot_pct = stats["dots"] / max(stats["balls"], 1)

        # Lower economy and average = better
        rating = min(100, max(0,
            (50 - avg * 0.3) + (12 - econ) * 3 + dot_pct * 20
            + min(stats["matches"], 50) * 0.2))
        return round(max(rating, 0), 1)

    def get_team_features(self, team_name):
        """
        Compute aggregate team-level features from player stats.

        Returns dict with:
            team_avg_batting_rating, team_avg_bowling_rating,
            team_experience, star_player_rating
        """
        roster = self._team_rosters.get(team_name, set())
        if not roster:
            return {
                "team_avg_batting_rating": 30.0,
                "team_avg_bowling_rating": 30.0,
                "team_experience": 0,
                "star_player_rating": 30.0,
            }

        # Top 7 batters by rating
        batter_ratings = sorted(
            [(p, self.get_batter_rating(p)) for p in roster],
            key=lambda x: -x[1]
        )[:7]

        # Top 5 bowlers by rating
        bowler_ratings = sorted(
            [(p, self.get_bowler_rating(p))
             for p in roster if p in self._bowler_stats],
            key=lambda x: -x[1]
        )[:5]

        bat_avg = np.mean([r for _, r in batter_ratings]) if batter_ratings else 30.0
        bowl_avg = np.mean([r for _, r in bowler_ratings]) if bowler_ratings else 30.0
        experience = sum(self._player_matches.get(p, 0) for p in roster)
        star = max((r for _, r in batter_ratings), default=30.0)

        return {
            "team_avg_batting_rating": round(float(bat_avg), 1),
            "team_avg_bowling_rating": round(float(bowl_avg), 1),
            "team_experience": int(experience),
            "star_player_rating": round(float(star), 1),
        }

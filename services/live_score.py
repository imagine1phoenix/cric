"""
live_score.py — Fetch live cricket match data for in-match predictions.

Integrates with CricAPI (free tier: 100 req/day) with ESPN scraping fallback.
Computes live features: current score, run rate, required run rate, etc.

Usage:
    from services.live_score import LiveScoreService
    service = LiveScoreService(api_key="YOUR_KEY")
    matches = service.get_current_matches()
    features = service.compute_live_features(match_data)
"""

import os
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional async HTTP
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LiveScoreService:
    """Fetch live cricket scores and compute in-match prediction features."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("CRICAPI_KEY", "")
        self.base_url = "https://api.cricapi.com/v1"
        self._cache = {}
        self._cache_ttl = 30  # seconds

    # ── API calls ─────────────────────────────────────────────────────────

    def get_current_matches(self):
        """
        Fetch list of currently live matches.

        Returns list of dicts with: id, name, status, teams, matchType, etc.
        """
        if not self.api_key:
            logger.warning("No CRICAPI_KEY set. Cannot fetch live matches.")
            return []

        cache_key = "current_matches"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/currentMatches"
            params = {"apikey": self.api_key, "offset": 0}

            if HAS_REQUESTS:
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
            elif HAS_HTTPX:
                with httpx.Client() as client:
                    resp = client.get(url, params=params, timeout=10)
                    data = resp.json()
            else:
                logger.error("No HTTP library available. Install requests or httpx.")
                return []

            if data.get("status") != "success":
                logger.warning(f"CricAPI error: {data.get('status')}")
                return []

            matches = data.get("data", [])
            # Filter to only live matches
            live = [m for m in matches
                    if m.get("matchStarted", False)
                    and not m.get("matchEnded", False)]

            self._set_cached(cache_key, live)
            return live

        except Exception as e:
            logger.error(f"Failed to fetch live matches: {e}")
            return []

    def get_match_details(self, match_id):
        """
        Fetch detailed scorecard for a specific match.

        Returns dict with: teams, score, overs, wickets, current innings, etc.
        """
        if not self.api_key:
            return None

        cache_key = f"match_{match_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/match_info"
            params = {"apikey": self.api_key, "id": match_id}

            if HAS_REQUESTS:
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
            elif HAS_HTTPX:
                with httpx.Client() as client:
                    resp = client.get(url, params=params, timeout=10)
                    data = resp.json()
            else:
                return None

            if data.get("status") == "success":
                result = data.get("data", {})
                self._set_cached(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"Failed to fetch match {match_id}: {e}")

        return None

    # ── Live feature computation ──────────────────────────────────────────

    def compute_live_features(self, match_data):
        """
        Extract in-match prediction features from live match data.

        Parameters
        ----------
        match_data : dict — from get_match_details()

        Returns
        -------
        dict with computed features:
            current_score, current_wickets, current_over,
            current_run_rate, required_run_rate, run_rate_ratio,
            target, innings (1 or 2), team_batting, team_bowling
        """
        if not match_data:
            return None

        features = {
            "match_id": match_data.get("id", ""),
            "match_name": match_data.get("name", ""),
            "match_type": match_data.get("matchType", ""),
            "team1": "",
            "team2": "",
            "team_batting": "",
            "team_bowling": "",
            "innings": 1,
            "current_score": 0,
            "current_wickets": 0,
            "current_over": 0.0,
            "current_run_rate": 0.0,
            "required_run_rate": 0.0,
            "run_rate_ratio": 0.0,
            "target": 0,
            "innings1_total": 0,
        }

        # Extract teams
        teams = match_data.get("teams", [])
        if len(teams) >= 2:
            features["team1"] = teams[0]
            features["team2"] = teams[1]

        # Parse score from the score array
        scores = match_data.get("score", [])
        if not scores:
            return features

        # Determine innings (last entry in score array is current)
        current_innings = scores[-1] if scores else {}
        features["innings"] = len(scores)

        # If 2nd innings, first entry is 1st innings total
        if len(scores) >= 2:
            first = scores[0]
            features["innings1_total"] = first.get("r", 0)
            features["target"] = first.get("r", 0) + 1

        features["team_batting"] = current_innings.get("inning", "").replace(" Inning 1", "").replace(" Inning 2", "").strip()
        features["current_score"] = current_innings.get("r", 0)
        features["current_wickets"] = current_innings.get("w", 0)
        features["current_over"] = current_innings.get("o", 0.0)

        # Compute rates
        overs = features["current_over"]
        if overs > 0:
            features["current_run_rate"] = round(features["current_score"] / overs, 2)

        # Required run rate (2nd innings only)
        if features["innings"] >= 2 and features["target"] > 0:
            match_type = features["match_type"].upper()
            total_overs = 20 if "T20" in match_type else 50 if "ODI" in match_type else 90
            remaining_overs = max(total_overs - overs, 0.1)
            runs_needed = features["target"] - features["current_score"]
            features["required_run_rate"] = round(
                max(runs_needed, 0) / remaining_overs, 2)

            if features["required_run_rate"] > 0:
                features["run_rate_ratio"] = round(
                    features["current_run_rate"] / features["required_run_rate"], 3)

        # Team bowling is the other team
        batting = features["team_batting"]
        if batting == features["team1"]:
            features["team_bowling"] = features["team2"]
        else:
            features["team_bowling"] = features["team1"]

        return features

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _get_cached(self, key):
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
        return None

    def _set_cached(self, key, data):
        self._cache[key] = (time.time(), data)

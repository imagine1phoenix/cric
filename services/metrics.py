"""
metrics.py — Prediction metrics tracking for analytics dashboard.
"""
import time
import logging
from collections import defaultdict, deque
from threading import Lock

logger = logging.getLogger("criccric")


class PredictionMetrics:
    """Thread-safe in-memory metrics tracker for predictions."""

    def __init__(self, max_history=10000):
        self._lock = Lock()
        self._total = 0
        self._today_count = 0
        self._today_date = None
        self._response_times = deque(maxlen=max_history)
        self._cache_hits = 0
        self._cache_misses = 0
        self._team_counts = defaultdict(int)
        self._league_counts = defaultdict(int)
        self._winner_counts = defaultdict(int)
        self._hourly = defaultdict(int)

    def record_prediction(self, team1, team2, league, winner,
                          probability, response_time_ms, cache_hit):
        """Record a prediction event."""
        with self._lock:
            self._total += 1

            import datetime
            today = datetime.date.today().isoformat()
            if self._today_date != today:
                self._today_date = today
                self._today_count = 0
            self._today_count += 1

            self._response_times.append(response_time_ms)

            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            if team1:
                self._team_counts[team1] += 1
            if team2:
                self._team_counts[team2] += 1
            if league:
                self._league_counts[league] += 1
            if winner:
                self._winner_counts[winner] += 1

            hour = time.strftime("%H")
            self._hourly[hour] += 1

    def get_dashboard(self):
        """Return metrics summary for the admin dashboard."""
        with self._lock:
            avg_rt = 0.0
            if self._response_times:
                avg_rt = sum(self._response_times) / len(self._response_times)

            total_cache = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_cache * 100) if total_cache > 0 else 0

            # Top items
            top_teams = sorted(self._team_counts.items(),
                              key=lambda x: -x[1])[:10]
            top_leagues = sorted(self._league_counts.items(),
                                key=lambda x: -x[1])[:10]
            top_winners = sorted(self._winner_counts.items(),
                                key=lambda x: -x[1])[:10]

            return {
                "total_predictions": self._total,
                "predictions_today": self._today_count,
                "avg_response_time_ms": round(avg_rt, 2),
                "cache_hit_rate_pct": round(hit_rate, 1),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "most_predicted_teams": [
                    {"team": t, "count": c} for t, c in top_teams
                ],
                "most_predicted_leagues": [
                    {"league": l, "count": c} for l, c in top_leagues
                ],
                "most_predicted_winners": [
                    {"winner": w, "count": c} for w, c in top_winners
                ],
                "hourly_distribution": dict(self._hourly),
            }

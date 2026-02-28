"""
history_db.py — SQLite-backed prediction history for tracking and accuracy analysis.
"""
import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime
from threading import Lock

logger = logging.getLogger("criccric")


class HistoryDB:
    """Thread-safe SQLite prediction history store."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    team1 TEXT,
                    team2 TEXT,
                    venue TEXT,
                    city TEXT,
                    league TEXT,
                    format TEXT,
                    predicted_winner TEXT,
                    win_probability REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    response_time_ms REAL,
                    cache_hit BOOLEAN,
                    model_version TEXT,
                    actual_winner TEXT,
                    correct BOOLEAN,
                    session_id TEXT,
                    input_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON prediction_history(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_league
                ON prediction_history(league)
            """)
            conn.commit()
        logger.info(f"History DB initialised: {self.db_path}")

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def record_prediction(self, prediction_id, input_data, result,
                          response_time_ms=0, cache_hit=False, session_id=None):
        """Insert a prediction record."""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT INTO prediction_history
                        (id, team1, team2, venue, city, league, format,
                         predicted_winner, win_probability, confidence_lower,
                         confidence_upper, response_time_ms, cache_hit,
                         model_version, session_id, input_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction_id,
                        input_data.get("team1", ""),
                        input_data.get("team2", ""),
                        input_data.get("venue", ""),
                        input_data.get("city", ""),
                        input_data.get("league", ""),
                        input_data.get("match_type", ""),
                        result.get("winner", ""),
                        result.get("confidence", 0),
                        result.get("ci_low", 0),
                        result.get("ci_high", 0),
                        response_time_ms,
                        cache_hit,
                        result.get("model_version", ""),
                        session_id or "",
                        json.dumps(input_data),
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to record prediction: {e}")

    def get_history(self, page=1, per_page=20, league="", match_format=""):
        """Get paginated prediction history."""
        offset = (page - 1) * per_page
        where_parts = []
        params = []

        if league:
            where_parts.append("league = ?")
            params.append(league)
        if match_format:
            where_parts.append("format = ?")
            params.append(match_format)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row

            # Count
            count = conn.execute(
                f"SELECT COUNT(*) FROM prediction_history WHERE {where_clause}",
                params
            ).fetchone()[0]

            # Records
            rows = conn.execute(
                f"""SELECT * FROM prediction_history
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?""",
                params + [per_page, offset]
            ).fetchall()

        return {
            "predictions": [dict(r) for r in rows],
            "page": page,
            "per_page": per_page,
            "total": count,
            "pages": (count + per_page - 1) // per_page,
        }

    def get_accuracy_stats(self):
        """Compute accuracy for predictions where actual_winner is known."""
        with self._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM prediction_history WHERE actual_winner IS NOT NULL"
            ).fetchone()[0]

            correct = conn.execute(
                "SELECT COUNT(*) FROM prediction_history WHERE correct = 1"
            ).fetchone()[0]

            # By league
            league_stats = conn.execute("""
                SELECT league,
                       COUNT(*) as total,
                       SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct
                FROM prediction_history
                WHERE actual_winner IS NOT NULL
                GROUP BY league
            """).fetchall()

            # By format
            format_stats = conn.execute("""
                SELECT format,
                       COUNT(*) as total,
                       SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct
                FROM prediction_history
                WHERE actual_winner IS NOT NULL
                GROUP BY format
            """).fetchall()

        return {
            "total_verified": total,
            "total_correct": correct,
            "accuracy_pct": round(correct / total * 100, 1) if total > 0 else 0,
            "by_league": [
                {"league": r[0], "total": r[1], "correct": r[2],
                 "accuracy": round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0}
                for r in league_stats
            ],
            "by_format": [
                {"format": r[0], "total": r[1], "correct": r[2],
                 "accuracy": round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0}
                for r in format_stats
            ],
        }

    def update_actual_result(self, prediction_id, actual_winner):
        """Update a prediction with the actual match result."""
        with self._lock:
            with self._get_conn() as conn:
                # Get predicted winner
                row = conn.execute(
                    "SELECT predicted_winner FROM prediction_history WHERE id = ?",
                    (prediction_id,)
                ).fetchone()
                if not row:
                    return False

                correct = row[0] == actual_winner
                conn.execute(
                    """UPDATE prediction_history
                       SET actual_winner = ?, correct = ?
                       WHERE id = ?""",
                    (actual_winner, correct, prediction_id)
                )
                conn.commit()
                return True

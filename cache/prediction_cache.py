"""
prediction_cache.py — Cache layer for prediction results.

Uses Flask-Caching with MD5-hashed keys to avoid redundant model inference.
Supports SimpleCache (single-server) and Redis (multi-server).
"""

import json
import hashlib
import logging
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Cache key generator ────────────────────────────────────────────────────

def make_prediction_cache_key(input_data):
    """
    Generate a deterministic cache key from prediction inputs.

    Normalises input by lowercasing, stripping whitespace, and sorting keys
    so that {"team1":"India","team2":"Aus"} and {"team2":"aus","team1":"india"}
    produce the same key.
    """
    normalised = {}
    for k, v in input_data.items():
        v_clean = str(v).strip().lower() if v else ""
        normalised[k] = v_clean

    sorted_input = json.dumps(normalised, sort_keys=True)
    digest = hashlib.md5(sorted_input.encode()).hexdigest()
    return f"pred_{digest}"


# ── Cache stats tracking ──────────────────────────────────────────────────

class CacheStats:
    """Simple hit/miss counter (in-memory, resets on restart)."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.clears = 0
        self.last_clear = None

    @property
    def total(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return round(self.hits / self.total * 100, 1) if self.total > 0 else 0.0

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    def record_clear(self):
        self.clears += 1
        self.last_clear = datetime.utcnow().isoformat() + "Z"
        # Reset counters after clear
        self.hits = 0
        self.misses = 0

    def to_dict(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total,
            "hit_rate_pct": self.hit_rate,
            "cache_clears": self.clears,
            "last_clear": self.last_clear,
        }


# Singleton stats instance
cache_stats = CacheStats()


# ── Init helper ───────────────────────────────────────────────────────────

def init_cache(app, cache):
    """
    Configure Flask-Caching on the app.

    Call this once at app creation time:
        from flask_caching import Cache
        cache = Cache()
        init_cache(app, cache)
    """
    app.config.setdefault("CACHE_TYPE", "SimpleCache")
    app.config.setdefault("CACHE_DEFAULT_TIMEOUT", 86400)   # 24 hours
    app.config.setdefault("CACHE_THRESHOLD", 10000)         # max items
    app.config.setdefault("CACHE_KEY_PREFIX", "cricp_")

    cache.init_app(app)
    logger.info(
        f"Cache initialised: type={app.config['CACHE_TYPE']}, "
        f"TTL={app.config['CACHE_DEFAULT_TIMEOUT']}s, "
        f"max_items={app.config['CACHE_THRESHOLD']}"
    )

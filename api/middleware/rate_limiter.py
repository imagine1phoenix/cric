"""
rate_limiter.py — Flask-Limiter setup and helpers.
"""
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger("criccric")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri="memory://",
)


def init_limiter(app):
    """Initialize rate limiter on the Flask app."""
    limiter.init_app(app)
    logger.info("Rate limiter initialised")
    return limiter

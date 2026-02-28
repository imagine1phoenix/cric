"""
config.py — Application configuration classes.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "criccric-dev-key")
    BASE_DIR = BASE_DIR

    # Model
    MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
    LEGACY_MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Cache
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 86400
    CACHE_THRESHOLD = 10000
    CACHE_KEY_PREFIX = "cricp_"
    CACHE_ADMIN_KEY = os.environ.get("CACHE_ADMIN_KEY", "criccric-admin")

    # Compression
    COMPRESS_MIMETYPES = [
        "text/html", "text/css", "text/xml", "text/javascript",
        "application/json", "application/javascript",
    ]
    COMPRESS_MIN_SIZE = 256

    # Rate limiting
    RATELIMIT_DEFAULT = "60/minute"
    RATELIMIT_STORAGE_URI = "memory://"
    RATELIMIT_HEADERS_ENABLED = True

    # Logging
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    LOG_LEVEL = "INFO"

    # Database
    HISTORY_DB_PATH = os.path.join(BASE_DIR, "data", "history.db")

    # External APIs
    CRICAPI_KEY = os.environ.get("CRICAPI_KEY", "")
    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")

    # Dropdowns
    DROPDOWNS_JSON = os.path.join(BASE_DIR, "static", "data", "dropdowns.json")


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    RATELIMIT_DEFAULT = "30/minute"
    CACHE_TYPE = "SimpleCache"


class TestingConfig(Config):
    TESTING = True
    CACHE_TYPE = "NullCache"


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

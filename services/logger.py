"""
logger.py — Structured JSON logging with daily rotation.
"""
import os
import logging
import logging.handlers
from datetime import datetime

_initialized = False


def setup_logging(log_dir="logs", log_level="INFO"):
    """
    Configure structured logging with daily rotating file handler.

    Parameters
    ----------
    log_dir : str — directory for log files
    log_level : str — logging level
    """
    global _initialized
    if _initialized:
        return logging.getLogger("criccric")

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("criccric")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # JSON-like structured formatter
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            # Flatten to structured log line
            parts = [f"{k}={v}" for k, v in log_entry.items()]
            return " | ".join(parts)

    # File handler — daily rotation, keep 30 days
    log_path = os.path.join(log_dir, "criccric.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)

    # Console handler for dev
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(console)

    _initialized = True
    logger.info("Structured logging initialised")
    return logger

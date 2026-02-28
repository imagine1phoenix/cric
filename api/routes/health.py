"""
health.py — /api/health endpoint for readiness and liveness probes.
"""
import logging
from flask import Blueprint, jsonify
from datetime import datetime

logger = logging.getLogger("criccric")

health_bp = Blueprint("health", __name__, url_prefix="/api")

_model_mgr = None


def init_health_bp(model_mgr):
    global _model_mgr
    _model_mgr = model_mgr


@health_bp.route("/health")
def health():
    """Health check / liveness probe."""
    checks = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": False,
        "model_version": None,
    }

    if _model_mgr:
        checks["model_loaded"] = _model_mgr.is_loaded
        if _model_mgr.is_loaded:
            try:
                meta = _model_mgr.get_metadata()
                checks["model_version"] = meta.get("version_hash", "unknown")
            except:
                pass

    status_code = 200 if checks["model_loaded"] else 503
    return jsonify(checks), status_code


@health_bp.route("/ready")
def ready():
    """Readiness probe — is the app ready to serve predictions?"""
    if _model_mgr and _model_mgr.is_loaded:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not_ready", "reason": "Model not loaded"}), 503

"""
metadata.py — /api/meta endpoints for dropdowns, stats, model info.
"""
import os
import json
import logging
from flask import Blueprint, jsonify, request

from cache.prediction_cache import cache_stats

logger = logging.getLogger("criccric")

meta_bp = Blueprint("meta", __name__, url_prefix="/api")

_cache = None
_model_mgr = None
_metrics = None


def init_meta_bp(cache, model_mgr, metrics=None):
    global _cache, _model_mgr, _metrics
    _cache = cache
    _model_mgr = model_mgr
    _metrics = metrics


@meta_bp.route("/dropdown-data")
def dropdown_data():
    """Return all dropdown options."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base, "static", "data", "dropdowns.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            dd = json.load(f)
        return jsonify({
            "leagues": dd.get("leagues", []),
            "match_types": dd.get("formats", []),
            "genders": dd.get("genders", []),
            "cities": dd.get("cities", []),
            "teams": dd.get("teams", []),
            "venues": dd.get("venues", []),
            "league_teams": {k: v.get("teams", [])
                            for k, v in dd.get("league_mappings", {}).items()},
            "league_venues": {k: v.get("venues", [])
                             for k, v in dd.get("league_mappings", {}).items()},
            "league_cities": {k: v.get("cities", [])
                             for k, v in dd.get("league_mappings", {}).items()},
        })
    return jsonify({"error": "Dropdown data not found"}), 404


@meta_bp.route("/model-info")
def model_info():
    """Return model version, accuracy, features."""
    try:
        meta = _model_mgr.get_metadata()
        return jsonify({
            "version": meta.get("version_hash", "unknown"),
            "trained_at": meta.get("trained_at", "unknown"),
            "accuracy": meta.get("accuracy"),
            "roc_auc": meta.get("roc_auc"),
            "ci": meta.get("bootstrap_ci_accuracy"),
            "n_features": meta.get("n_features"),
            "compression": meta.get("compression_level"),
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not loaded"}), 503


@meta_bp.route("/warmup")
def warmup():
    """Pre-load model to avoid cold-start latency."""
    try:
        return jsonify(_model_mgr.warmup())
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@meta_bp.route("/cache/stats")
def cache_stats_endpoint():
    return jsonify(cache_stats.to_dict())


@meta_bp.route("/cache/clear", methods=["POST"])
def cache_clear():
    admin_key = request.headers.get("X-Admin-Key", "")
    expected = os.environ.get("CACHE_ADMIN_KEY", "criccric-admin")
    if admin_key != expected:
        return jsonify({"error": "Unauthorized"}), 403
    if _cache:
        _cache.clear()
    cache_stats.record_clear()
    return jsonify({"status": "cleared", "stats": cache_stats.to_dict()})


@meta_bp.route("/admin/metrics")
def admin_metrics():
    """Return prediction metrics dashboard data."""
    admin_key = request.headers.get("X-Admin-Key", "")
    expected = os.environ.get("CACHE_ADMIN_KEY", "criccric-admin")
    if admin_key != expected:
        return jsonify({"error": "Unauthorized"}), 403

    if _metrics:
        return jsonify(_metrics.get_dashboard())
    return jsonify({"error": "Metrics not available"}), 503

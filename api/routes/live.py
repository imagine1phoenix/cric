"""
live.py — /api/live endpoints for live match predictions.
"""
import logging
from flask import Blueprint, jsonify

logger = logging.getLogger("criccric")

live_bp = Blueprint("live", __name__, url_prefix="/api/live")

_live_service = None
_model_mgr = None


def init_live_bp(live_service, model_mgr):
    global _live_service, _model_mgr
    _live_service = live_service
    _model_mgr = model_mgr


@live_bp.route("/matches")
def live_matches():
    """List currently live cricket matches."""
    if not _live_service:
        return jsonify({"matches": [], "count": 0, "note": "No API key configured"})
    matches = _live_service.get_current_matches()
    return jsonify({"matches": matches, "count": len(matches)})


@live_bp.route("/predict/<match_id>")
def live_predict(match_id):
    """In-match prediction for a live match."""
    if not _live_service:
        return jsonify({"error": "Live score service not configured"}), 503

    details = _live_service.get_match_details(match_id)
    if not details:
        return jsonify({"error": f"Match {match_id} not found or not live"}), 404

    features = _live_service.compute_live_features(details)
    if not features:
        return jsonify({"error": "Could not compute live features"}), 500

    try:
        prediction = _model_mgr.predict({
            "team1": features.get("team1", ""),
            "team2": features.get("team2", ""),
            "venue": "",
            "match_type": features.get("match_type", "T20"),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({"match": features, "prediction": prediction})


@live_bp.route("/subscriptions")
def live_subscriptions():
    """Active WebSocket subscriptions."""
    try:
        from services.websocket_manager import get_active_subscriptions
        return jsonify(get_active_subscriptions())
    except ImportError:
        return jsonify({"error": "WebSocket not available"}), 503

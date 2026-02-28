"""
history.py — /api/history and /history page for prediction tracking.
"""
import logging
from flask import Blueprint, request, jsonify, render_template

logger = logging.getLogger("criccric")

history_bp = Blueprint("history", __name__)

_history_db = None


def init_history_bp(history_db):
    global _history_db
    _history_db = history_db


@history_bp.route("/history")
def history_page():
    """Render prediction history page."""
    return render_template("history.html")


@history_bp.route("/api/history")
def history_api():
    """
    Paginated prediction history.
    GET /api/history?page=1&per_page=20&league=IPL
    """
    if not _history_db:
        return jsonify({"error": "History tracking not available"}), 503

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    league = request.args.get("league", "").strip()
    match_format = request.args.get("format", "").strip()

    result = _history_db.get_history(
        page=page, per_page=per_page,
        league=league, match_format=match_format,
    )
    return jsonify(result)


@history_bp.route("/api/history/accuracy")
def history_accuracy():
    """Accuracy stats for completed predictions."""
    if not _history_db:
        return jsonify({"error": "History tracking not available"}), 503
    return jsonify(_history_db.get_accuracy_stats())

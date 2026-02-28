"""
compare.py — /api/compare and /compare page for team comparisons.
"""
import os
import json
import logging
from flask import Blueprint, request, jsonify, render_template

logger = logging.getLogger("criccric")

compare_bp = Blueprint("compare", __name__)

_model_mgr = None


def init_compare_bp(model_mgr):
    global _model_mgr
    _model_mgr = model_mgr


def _get_dropdown_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "static", "data", "dropdowns.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@compare_bp.route("/compare")
def compare_page():
    """Render team comparison page."""
    dd = _get_dropdown_data()
    return render_template("compare.html",
                           teams=dd.get("teams", []),
                           formats=dd.get("formats", []))


@compare_bp.route("/api/compare")
def compare_api():
    """
    Team comparison data.
    GET /api/compare?team1=MI&team2=CSK&format=T20
    """
    team1 = request.args.get("team1", "").strip()
    team2 = request.args.get("team2", "").strip()
    match_format = request.args.get("format", "").strip()

    if not team1 or not team2:
        return jsonify({"error": "team1 and team2 are required"}), 400

    # Load artifacts to compute stats
    try:
        artifacts = _model_mgr.get_artifacts()
    except:
        return jsonify({"error": "Model artifacts not loaded"}), 503

    dropdown_data = artifacts.get("dropdown_data", {})

    # Build basic comparison data from available information
    comparison = {
        "team1": team1,
        "team2": team2,
        "format": match_format,
        "h2h": {"team1_wins": 0, "team2_wins": 0, "draws": 0, "no_results": 0},
        "team1_stats": {
            "overall_win_rate": 0.0,
            "recent_form": [],
            "elo": 1500,
            "leagues": [],
        },
        "team2_stats": {
            "overall_win_rate": 0.0,
            "recent_form": [],
            "elo": 1500,
            "leagues": [],
        },
        "venue_comparison": {},
        "prediction": None,
    }

    # Run prediction
    try:
        pred = _model_mgr.predict({
            "team1": team1,
            "team2": team2,
            "venue": "",
            "match_type": match_format or None,
        })
        comparison["prediction"] = pred
    except:
        pass

    # Identify common leagues
    lt = dropdown_data.get("league_teams", {})
    for league, teams_in_league in lt.items():
        t1_in = team1 in teams_in_league
        t2_in = team2 in teams_in_league
        if t1_in:
            comparison["team1_stats"]["leagues"].append(league)
        if t2_in:
            comparison["team2_stats"]["leagues"].append(league)

    return jsonify(comparison)

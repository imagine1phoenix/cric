"""
request_validator.py — Validate prediction inputs against known valid values.
"""
import os
import json
import logging
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger("criccric")

_valid_values = None


def _load_valid_values():
    """Load valid values from precomputed dropdowns.json."""
    global _valid_values
    if _valid_values is not None:
        return _valid_values

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "static", "data", "dropdowns.json")
    if os.path.exists(path):
        with open(path) as f:
            dd = json.load(f)
        _valid_values = {
            "teams": set(dd.get("teams", [])),
            "venues": set(dd.get("venues", [])),
            "cities": set(dd.get("cities", [])),
            "leagues": set(dd.get("leagues", [])),
            "formats": set(dd.get("formats", [])),
            "genders": set(dd.get("genders", [])),
        }
    else:
        _valid_values = {
            "teams": set(), "venues": set(), "cities": set(),
            "leagues": set(), "formats": set(), "genders": set(),
        }
    return _valid_values


def validate_prediction_input(func):
    """Decorator that validates prediction inputs against known values."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json()
        if not data:
            return jsonify({"errors": ["No JSON data provided"]}), 400

        vv = _load_valid_values()
        errors = []

        team1 = data.get("team1", "").strip()
        team2 = data.get("team2", "").strip()

        if not team1:
            errors.append("team1 is required")
        elif vv["teams"] and team1 not in vv["teams"]:
            errors.append(f"Unknown team1: {team1}")

        if not team2:
            errors.append("team2 is required")
        elif vv["teams"] and team2 not in vv["teams"]:
            errors.append(f"Unknown team2: {team2}")

        if team1 and team2 and team1 == team2:
            errors.append("team1 and team2 cannot be the same")

        venue = data.get("venue", "").strip()
        if venue and vv["venues"] and venue not in vv["venues"]:
            errors.append(f"Unknown venue: {venue}")

        toss_winner = data.get("toss_winner", "").strip()
        if toss_winner and team1 and team2:
            if toss_winner not in (team1, team2):
                errors.append("toss_winner must be either team1 or team2")

        toss_decision = data.get("toss_decision", "").strip()
        if toss_decision and toss_decision not in ("bat", "field"):
            errors.append("toss_decision must be 'bat' or 'field'")

        if errors:
            return jsonify({"errors": errors}), 400

        return func(*args, **kwargs)
    return wrapper

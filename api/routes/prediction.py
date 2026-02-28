"""
prediction.py — /api/predict endpoints.
"""
import uuid
import time
import logging
from flask import Blueprint, request, jsonify, make_response

from api.middleware.rate_limiter import limiter
from api.middleware.request_validator import validate_prediction_input
from cache.prediction_cache import make_prediction_cache_key, cache_stats

logger = logging.getLogger("criccric")

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

# These get set by the app factory
_cache = None
_model_mgr = None
_history_db = None
_metrics = None


def init_predict_bp(cache, model_mgr, history_db=None, metrics=None):
    global _cache, _model_mgr, _history_db, _metrics
    _cache = cache
    _model_mgr = model_mgr
    _history_db = history_db
    _metrics = metrics


@predict_bp.route("/predict", methods=["POST"])
@limiter.limit("30/minute")
@validate_prediction_input
def predict():
    start_time = time.time()
    data = request.get_json()

    team1 = data.get("team1", "").strip()
    team2 = data.get("team2", "").strip()

    input_data = {
        "team1": team1, "team2": team2,
        "venue": data.get("venue", "").strip(),
        "toss_winner": data.get("toss_winner", "").strip() or None,
        "toss_decision": data.get("toss_decision", "").strip() or None,
        "match_type": data.get("match_type", data.get("format", "")).strip() or None,
        "gender": data.get("gender", "").strip() or None,
        "league": data.get("league", "").strip() or None,
        "city": data.get("city", "").strip() or None,
    }

    # Cache check
    cache_key = make_prediction_cache_key(input_data)
    cached_result = _cache.get(cache_key) if _cache else None

    if cached_result is not None:
        cache_stats.record_hit()
        cached_result["cached"] = True
        cached_result["prediction_id"] = str(uuid.uuid4())
        resp = make_response(jsonify(cached_result))
        resp.headers["X-Cache-Status"] = "HIT"

        elapsed = (time.time() - start_time) * 1000
        _log_prediction(input_data, cached_result, elapsed, True)
        return resp

    cache_stats.record_miss()

    try:
        result = _model_mgr.predict(input_data)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    prediction_id = str(uuid.uuid4())
    result["prediction_id"] = prediction_id
    result["cached"] = False

    # Compute SHAP explanation if available
    explanation = _compute_explanation(input_data)
    if explanation:
        result["explanation"] = explanation

    # Cache the result
    if _cache:
        _cache.set(cache_key, result)

    resp = make_response(jsonify(result))
    resp.headers["X-Cache-Status"] = "MISS"

    elapsed = (time.time() - start_time) * 1000

    # Record in history DB
    if _history_db:
        try:
            _history_db.record_prediction(
                prediction_id=prediction_id,
                input_data=input_data,
                result=result,
                response_time_ms=elapsed,
                cache_hit=False,
            )
        except Exception as e:
            logger.warning(f"Failed to record prediction: {e}")

    _log_prediction(input_data, result, elapsed, False)
    return resp


def _log_prediction(input_data, result, elapsed_ms, cache_hit):
    """Structured logging for predictions."""
    if _metrics:
        _metrics.record_prediction(
            team1=input_data.get("team1", ""),
            team2=input_data.get("team2", ""),
            league=input_data.get("league", ""),
            winner=result.get("winner", ""),
            probability=result.get("confidence", 0),
            response_time_ms=elapsed_ms,
            cache_hit=cache_hit,
        )
    logger.info(
        f"Prediction: {input_data.get('team1')} vs {input_data.get('team2')} "
        f"→ {result.get('winner')} ({result.get('confidence', 0):.1f}%) "
        f"[{'HIT' if cache_hit else 'MISS'}] {elapsed_ms:.1f}ms"
    )


def _compute_explanation(input_data):
    """Compute SHAP-based explanation for a prediction (best-effort)."""
    try:
        import shap
        import numpy as np

        mgr = _model_mgr
        if not mgr or not mgr.is_loaded:
            return None

        model = mgr._model
        artifacts = mgr.get_artifacts()
        feature_names = artifacts.get("feature_list", [])
        if not feature_names or model is None:
            return None

        # Build feature vector (same logic as ModelManager.predict)
        row = mgr._build_feature_row(input_data)
        if row is None:
            return None

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]  # class 1

        vals = sv[0] if len(sv.shape) > 1 else sv
        abs_vals = np.abs(vals)
        top_idx = np.argsort(abs_vals)[-5:][::-1]

        # Friendly feature names
        name_map = {
            "elo_team1": "Elo Rating (Team 1)",
            "elo_team2": "Elo Rating (Team 2)",
            "team1_recent_form": "Recent Form (Team 1)",
            "team2_recent_form": "Recent Form (Team 2)",
            "venue_avg_score": "Venue Average Score",
            "h2h_win_rate": "Head-to-Head Win Rate",
            "team1_overall_win_rate": "Overall Win Rate (Team 1)",
            "team2_overall_win_rate": "Overall Win Rate (Team 2)",
            "team1_weighted_form": "Weighted Form (Team 1)",
            "team2_weighted_form": "Weighted Form (Team 2)",
            "venue_toss_win_advantage": "Venue Toss Advantage",
            "format_bat_first_win_rate": "Bat-First Win Rate (Format)",
            "innings1_total_runs": "1st Innings Runs",
            "innings2_total_runs": "2nd Innings Runs",
            "innings1_wickets": "1st Innings Wickets",
            "innings2_wickets": "2nd Innings Wickets",
            "toss_decision_bat_first": "Toss: Bat First",
        }

        factors = []
        for i in top_idx:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            impact = float(vals[i])
            factors.append({
                "feature": name_map.get(fname, fname.replace("_", " ").title()),
                "feature_raw": fname,
                "impact": round(impact, 4),
                "direction": "positive" if impact > 0 else "negative",
                "description": "",
            })

        return {"factors": factors}

    except Exception as e:
        logger.debug(f"SHAP explanation failed: {e}")
        return None

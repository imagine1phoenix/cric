"""
app.py — Application Factory for CricPredict.

Assembles all blueprints, middleware, and services.
Usage:
    flask --app app run
"""

import os
from flask import Flask, render_template

from config import config_map

# Middleware & Ext
from flask_caching import Cache
from flask_compress import Compress
from cache.prediction_cache import init_cache
from api.middleware.rate_limiter import init_limiter
from api.middleware.error_handler import register_error_handlers

# Models & Services
from model.model_manager import ModelManager
from services.logger import setup_logging
from services.metrics import PredictionMetrics
from services.history_db import HistoryDB
from services.task_queue import TaskQueue
from services.live_score import LiveScoreService

# Blueprints
from api.routes.prediction import predict_bp, init_predict_bp
from api.routes.live import live_bp, init_live_bp
from api.routes.metadata import meta_bp, init_meta_bp
from api.routes.health import health_bp, init_health_bp
from api.routes.compare import compare_bp, init_compare_bp
from api.routes.history import history_bp, init_history_bp

# Globals (for CLI or simple imports if needed)
cache = Cache()
metrics = PredictionMetrics()
history_db = None
task_queue = TaskQueue()


def create_app(config_name="development"):
    """Flask application factory."""
    app = Flask(__name__)
    
    # ── Configuration ──
    app.config.from_object(config_map[config_name])
    
    # ── Setup Logging ──
    logger = setup_logging(app.config["LOG_DIR"], app.config["LOG_LEVEL"])
    logger.info(f"Starting CricPredict ({config_name} mode)")

    # ── Initialize Extensions ──
    Compress(app)
    init_cache(app, cache)
    init_limiter(app)
    register_error_handlers(app)

    # ── Initialize Core Services ──
    global history_db
    history_db = HistoryDB(app.config["HISTORY_DB_PATH"])
    
    live_service = LiveScoreService(app.config["CRICAPI_KEY"])
    
    # Load model manager
    model_mgr = ModelManager.get_instance()
    # Check for ensemble model or compressed single model
    if os.path.exists(os.path.join(app.config["MODEL_ARTIFACTS_DIR"], "ensemble", "ensemble_config.json")) or \
       os.path.exists(os.path.join(app.config["MODEL_ARTIFACTS_DIR"], "model_v2.joblib")):
        model_mgr.set_artifact_path(app.config["MODEL_ARTIFACTS_DIR"])
        logger.info(f"Using model artifacts from {app.config['MODEL_ARTIFACTS_DIR']}")
    elif os.path.exists(os.path.join(app.config["LEGACY_MODEL_DIR"], "cricket_rf_bootstrap.pkl")):
        model_mgr.set_artifact_path(app.config["LEGACY_MODEL_DIR"])
        logger.info(f"Using legacy model from {app.config['LEGACY_MODEL_DIR']}")
    else:
        logger.warning("No model files found! Run train_model.py first.")

    # ── Initialize Blueprint Dependencies ──
    init_predict_bp(cache, model_mgr, history_db, metrics)
    init_live_bp(live_service, model_mgr)
    init_meta_bp(cache, model_mgr, metrics)
    init_health_bp(model_mgr)
    init_compare_bp(model_mgr)
    init_history_bp(history_db)

    # ── Register Blueprints ──
    app.register_blueprint(predict_bp)
    app.register_blueprint(live_bp)
    app.register_blueprint(meta_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(history_bp)

    # ── Static/Root Routes ──
    @app.route("/")
    def index():
        try:
            model_mgr.get_model()  # trigger lazy load
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
        dd = _get_dropdown_data(app)
        return render_template("index.html", 
                               model_loaded=model_mgr.is_loaded,
                               **dd)

    # Static Cache Headers
    @app.after_request
    def add_cache_headers(response):
        if request.path.startswith("/static/dist/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.path.startswith("/static/data/"):
            response.headers["Cache-Control"] = "public, max-age=604800"
        elif request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    return app


def _get_dropdown_data(app):
    import json
    json_path = app.config["DROPDOWNS_JSON"]
    if os.path.exists(json_path):
        with open(json_path) as f:
            dd = json.load(f)
        return {
            "leagues": dd.get("leagues", []),
            "match_types": dd.get("formats", []),
            "genders": dd.get("genders", []),
            "cities": dd.get("cities", []),
            "teams": dd.get("teams", []),
            "venues": dd.get("venues", []),
            "league_teams": {k: v.get("teams", []) for k, v in dd.get("league_mappings", {}).items()},
            "league_venues": {k: v.get("venues", []) for k, v in dd.get("league_mappings", {}).items()},
            "league_cities": {k: v.get("cities", []) for k, v in dd.get("league_mappings", {}).items()},
        }
    return {"leagues": [], "match_types": [], "genders": [], "cities": [], "teams": [], "venues": [], "league_teams": {}, "league_venues": {}, "league_cities": {}}

# For `flask run`
from flask import request
app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

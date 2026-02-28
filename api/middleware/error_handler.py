"""
error_handler.py — Centralized error handling for the Flask app.
"""
import logging
import traceback
from flask import jsonify

logger = logging.getLogger("criccric")


def register_error_handlers(app):
    """Register all error handlers on the Flask app."""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({"error": "Forbidden", "message": str(e)}), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": str(e)}), 404

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({
            "error": "Rate Limited",
            "message": "Too many requests. Please slow down.",
        }), 429

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "Something went wrong.",
        }), 500

    @app.errorhandler(503)
    def service_unavailable(e):
        return jsonify({
            "error": "Service Unavailable",
            "message": "Model not loaded. Run train_model.py first.",
        }), 503

"""
websocket_manager.py — Flask-SocketIO manager for live updates.

Pushes live scores, probability shifts, and match events to subscribed clients.
Background loop runs every 30 seconds to fetch and push updates.
"""

import logging
import threading
import time

logger = logging.getLogger("criccric")

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    logger.warning("Flask-SocketIO not installed. WebSockets disabled.")


class WebSocketManager:
    """Manages WebSocket connections and background push updates."""

    def __init__(self, app=None, live_service=None, model_mgr=None):
        self.socketio = None
        self.live_service = live_service
        self.model_mgr = model_mgr
        self.active_rooms = set()
        self._bg_thread = None
        self._stop_event = threading.Event()

        if app and HAS_SOCKETIO:
            self.init_app(app)

    def init_app(self, app):
        """Initialize with Flask app."""
        if not HAS_SOCKETIO:
            return

        # Initialize SocketIO (cors relaxed for dev)
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
        logger.info("WebSocket Manager initialised")

        # Register event handlers
        @self.socketio.on("connect")
        def handle_connect():
            logger.debug("Client connected via WebSocket")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            logger.debug("Client disconnected")

        @self.socketio.on("subscribe_match")
        def handle_subscribe(data):
            match_id = data.get("match_id")
            if match_id:
                join_room(match_id)
                self.active_rooms.add(match_id)
                logger.info(f"Client subscribed to match: {match_id}")
                # Immediately send an update if possible
                self._push_update(match_id)

        @self.socketio.on("unsubscribe_match")
        def handle_unsubscribe(data):
            match_id = data.get("match_id")
            if match_id:
                leave_room(match_id)
                logger.info(f"Client unsubscribed from match: {match_id}")

        # Start background updater
        self.start_background_task()

    def start_background_task(self):
        """Start the loop that pushes updates to active rooms."""
        if not self.socketio:
            return

        if self._bg_thread and self._bg_thread.is_alive():
            return

        self._stop_event.clear()
        self._bg_thread = self.socketio.start_background_task(self._update_loop)
        logger.info("Started WebSocket background update loop")

    def stop_background_task(self):
        """Stop the background loop."""
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=2)

    def _update_loop(self):
        """Infinite loop fetching live scores and emitting predictions."""
        while not self._stop_event.is_set():
            if not self.active_rooms:
                self.socketio.sleep(5)  # Sleep longer if no one is watching
                continue

            # Need to copy to avoid RuntimeError if set changes during iteration
            for match_id in list(self.active_rooms):
                try:
                    self._push_update(match_id)
                except Exception as e:
                    logger.error(f"Error pushing update for {match_id}: {e}")

            # Wait 30 seconds before next API hit
            self.socketio.sleep(30)

    def _push_update(self, match_id):
        """Fetch live score, run model, emit to room."""
        if not self.live_service or not self.model_mgr:
            return

        score_data = self.live_service.get_match_score(match_id)
        if not score_data or "error" in score_data:
            return

        # If match is over, we might want to tell clients to stop polling
        status = score_data.get("status", "").lower()
        if "won" in status or "abandoned" in status or "no result" in status:
            self.socketio.emit("match_complete", score_data, to=match_id)
            # We can optionally remove from active_rooms, but let clients unsubscribe
            return

        # Prepare features for prediction
        # (This requires live_service.get_live_features to be implemented)
        features_dict = self.live_service.get_live_features(match_id)
        if not features_dict:
            # Emit just the score if we can't build features
            self.socketio.emit("score_update", score_data, to=match_id)
            return

        # Make prediction
        pred_res = self.model_mgr.predict([features_dict])
        if "error" in pred_res:
            self.socketio.emit("score_update", score_data, to=match_id)
            return

        # Combine output
        payload = {
            "score": score_data,
            "prediction": pred_res["results"][0],
            "timestamp": time.time()
        }

        # Push to room
        self.socketio.emit("prediction_update", payload, to=match_id)
        logger.debug(f"Pushed prediction update for {match_id}")

    def emit_global(self, event_name, data):
        """Emit an event to all connected clients."""
        if self.socketio:
            self.socketio.emit(event_name, data)

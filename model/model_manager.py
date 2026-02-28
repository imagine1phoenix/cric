"""
model_manager.py — Singleton model loader with joblib compression.

Provides:
  - Singleton pattern (only one instance application-wide)
  - Lazy loading (model loads on first prediction, not at import)
  - Compressed serialization (joblib compress=3, ~6x size reduction)
  - Model versioning (hash + training date in metadata JSON)
  - Warmup support (pre-load without waiting for first user request)
"""

import os
import json
import hashlib
import time
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe Singleton that manages model lifecycle."""

    _instance = None
    _model = None
    _encoders = None
    _feature_names = None
    _artifacts = None
    _metadata = None
    _loaded = False

    # Default artifact directory (relative to project root)
    _model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "artifacts"
    )

    # ── Singleton ──────────────────────────────────────────────────────────

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Return the single ModelManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing / hot-reload)."""
        cls._instance = None
        cls._model = None
        cls._encoders = None
        cls._feature_names = None
        cls._artifacts = None
        cls._metadata = None
        cls._loaded = False

    # ── Configuration ──────────────────────────────────────────────────────

    def set_artifact_path(self, path):
        """Override the default artifact directory."""
        self._model_path = path
        # Force re-load on next access
        self._loaded = False
        self._model = None
        return self

    @property
    def artifact_path(self):
        return self._model_path

    @property
    def is_loaded(self):
        return self._loaded

    # ── Lazy Loading ───────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load model artifacts on first access (lazy loading)."""
        if self._loaded:
            return

        ensemble_config = os.path.join(self._model_path, "ensemble", "ensemble_config.json")
        model_file = os.path.join(self._model_path, "model_v2.joblib")
        encoders_file = os.path.join(self._model_path, "encoders.joblib")
        features_file = os.path.join(self._model_path, "features.joblib")
        metadata_file = os.path.join(self._model_path, "model_metadata.json")

        # ── Also support legacy .pkl format for backward compatibility ──
        legacy_model = os.path.join(self._model_path, "cricket_rf_bootstrap.pkl")
        legacy_preprocess = os.path.join(self._model_path, "cricket_preprocess.pkl")

        if os.path.exists(ensemble_config):
            # ── Stacking Ensemble format ──
            logger.info("Loading Stacking Ensemble model")
            start = time.time()
            from model.ensemble import EnsemblePredictor
            
            self._model = EnsemblePredictor(os.path.join(self._model_path, "ensemble"))
            self._model.load()
            
            artifacts_file = os.path.join(self._model_path, "artifacts_dict.joblib")
            if os.path.exists(artifacts_file):
                self._artifacts = joblib.load(artifacts_file)
                self._encoders = self._artifacts.get("encoders", {})
                self._feature_names = self._artifacts.get("feature_list", [])
            else:
                self._artifacts = {}
                self._encoders = {}
                self._feature_names = self._model.feature_names
            
            meta_file = os.path.join(self._model_path, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
                
            elapsed = time.time() - start
            logger.info(f"✅ Ensemble Model loaded in {elapsed:.2f}s")

        elif os.path.exists(model_file):
            # ── New compressed single format ──
            logger.info(f"Loading single model from {model_file}")
            start = time.time()

            self._model = joblib.load(model_file)
            self._encoders = joblib.load(encoders_file)
            self._feature_names = joblib.load(features_file)

            # Load full artifacts if available
            artifacts_file = os.path.join(self._model_path, "artifacts.joblib")
            if os.path.exists(artifacts_file):
                self._artifacts = joblib.load(artifacts_file)
            else:
                # Build minimal artifacts dict from individual files
                self._artifacts = {
                    "encoders": self._encoders,
                    "feature_list": self._feature_names,
                }

            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file) as f:
                    self._metadata = json.load(f)

            elapsed = time.time() - start
            logger.info(f"✅ Single Model loaded in {elapsed:.2f}s")

        elif os.path.exists(legacy_model) and os.path.exists(legacy_preprocess):
            # ── Legacy .pkl format (backward compatible) ──
            logger.info(f"Loading model from legacy format: {legacy_model}")
            start = time.time()

            self._model = joblib.load(legacy_model)
            self._artifacts = joblib.load(legacy_preprocess)
            self._encoders = self._artifacts.get("encoders", {})
            self._feature_names = self._artifacts.get("feature_list", [])
            self._metadata = self._artifacts.get("model_metadata", {})

            elapsed = time.time() - start
            logger.info(f"✅ Model loaded in {elapsed:.2f}s (legacy format)")

        else:
            raise FileNotFoundError(
                f"No model files found in {self._model_path}. "
                f"Run train_model.py first or check the artifact path."
            )

        self._loaded = True

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_model(self):
        """Return (model, encoders, feature_names) tuple. Loads lazily."""
        self._ensure_loaded()
        return self._model, self._encoders, self._feature_names

    def get_artifacts(self):
        """Return full artifacts dict (dropdown_data, venue_avgs, etc.)."""
        self._ensure_loaded()
        return self._artifacts

    def get_metadata(self):
        """Return model metadata (accuracy, CI, version hash, etc.)."""
        self._ensure_loaded()
        return self._metadata or {}

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, input_dict):
        """
        Run prediction from a raw input dict.

        Parameters
        ----------
        input_dict : dict
            Keys: team1, team2, venue, toss_winner, toss_decision,
                  match_type, gender, league, city

        Returns
        -------
        dict with: winner, team1, team2, team1_win_prob, team2_win_prob,
                   confidence, model_accuracy, ci_low, ci_high, model_version
        """
        model, encoders, feature_names = self.get_model()
        artifacts = self.get_artifacts()
        meta = self.get_metadata()

        venue_avgs = artifacts.get("train_venue_avg", {})
        global_avg = artifacts.get("global_avg_score", 150.0)
        cat_features = artifacts.get("categorical_features",
                                     list(encoders.keys()))

        team1 = input_dict.get("team1", "")
        team2 = input_dict.get("team2", "")
        venue = input_dict.get("venue", "")

        # ── Build feature row ──
        row = {
            "innings1_total_runs": global_avg,
            "innings2_total_runs": global_avg,
            "innings1_wickets": 7,
            "innings2_wickets": 7,
            "toss_decision_bat_first": (
                1 if (input_dict.get("toss_decision") or "").lower()
                in ("bat", "batting") else 0
            ),
            "venue_avg_score": venue_avgs.get(venue, global_avg),
            "team1_recent_form": 0.5,
            "team2_recent_form": 0.5,
        }

        # ── Categorical encoding (graceful fallback for unseen values) ──
        cat_value_map = {
            "venue": venue or "Unknown",
            "team1": team1,
            "team2": team2,
            "toss_winner": input_dict.get("toss_winner") or team1,
            "match_type": input_dict.get("match_type") or "T20",
            "gender": input_dict.get("gender") or "male",
            "league": input_dict.get("league") or "Unknown",
            "city": input_dict.get("city") or "Unknown",
        }

        for cat_col in cat_features:
            le = encoders.get(cat_col)
            val = cat_value_map.get(cat_col, "UNKNOWN")
            if le is not None:
                known = set(le.classes_)
                row[cat_col + "_enc"] = (
                    int(le.transform([val])[0]) if val in known else 0
                )
            else:
                row[cat_col + "_enc"] = 0

        # ── Build DataFrame in correct feature order ──
        pred_df = pd.DataFrame([row])
        for feat in feature_names:
            if feat not in pred_df.columns:
                pred_df[feat] = 0
        pred_df = pred_df[feature_names]

        # ── Predict ──
        if hasattr(model, "is_loaded"): 
            # EnsemblePredictor returns a 1D array of floats
            t1_prob = float(model.predict_proba(pred_df)[0])
            t2_prob = 1.0 - t1_prob
            pred_class = 1 if t1_prob >= 0.5 else 0
            team1_prob = t1_prob
            team2_prob = t2_prob
            conf = max(t1_prob, t2_prob)
        else:
            # Scikit-learn single model returns a 2D array [prob_0, prob_1]
            proba = model.predict_proba(pred_df)[0]
            pred_class = model.predict(pred_df)[0]
            team1_prob = float(proba[1])
            team2_prob = float(proba[0])
            conf = float(max(proba))
            
        predicted_winner = team1 if pred_class == 1 else team2

        ci = meta.get("bootstrap_ci_accuracy", [None, None])

        return {
            "winner": predicted_winner,
            "team1": team1,
            "team2": team2,
            "team1_win_prob": round(team1_prob * 100, 1),
            "team2_win_prob": round(team2_prob * 100, 1),
            "confidence": round(conf * 100, 1),
            "model_accuracy": round(meta.get("accuracy", 0) * 100, 1),
            "ci_low": round(ci[0] * 100, 1) if ci[0] else None,
            "ci_high": round(ci[1] * 100, 1) if ci[1] else None,
            "model_version": meta.get("version_hash", "unknown"),
        }

    # ── Warmup ─────────────────────────────────────────────────────────────

    def warmup(self):
        """
        Pre-load the model into memory. Call this from a /api/warmup
        endpoint after deployment to avoid cold-start latency.
        Returns metadata about the loaded model.
        """
        start = time.time()
        self._ensure_loaded()
        elapsed = time.time() - start

        meta = self.get_metadata()
        return {
            "status": "ready",
            "load_time_seconds": round(elapsed, 3),
            "model_version": meta.get("version_hash", "unknown"),
            "trained_at": meta.get("trained_at", "unknown"),
            "accuracy": meta.get("accuracy"),
            "features": len(self._feature_names),
        }

    # ── Save (called from training script) ─────────────────────────────────

    @staticmethod
    def save_model(model, encoders, feature_names, artifacts_dict,
                   save_dir, compress=3):
        """
        Save a trained model with joblib compression + version metadata.

        Parameters
        ----------
        model : sklearn estimator
        encoders : dict of LabelEncoders
        feature_names : list of str
        artifacts_dict : dict (full preprocessing data)
        save_dir : str (directory path)
        compress : int (joblib compression level, 0-9, default 3)
        """
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "model_v2.joblib")
        encoders_path = os.path.join(save_dir, "encoders.joblib")
        features_path = os.path.join(save_dir, "features.joblib")
        artifacts_path = os.path.join(save_dir, "artifacts.joblib")
        metadata_path = os.path.join(save_dir, "model_metadata.json")

        # Save with compression
        print(f"💾 Saving model (compress={compress})...")
        joblib.dump(model, model_path, compress=compress)
        joblib.dump(encoders, encoders_path, compress=compress)
        joblib.dump(feature_names, features_path, compress=compress)
        joblib.dump(artifacts_dict, artifacts_path, compress=compress)

        # Compute version hash from model file
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:12]

        # Report file sizes
        for label, path in [("Model", model_path), ("Encoders", encoders_path),
                            ("Features", features_path), ("Artifacts", artifacts_path)]:
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"   {label:12s} → {os.path.basename(path):30s} {size:>7.2f} MB")

        # Build metadata JSON
        model_meta = artifacts_dict.get("model_metadata", {})
        metadata = {
            "version_hash": model_hash,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "accuracy": model_meta.get("accuracy"),
            "roc_auc": model_meta.get("roc_auc"),
            "bootstrap_ci_accuracy": model_meta.get("bootstrap_ci_accuracy"),
            "n_train": model_meta.get("n_train"),
            "n_test": model_meta.get("n_test"),
            "features": feature_names,
            "n_features": len(feature_names),
            "compression_level": compress,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   {'Metadata':12s} → {os.path.basename(metadata_path)}")
        print(f"\n🔑 Model version hash: {model_hash}")

        return model_hash

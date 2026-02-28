"""
ensemble.py — Stacking Ensemble Model for Cricket Match Prediction.

Train base learners (RandomForest, XGBoost, LightGBM, LogisticRegression)
and a meta-learner (LogisticRegression) on out-of-fold predictions.
Chronological splits (TimeSeriesSplit) to prevent future data leakage.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("criccric")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    logger.warning(f"Could not import ML libraries: {e}")


class EnsemblePredictor:
    """Stacking Ensemble with Chronological Validation."""

    def __init__(self, artifacts_dir="model/artifacts/ensemble"):
        self.artifacts_dir = artifacts_dir
        self.base_models = {}
        self.meta_learner = None
        self.feature_names = None
        self.is_loaded = False
        os.makedirs(artifacts_dir, exist_ok=True)

    def train(self, df_features, target_col="winner_is_team1", n_splits=5):
        """Train the stacking ensemble."""
        logger.info("Starting Ensemble Training (Level 1 + Level 2)")
        
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col].astype(int)
        self.feature_names = list(X.columns)

        # Base models
        models = {
            "rf": RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "xgb": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, n_jobs=-1, random_state=42),
            "lgb": lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, n_jobs=-1, random_state=42),
            "lr": LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=42)
        }

        X_arr = X.values
        y_arr = y.values

        # Level 1 OOF predictions
        oof_preds = {name: np.zeros(len(y_arr)) for name in models}
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            logger.info(f"--- Fold {fold + 1}/{n_splits} ---")
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            for name, model in models.items():
                # Clone model for each fold
                import sklearn.base
                cloned_model = sklearn.base.clone(model)
                cloned_model.fit(X_train, y_train)
                preds = cloned_model.predict_proba(X_val)[:, 1]
                oof_preds[name][val_idx] = preds
                fold_auc = roc_auc_score(y_val, preds)
                logger.debug(f"{name} Fold {fold+1} AUC: {fold_auc:.4f}")

        # Train final base models on FULL data
        logger.info("Training final base models on full datset...")
        self.base_models = {}
        for name, model in models.items():
            model.fit(X_arr, y_arr)
            self.base_models[name] = model

        # Level 2 Meta-Leaner (train on OOF predictions)
        # We only train meta-learner on the validation folds
        # Exclude the very first train_idx which was never validated
        val_indices = np.concatenate([val_idx for _, val_idx in tscv.split(X_arr)])
        
        # Build meta-features matrix
        X_meta = np.column_stack([oof_preds[name][val_indices] for name in models])
        y_meta = y_arr[val_indices]

        logger.info("Training Level-2 Meta-Learner (LogisticRegression)...")
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_learner.fit(X_meta, y_meta)

        # Meta-learner evaluation
        meta_preds = self.meta_learner.predict_proba(X_meta)[:, 1]
        meta_auc = roc_auc_score(y_meta, meta_preds)
        meta_acc = accuracy_score(y_meta, (meta_preds >= 0.5).astype(int))
        logger.info(f"Meta-Learner OOF AUC: {meta_auc:.4f} | Acc: {meta_acc:.4f}")

        self.is_loaded = True
        self.save()
        return {"meta_auc": meta_auc, "meta_acc": meta_acc}

    def save(self):
        """Save ensemble models and config."""
        if not self.is_loaded:
            return

        for name, model in self.base_models.items():
            joblib.dump(model, os.path.join(self.artifacts_dir, f"{name}.joblib"), compress=3)
        
        joblib.dump(self.meta_learner, os.path.join(self.artifacts_dir, "meta_learner.joblib"), compress=3)

        config = {
            "trained_at": datetime.utcnow().isoformat(),
            "models": list(self.base_models.keys()),
            "feature_names": self.feature_names
        }
        with open(os.path.join(self.artifacts_dir, "ensemble_config.json"), "w") as f:
            json.dump(config, f)
            
        logger.info(f"Ensemble models saved to {self.artifacts_dir}")

    def load(self):
        """Load ensemble models."""
        config_path = os.path.join(self.artifacts_dir, "ensemble_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("Ensemble config not found")

        with open(config_path) as f:
            config = json.load(f)
            self.feature_names = config["feature_names"]

        for name in config["models"]:
            path = os.path.join(self.artifacts_dir, f"{name}.joblib")
            self.base_models[name] = joblib.load(path)

        self.meta_learner = joblib.load(os.path.join(self.artifacts_dir, "meta_learner.joblib"))
        self.is_loaded = True
        logger.info("Ensemble models loaded")

    def predict_proba(self, X_eval):
        """Generate ensemble prediction."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        if isinstance(X_eval, pd.DataFrame):
            X_arr = X_eval[self.feature_names].values
        else:
            X_arr = np.array(X_eval)

        # Get level-1 predictions
        base_preds = []
        for name in self.base_models:
            preds = self.base_models[name].predict_proba(X_arr)[:, 1]
            base_preds.append(preds)

        X_meta = np.column_stack(base_preds)
        
        # Level-2 prediction
        return self.meta_learner.predict_proba(X_meta)[:, 1]

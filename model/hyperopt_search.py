"""
hyperopt_search.py — Hyperparameter optimization using Optuna.

Search spaces for Random Forest, XGBoost, and LightGBM using
chronological cross-validation (TimeSeriesSplit) to maximize ROC-AUC.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger("criccric")

try:
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    logger.warning(f"Could not import optuna or ml libs: {e}")


class HyperoptSearch:
    """Optuna HPO for base models."""

    def __init__(self, n_trials=50, n_splits=3):
        self.n_trials = n_trials
        self.n_splits = n_splits

    def optimize_rf(self, X, y):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": -1,
                "random_state": 42
            }
            return self._cv_score(RandomForestClassifier(**params), X, y)

        return self._run_study("rf", objective)

    def optimize_xgb(self, X, y):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_jobs": -1,
                "random_state": 42
            }
            return self._cv_score(xgb.XGBClassifier(**params), X, y)

        return self._run_study("xgb", objective)

    def optimize_lgb(self, X, y):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "n_jobs": -1,
                "random_state": 42
            }
            return self._cv_score(lgb.LGBMClassifier(**params), X, y)

        return self._run_study("lgb", objective)

    def _cv_score(self, model, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, preds))
        return np.mean(aucs)

    def _run_study(self, name, objective):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        logger.info(f"[{name}] Best AUC: {study.best_value:.4f}")
        return study.best_params

    def run_all(self, df, target_col="winner_is_team1", output_file="model/artifacts/best_params.json"):
        """Run full hyperparameter optimization."""
        logger.info("Starting Hyperparameter Optimization...")
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        best_params = {
            "rf": self.optimize_rf(X, y),
            "xgb": self.optimize_xgb(X, y),
            "lgb": self.optimize_lgb(X, y)
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        logger.info(f"Hyperopt complete. Best params saved to {output_file}")
        return best_params

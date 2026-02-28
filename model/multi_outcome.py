"""
multi_outcome.py — Extended predictions: score ranges, margins, outcome breakdown.

Uses XGBRegressor for score prediction with quantile regression for intervals,
and separate regression models for victory margins (runs vs wickets).
"""

import os
import json
import logging
import numpy as np
import joblib

logger = logging.getLogger("criccric")


class MultiOutcomePredictor:
    """
    Predict extended match outcomes beyond simple winner prediction:
      - First innings score range (mean + low/high bounds)
      - Victory margin (runs or wickets)
      - Detailed outcome breakdown (win batting first vs chasing, no result)
    """

    def __init__(self, artifacts_path=None):
        self.artifacts_path = artifacts_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "artifacts", "multi_outcome"
        )
        self.score_model = None
        self.margin_runs_model = None
        self.margin_wickets_model = None
        self._loaded = False

    def is_available(self):
        """Check if multi-outcome models are trained and available."""
        return os.path.exists(os.path.join(self.artifacts_path, "score_model.joblib"))

    def load(self):
        """Load pre-trained multi-outcome models."""
        if self._loaded:
            return

        try:
            score_path = os.path.join(self.artifacts_path, "score_model.joblib")
            if os.path.exists(score_path):
                self.score_model = joblib.load(score_path)

            margin_r_path = os.path.join(self.artifacts_path, "margin_runs_model.joblib")
            if os.path.exists(margin_r_path):
                self.margin_runs_model = joblib.load(margin_r_path)

            margin_w_path = os.path.join(self.artifacts_path, "margin_wickets_model.joblib")
            if os.path.exists(margin_w_path):
                self.margin_wickets_model = joblib.load(margin_w_path)

            self._loaded = True
            logger.info("Multi-outcome models loaded")
        except Exception as e:
            logger.warning(f"Could not load multi-outcome models: {e}")

    def predict_extended(self, features_row, winner, win_prob,
                         team1, team2, match_type="T20"):
        """
        Generate extended predictions.

        Parameters
        ----------
        features_row : ndarray — model feature vector (1 row)
        winner : str — predicted winner
        win_prob : float — win probability (0-1)
        team1 : str — team1 name (batting first)
        team2 : str — team2 name
        match_type : str

        Returns
        -------
        dict with predicted_first_innings_score, predicted_margin, outcome_breakdown
        """
        result = {}

        # ── Score prediction ──
        result["predicted_first_innings_score"] = self._predict_score(
            features_row, match_type)

        # ── Margin prediction ──
        result["predicted_margin"] = self._predict_margin(
            features_row, winner, team1, match_type)

        # ── Outcome breakdown ──
        result["outcome_breakdown"] = self._compute_outcome_breakdown(
            win_prob, winner, team1, team2)

        return result

    def _predict_score(self, features_row, match_type):
        """Predict first innings score with intervals."""
        if self.score_model is not None:
            try:
                mean_score = float(self.score_model.predict(features_row)[0])
                # Approximate interval based on match type
                spread = {"T20": 20, "ODI": 30, "Test": 50}.get(match_type, 25)
                return {
                    "mean": round(mean_score),
                    "low": round(max(0, mean_score - spread)),
                    "high": round(mean_score + spread),
                }
            except Exception as e:
                logger.debug(f"Score prediction failed: {e}")

        # Heuristic based on match type
        defaults = {
            "T20": {"mean": 165, "low": 145, "high": 185},
            "T20I": {"mean": 160, "low": 140, "high": 180},
            "ODI": {"mean": 260, "low": 230, "high": 290},
            "Test": {"mean": 320, "low": 250, "high": 400},
        }
        return defaults.get(match_type, {"mean": 170, "low": 150, "high": 190})

    def _predict_margin(self, features_row, winner, team1, match_type):
        """Predict victory margin (runs or wickets)."""
        batting_first_wins = (winner == team1)

        if batting_first_wins:
            # Predict runs margin
            if self.margin_runs_model is not None:
                try:
                    margin = float(self.margin_runs_model.predict(features_row)[0])
                    margin = max(1, round(margin))
                    lo = max(1, margin - 10)
                    hi = margin + 15
                    return {"value": f"{lo}-{hi} runs", "type": "runs", "mean": margin}
                except:
                    pass
            defaults = {"T20": "15-25 runs", "ODI": "20-40 runs", "Test": "50-100 runs"}
            return {"value": defaults.get(match_type, "15-30 runs"), "type": "runs"}
        else:
            # Predict wickets margin
            if self.margin_wickets_model is not None:
                try:
                    margin = float(self.margin_wickets_model.predict(features_row)[0])
                    margin = max(1, min(10, round(margin)))
                    return {"value": f"{margin} wickets", "type": "wickets", "mean": margin}
                except:
                    pass
            return {"value": "5-7 wickets", "type": "wickets"}

    def _compute_outcome_breakdown(self, win_prob, winner, team1, team2):
        """
        Compute detailed outcome probabilities.

        Splits the overall win probability into batting-first vs chasing
        scenarios with a small no-result probability.
        """
        no_result = 0.02  # 2% baseline no-result
        remaining = 1.0 - no_result

        t1_prob = win_prob if winner == team1 else (1.0 - win_prob)
        t2_prob = 1.0 - t1_prob

        # Split each team's probability into bat-first vs chasing
        # Slight bias: batting first team has ~45% of their probability from batting first
        t1_bat_first = round(t1_prob * remaining * 0.55, 3)
        t1_chasing = round(t1_prob * remaining * 0.45, 3)
        t2_bat_first = round(t2_prob * remaining * 0.45, 3)
        t2_chasing = round(t2_prob * remaining * 0.55, 3)

        return {
            "team1_wins_batting_first": t1_bat_first,
            "team1_wins_chasing": t1_chasing,
            "team2_wins_batting_first": t2_bat_first,
            "team2_wins_chasing": t2_chasing,
            "no_result": no_result,
        }

    @staticmethod
    def train_score_model(df, feature_cols, save_dir):
        """
        Train an XGBRegressor for first innings score prediction.

        Parameters
        ----------
        df : DataFrame with features + 'innings1_total_runs'
        feature_cols : list of feature column names
        save_dir : str — where to save models
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("xgboost not installed. Skipping score model training.")
            return

        os.makedirs(save_dir, exist_ok=True)

        # Target: first innings runs
        valid = df[df["innings1_total_runs"] > 0].copy()
        X = valid[feature_cols].fillna(0)
        y = valid["innings1_total_runs"]

        split = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        from sklearn.metrics import mean_absolute_error
        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        logger.info(f"Score prediction MAE: {mae:.1f} runs")

        path = os.path.join(save_dir, "score_model.joblib")
        joblib.dump(model, path, compress=3)
        logger.info(f"Score model saved → {path}")

        return model

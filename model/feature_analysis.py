"""
feature_analysis.py — Advanced wrapper for interpreting features.

Computes Tree Importance, Permutation Importance, and SHAP Values.
Capable of auto-removing bottom-10% noisy features.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("criccric")

try:
    from sklearn.inspection import permutation_importance
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warning(f"Could not import feature analysis libs: {e}")


class FeatureAnalyzer:
    """Analyze feature importance and drop noisy features."""

    def __init__(self, model, feature_names, artifacts_dir="model/artifacts"):
        self.model = model
        self.feature_names = feature_names
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

    def generate_report(self, X_val, y_val):
        """Generate a combination of importances."""
        logger.info("Generating feature importance report...")
        report = {}

        # 1. Native Impurity (Gini) Importance
        if hasattr(self.model, "feature_importances_"):
            gini_imp = self.model.feature_importances_
            report["native_importance"] = {
                name: float(val) for name, val in zip(self.feature_names, gini_imp)
            }

        # 2. Permutation Importance
        logger.info("Computing Permutation Importance...")
        perm_result = permutation_importance(
            self.model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1
        )
        report["permutation_importance"] = {
            name: float(val) for name, val in zip(self.feature_names, perm_result.importances_mean)
        }

        # Save to JSON
        path = os.path.join(self.artifacts_dir, "feature_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        
        return report

    def plot_shap_summary(self, X_val, sample_size=1000):
        """Generate SHAP summary plot."""
        logger.info("Computing SHAP values for summary plot...")
        try:
            sample = X_val.sample(n=min(len(X_val), sample_size), random_state=42)
            
            # Using TreeExplainer for tree models
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] # For binary classification in some model types

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, sample, feature_names=self.feature_names, show=False)
            
            # Save
            path = os.path.join(self.artifacts_dir, "shap_summary.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"SHAP summary saved to {path}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP plot: {e}")

    def filter_noisy_features(self, report, drop_bottom_percent=10):
        """Returns a list of top features to keep."""
        if "permutation_importance" not in report: return self.feature_names
        
        pi = report["permutation_importance"]
        ranked = sorted(pi.items(), key=lambda x: x[1])
        
        drop_count = int(len(ranked) * (drop_bottom_percent / 100))
        noisy = [f[0] for f in ranked[:drop_count]]
        
        logger.info(f"Identifying bottom {drop_bottom_percent}% ({drop_count}) noisy features: {noisy}")
        
        keep = [f for f in self.feature_names if f not in noisy]
        return keep

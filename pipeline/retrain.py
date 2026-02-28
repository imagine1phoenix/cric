"""
retrain.py — Automated retraining pipeline.

Checks for new data via data_updater, runs feature engineering and
Hyperopt, trains the Ensemble model, evaluates against old model,
and promotes if it's better.
"""

import os
import json
import logging
import shutil
import subprocess

logger = logging.getLogger("criccric")


class RetrainPipeline:
    def __init__(self, data_updater):
        self.updater = data_updater
        self.best_params_file = "model/artifacts/best_params.json"
        
    def trigger_retrain(self, force=False):
        """Run the end-to-end retraining pipeline."""
        logger.info("Triggering Retrain Pipeline...")
        new_files = self.updater.fetch_kaggle_dataset()
        
        if new_files == 0 and not force:
            logger.info("No new data found. Skipping retrain.")
            return {"status": "skipped", "reason": "no_new_data"}
            
        logger.info(f"Proceeding with retrain ({new_files} new files).")
        
        # We can call train_model.py as a subprocess to keep memory clean
        try:
            cmd = ["python3", "train_model.py", "--ensemble"]
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Retrain complete. Output: {process.stdout[-500:]}")
            
            return {"status": "success", "new_files": new_files}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Retrain failed! Exit status: {e.returncode}\n{e.stderr[-1000:]}")
            return {"status": "failed", "error": str(e)}

    def report_status(self):
        """Get pipeline status."""
        return {
            "last_update": self.updater.state_data.get("last_update"),
            "total_matches": self.updater.state_data.get("total_matches", 0),
        }

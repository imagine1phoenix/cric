"""
data_updater.py — Automated script to download latest data from Cricsheet.

Can fetch Kaggle dataset or directly from Cricsheet (zip files).
Maintains state to only process new files.
"""

import os
import json
import logging
import zipfile
import shutil
import glob
from pathlib import Path

logger = logging.getLogger("criccric")


class DataUpdater:
    """Downloads and extracts new JSON files from Cricsheet/Kaggle."""

    def __init__(self, raw_data_dir="data/raw_json", state_file="pipeline/state.json"):
        self.raw_data_dir = raw_data_dir
        self.state_file = state_file
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.state_data = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"last_update": "", "total_matches": 0, "processed_files": []}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state_data, f, indent=4)

    def fetch_kaggle_dataset(self, dataset_name="veeralakrishna/cricsheet-a-match-by-match-dataset"):
        """Use kagglehub to pull the dataset."""
        logger.info(f"Checking for updates on Kaggle dataset: {dataset_name}")
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_name)
            logger.info(f"Dataset downloaded to {path}")
            
            # Find all JSON files and copy them to raw_data_dir
            json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
            
            new_files_count = 0
            for fpath in json_files:
                fname = os.path.basename(fpath)
                if fname not in self.state_data["processed_files"]:
                    shutil.copy(fpath, os.path.join(self.raw_data_dir, fname))
                    self.state_data["processed_files"].append(fname)
                    new_files_count += 1
            
            from datetime import datetime
            self.state_data["last_update"] = datetime.utcnow().isoformat()
            self.state_data["total_matches"] = len(self.state_data["processed_files"])
            self._save_state()
            
            logger.info(f"Added {new_files_count} new match files. Total files: {self.state_data['total_matches']}")
            return new_files_count
            
        except Exception as e:
            logger.error(f"Failed to fetch Kaggle dataset: {e}")
            return 0

    def download_cricsheet_zip(self, url="https://cricsheet.org/downloads/all_json.zip"):
        """Direct download from Cricsheet."""
        logger.info(f"Downloading directly from Cricsheet: {url}")
        try:
            import requests
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            
            zip_path = os.path.join(self.raw_data_dir, "downloads.zip")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Extracting zip...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.raw_data_dir)
                
            os.remove(zip_path)
            
            # Track newly extracted files
            import glob
            extracted_files = glob.glob(os.path.join(self.raw_data_dir, "*.json"))
            new_files_count = 0
            
            for fpath in extracted_files:
                fname = os.path.basename(fpath)
                if fname not in self.state_data["processed_files"]:
                    self.state_data["processed_files"].append(fname)
                    new_files_count += 1
            
            from datetime import datetime
            self.state_data["last_update"] = datetime.utcnow().isoformat()
            self.state_data["total_matches"] = len(self.state_data["processed_files"])
            self._save_state()
            
            logger.info(f"Extracted {new_files_count} new match files.")
            return new_files_count
            
        except Exception as e:
            logger.error(f"Failed to download from Cricsheet: {e}")
            return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    updater = DataUpdater()
    # Pull from kaggle by default
    updater.fetch_kaggle_dataset()

"""
kaggle_downloader.py — Batch downloader for Cricsheet Kaggle CSV Datasets.
"""

import os
import logging
import shutil
import kagglehub

logger = logging.getLogger("criccric")
logging.basicConfig(level=logging.INFO)

DATASETS = [
    "patrickb1912/ipl-complete-dataset-20082020",
    "jaykay12/odi-cricket-matches",
    "rajsengo/icc-mens-t20-world-cup",
    "madhav1993/womens-cricket",
    "faisaljanjua0555/bbl-complete-dataset",
    "faisaljanjua0555/psl-complete-dataset",
]

def download_all_datasets(raw_dir="data/raw_csv"):
    """Downloads Kaggle datasets and stages them into subdirectories."""
    os.makedirs(raw_dir, exist_ok=True)
    
    downloaded_paths = []
    
    for dataset in DATASETS:
        logger.info(f"Downloading Kaggle dataset: {dataset}...")
        try:
            # kagglehub isolates downloads in ~/.cache/kagglehub
            path = kagglehub.dataset_download(dataset)
            logger.info(f"Downloaded to: {path}")
            
            # Create a staging folder named after the dataset
            dataset_name = dataset.split("/")[-1]
            target_dir = os.path.join(raw_dir, dataset_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy all CSVs from the cache to the staging folder
            import glob
            csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            for csv_file in csv_files:
                shutil.copy(csv_file, os.path.join(target_dir, os.path.basename(csv_file)))
                
            logger.info(f"Staged {len(csv_files)} CSVs for {dataset_name} in {target_dir}")
            downloaded_paths.append(target_dir)
            
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")
            
    return downloaded_paths

if __name__ == "__main__":
    download_all_datasets()

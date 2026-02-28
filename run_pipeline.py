"""
run_pipeline.py — Entry point for the automated data collection and retraining.

Runs an APScheduler to:
1. Daily: Check for new matches via DataUpdater
2. Weekly: Trigger the RetrainPipeline if enough new data is found.
"""

import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler

from pipeline.data_updater import DataUpdater
from pipeline.retrain import RetrainPipeline
from services.logger import setup_logging

logger = logging.getLogger("criccric")

def setup_jobs():
    updater = DataUpdater()
    pipeline = RetrainPipeline(updater)

    def daily_check():
        logger.info("[Scheduler] Running daily data check...")
        updater.fetch_kaggle_dataset()

    def weekly_retrain():
        logger.info("[Scheduler] Running weekly retrain pipeline...")
        pipeline.trigger_retrain(force=False)

    scheduler = BackgroundScheduler()
    # Check for new data every day at 02:00 AM
    scheduler.add_job(daily_check, 'cron', hour=2, minute=0)
    # Retrain every Sunday at 04:00 AM
    scheduler.add_job(weekly_retrain, 'cron', day_of_week='sun', hour=4, minute=0)
    
    scheduler.start()
    logger.info("APScheduler started. Daily check @ 02:00, Weekly retrain @ Sun 04:00.")
    return scheduler

if __name__ == "__main__":
    setup_logging(log_level="INFO")
    logger.info("Starting CricPredict background pipeline service...")
    sched = setup_jobs()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline service...")
        sched.shutdown()

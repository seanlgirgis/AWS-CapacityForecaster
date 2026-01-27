# =============================================================================
# src/modules/module_02_data_load.py
#
# PURPOSE:
# SECOND module in the AWS-CapacityForecaster pipeline.
# Loads raw synthetic data from module_01, performs basic validation
# (duplicates, types, history length, missing rates), and saves validated data.
#
# Recreates Citi-style data ingestion:
# - Dynamic latest raw file loading
# - Duplicate timestamp removal per server
# - Basic stats logging for quality assurance
# - Optional SQLAlchemy for DB-like queries (simulating Oracle backup access)
#
# ROLE IN PIPELINE:
# module_01 (raw/*.parquet) → THIS → intermediate/validated_server_metrics.parquet
#
# OUTPUT GUARANTEES:
# - Validated Parquet → data/scratch/intermediate/validated_server_metrics.parquet
# - Summary JSON → data/scratch/reports/summaries/module_02_summary.json
# - Detailed logging → logs/module_02_data_load.log (rotated)
#
# USAGE:
# python -m src.modules.module_02_data_load --env local
#
# =============================================================================

import logging
import argparse
import json
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
import os

import pandas as pd

from src.utils.config import load_config, validate_config
from src.utils.data_utils import load_from_s3_or_local, save_to_s3_or_local, save_processed_data

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(console_handler)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "module_02_data_load.log"

file_handler = TimedRotatingFileHandler(
    filename=log_file,
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(file_handler)

logger.info("Logging initialized — console + file (logs/module_02_data_load.log)")

# =============================================================================
# Main Logic
# =============================================================================
def main_process(config):
    logger.info("=== MODULE 02 : Data Loading & Basic Validation ===")

    # Load latest raw file dynamically
    raw_dir = Path(config.get('paths', {}).get('local_data_dir', 'data/scratch')) / "raw"
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No raw parquet files found in data/scratch/raw/")
    latest_file = max(parquet_files, key=os.path.getmtime)
    logger.info(f"Loading latest raw data: {latest_file.name}")

    df = load_from_s3_or_local(config, prefix="raw/", filename=latest_file.name)
    if df is None:
        raise FileNotFoundError(f"Failed to load {latest_file.name}")

    # Basic validation & cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['server_id', 'timestamp'])
    duplicates = df.duplicated(subset=['server_id', 'timestamp']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps per server — removing.")
        df = df.drop_duplicates(subset=['server_id', 'timestamp'])

    # Per-server history check
    server_counts = df.groupby('server_id')['timestamp'].count()
    min_history = server_counts.min()
    if min_history < 365:
        logger.warning(f"Shortest server history: {min_history} days — check data generation.")

    # Missing rates per metric
    missing_rates = df.isnull().mean() * 100
    logger.info("Missing rates (%):\n" + missing_rates.to_string())

    # Basic stats
    stats = df.describe()
    logger.info("Data stats:\n" + stats.to_string())

    # Save validated data
    save_path = save_processed_data(
        df, config,
        prefix="intermediate/",
        filename="validated_server_metrics.parquet"
    )
    logger.info(f"Saved validated data to {save_path}")

    # Save summary
    # Save summary — force all values to native Python types for JSON
    summary = {
        "loaded_at": pd.Timestamp.now().isoformat(),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "servers": int(len(server_counts)),
        "min_max_dates": [
            df['timestamp'].min().isoformat(),
            df['timestamp'].max().isoformat()
        ],
        "missing_rates": {k: float(v) for k, v in missing_rates.items()},
        "duplicates_removed": int(duplicates)
    }

    save_to_s3_or_local(
        json.dumps(summary, indent=2),
        config,
        prefix="reports/summaries/",
        filename="module_02_summary.json"
    )
    logger.info("✔ Module 02 completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 02 — Data Loading & Validation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.env:
        if 'execution' not in config:
            config['execution'] = {}
        config['execution']['mode'] = args.env
    validate_config(config)

    main_process(config)
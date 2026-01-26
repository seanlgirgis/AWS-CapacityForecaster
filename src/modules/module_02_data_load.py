# =============================================================================
# src/modules/module_02_data_load.py
#
# PURPOSE:
#   SECOND module in AWS-CapacityForecaster pipeline.
#   Loads raw Parquet from module_01, performs schema/dtype/quality validation,
#   and delivers a clean, sorted DataFrame for ETL & feature engineering.
#
#   Defensive ingestion layer — fails fast on critical issues, warns on minor gaps.
#
# ROLE IN PIPELINE:
#   module_01 → raw/*.parquet → THIS → validated DataFrame → module_03_etl_feature_eng
#
# OUTPUT GUARANTEES:
#   - pd.DataFrame with exact schema, sorted, timestamp as datetime64
#   - Staged Parquet copy (prefix "staged/")
#   - JSON summary for audit
#   - Persistent log file: logs/module_02_data_load.log (rotates daily)
#
# CONFIG EXPECTATIONS:
#   storage.use_s3, storage.raw_prefix, storage.local_base_path
#   validation.max_missing_rate (e.g. 5.0)
#   data.num_servers (for cross-check)
#
# USAGE (after env_setter.ps1):
#   python -m src.modules.module_02_data_load --env local
#
# =============================================================================

import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

import pandas as pd

from src.utils.config import load_config, validate_config
from src.utils.data_utils import (
    load_from_s3_or_local,
    save_to_s3_or_local,
    save_processed_data
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler (always on)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(console_handler)

# File handler — persistent log with daily rotation (keeps last 7 days)
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


def load_and_validate_data(config: dict) -> pd.DataFrame:
    """Load raw parquet (local for now), validate, return trusted df."""
    storage = config.get('storage', {})
    data_cfg = config.get('data', {})
    val_cfg = config.get('validation', {})

    local_base = Path(storage.get('local_base_path', 'data/scratch'))
    raw_dir = local_base / storage.get('raw_prefix', 'raw')

    # Fixed filename from module_01 (update if date range changes)
    filename = "raw_server_metrics_20220101_to_20251231.parquet"
    full_path = raw_dir / filename

    if not full_path.exists():
        raise FileNotFoundError(f"Raw file not found: {full_path}")

    logger.info(f"Loading local Parquet: {full_path}")

    df = pd.read_parquet(full_path)

    # Schema validation
    expected_cols = [
        'server_id', 'timestamp', 'business_unit', 'region', 'criticality',
        'cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95'
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Timestamp fix
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().any():
        logger.warning("Some timestamps coerced to NaT")

    # Quality checks
    num_servers = df['server_id'].nunique()
    expected_num = data_cfg.get('num_servers', 120)
    if num_servers != expected_num:
        logger.warning(f"Server count mismatch: expected {expected_num}, got {num_servers}")

    gaps = df.groupby('server_id')['timestamp'].apply(
        lambda x: (x.sort_values().diff().dt.days > 1).sum()
    ).sum()
    if gaps > 0:
        logger.warning(f"{gaps} date gaps found (should be 0 in synthetic data)")

    numeric = ['cpu_p95','mem_p95','disk_p95','net_in_p95','net_out_p95']
    missing_rates = df[numeric].isna().mean() * 100
    max_allowed = val_cfg.get('max_missing_rate', 5.0)
    for col, rate in missing_rates.items():
        if rate > max_allowed:
            raise ValueError(f"{col} missing {rate:.2f}% > {max_allowed}% allowed")

    df = df.sort_values(['server_id', 'timestamp']).reset_index(drop=True)
    return df


def main_process(config: dict):
    logger.info("=== MODULE 02 : Data Loading & Validation ===")

    df = load_and_validate_data(config)

    summary = {
        "module": "02_data_load",
        "loaded_at": datetime.now().isoformat(),
        "rows": len(df),
        "unique_servers": df['server_id'].nunique(),
        "date_range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
        "missing_rates_pct": {col: round(rate, 3) for col, rate in (df[['cpu_p95','mem_p95','disk_p95','net_in_p95','net_out_p95']].isna().mean()*100).items()},
        "validation_passed": True
    }

    # Save staged Parquet using shared utility
    min_d = df['timestamp'].min().strftime('%Y%m%d')
    max_d = df['timestamp'].max().strftime('%Y%m%d')
    staged_fn = f"validated_server_metrics_{min_d}_to_{max_d}.parquet"
    save_processed_data(
        df=df,
        config=config,
        prefix="staged/",
        filename=staged_fn
    )
    logger.info("Staged file saved successfully")

    # Save JSON summary using shared utility
    save_to_s3_or_local(
        content=json.dumps(summary, indent=2),
        config=config,
        prefix="reports/summaries/",
        filename="module_02_summary.json"
    )

    logger.info(f"Validated shape: {df.shape} | Servers: {df['server_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info("Missing rates:\n" + pd.Series(summary['missing_rates_pct']).to_string())
    logger.info(f"Summary saved: data/scratch/reports/summaries/module_02_summary.json")
    logger.info("✔ Module 02 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 02 — Load & validate raw data")
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
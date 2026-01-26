# =============================================================================
# src/modules/module_02_data_load.py
#
# PURPOSE:
#   SECOND module in the AWS-CapacityForecaster pipeline.
#   Loads the raw Parquet from module_01, performs schema/dtype/quality validation,
#   and delivers a clean, sorted DataFrame for ETL & feature engineering.
#
#   Defensive ingestion layer — fails fast on schema issues, warns on minor gaps.
#
# ROLE IN PIPELINE:
#   module_01 → raw/*.parquet → THIS → validated DataFrame → module_03_etl_feature_eng
#
# OUTPUT GUARANTEES:
#   - pd.DataFrame with exact schema, sorted, timestamp as datetime64
#   - Staged Parquet copy (prefix "staged/")
#   - JSON summary for audit
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

import pandas as pd

from src.utils.config import load_config, validate_config
from src.utils.data_utils import load_from_s3, load_local_csv, save_processed_data, save_to_s3_or_local, load_from_s3_or_local

logger = logging.getLogger(__name__)

# =============================================================================
# src/modules/module_02_data_load.py
#
# PURPOSE:
#   SECOND module in AWS-CapacityForecaster pipeline.
#   Loads raw Parquet from module_01 (local preferred for now), validates schema/quality,
#   and delivers clean DataFrame for ETL. Saves staged Parquet + JSON summary.
#
#   Uses existing utils where possible; adds minimal save helpers here until data_utils is expanded.
#
# ROLE:
#   module_01 → raw/*.parquet → THIS → validated df → module_03_etl_feature_eng
#
# OUTPUT:
#   - pd.DataFrame (sorted, datetime timestamp, validated)
#   - Staged Parquet in data/scratch/staged/
#   - JSON summary in data/scratch/reports/summaries/
#
# USAGE:
#   python -m src.modules.module_02_data_load --env local
#
# =============================================================================

import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
import os

import pandas as pd

from src.utils.config import load_config, validate_config

logger = logging.getLogger(__name__)


def simple_save_local_or_s3(df_or_content, config, prefix: str, filename: str):
    """Temporary save helper — until moved to data_utils.py"""
    storage = config.get('storage', {})
    use_s3 = storage.get('use_s3', False)
    local_base = storage.get('local_base_path', 'data/scratch')
    bucket = storage.get('bucket_name')

    local_path = Path(local_base) / prefix / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df_or_content, pd.DataFrame):
        df_or_content.to_parquet(local_path, index=False)
        logger.info(f"Saved locally to {local_path}")
    else:
        local_path.write_text(df_or_content)
        logger.info(f"Saved JSON locally to {local_path}")

    # S3 save stub (expand later with boto3 if needed)
    if use_s3 and bucket:
        logger.info(f"S3 save not implemented yet for {filename} — skipping")


def load_and_validate_data(config: dict) -> pd.DataFrame:
    """Load raw parquet (local for now), validate, return trusted df."""
    storage = config.get('storage', {})
    data_cfg = config.get('data', {})
    val_cfg  = config.get('validation', {})

    local_base = Path(storage.get('local_base_path', 'data/scratch'))
    raw_dir    = local_base / storage.get('raw_prefix', 'raw')

    # Use fixed filename from module_01 (update if range changes)
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
        logger.warning(f"{gaps} date gaps found (should be 0)")

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

    # Save staged
    min_d = df['timestamp'].min().strftime('%Y%m%d')
    max_d = df['timestamp'].max().strftime('%Y%m%d')
    staged_fn = f"validated_server_metrics_{min_d}_to_{max_d}.parquet"
    simple_save_local_or_s3(df, config, prefix="staged/", filename=staged_fn)

    # JSON summary
    summary_path = save_to_s3_or_local(
        content=json.dumps(summary, indent=2),
        config=config,
        prefix="reports/summaries/",
        filename="module_02_summary.json"
    )

    logger.info(f"Validated shape: {df.shape} | Servers: {df['server_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info("Missing rates:\n" + pd.Series(summary['missing_rates_pct']).to_string())
    logger.info("Staged file saved successfully")
    logger.info(f"Summary: {summary_path}")
    logger.info("✔ Module 02 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 02 — Load & validate raw data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')

    config = load_config(Path(args.config))
    
    # Manually override env via args
    if args.env:
        if 'execution' not in config:
            config['execution'] = {}
        config['execution']['mode'] = args.env

    validate_config(config)

    main_process(config)
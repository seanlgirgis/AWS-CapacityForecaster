# =============================================================================
# src/modules/module_02_data_load.py
#
# PURPOSE:
#   This is the SECOND module in the modular pipeline.
#   It is responsible for safely loading the raw Parquet file produced by
#   module_01_data_generation.py, performing rigorous validation, and
#   delivering a clean, trusted DataFrame for downstream ETL/feature engineering.
#
# ROLE IN THE PIPELINE:
#   module_01 → raw Parquet → THIS MODULE → validated DataFrame → module_03_etl_feature_eng
#
# DESIGN PHILOSOPHY:
#   - Prefer S3 when use_s3=true, else local scratch folder
#   - Support both "specific filename" and "latest file in prefix" modes
#   - Fail fast on critical validation errors
#   - Always produce rich logging + JSON summary for auditability
#   - Make output predictable and reusable by next modules
#
# EXPECTED INPUT FROM MODULE 01:
#   File: raw_server_metrics_YYYYMMDD_to_YYYYMMDD.parquet
#   Columns (exact): server_id, timestamp, business_unit, region, criticality,
#                    cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95
#   Shape: (num_servers × num_days, 10)
#   Missing rate: ~0.5–2% on numeric columns only
#
# USAGE:
#   python -m src.modules.module_02_data_load --env local
#
# =============================================================================

import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime

import pandas as pd

# Project root + src import setup (same pattern as module_01)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import load_config, validate_config
from utils.data_utils import load_from_s3_or_local, save_processed_data

logger = logging.getLogger(__name__)


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Main loading function with full validation.
    Returns clean DataFrame or raises descriptive error.
    """
    storage = config.get('storage', {})
    data_cfg = config.get('data', {})
    validation = config.get('validation', {})

    use_s3 = storage.get('use_s3', False)
    raw_prefix = storage.get('raw_prefix', 'raw/')
    local_base = storage.get('local_base_path', 'data/scratch')

    # ────────────────────────────────────────────────
    # 1. Determine which file to load
    # ────────────────────────────────────────────────
    target_filename = storage.get('raw_filename')   # optional: specific file
    if not target_filename:
        # Auto-detect latest parquet in raw prefix (most common case)
        # Pseudo-logic:
        #   if use_s3:
        #       list objects in bucket/raw_prefix/*.parquet → pick newest by LastModified
        #   else:
        #       list local files in local_base/raw_prefix/*.parquet → pick newest by mtime
        target_filename = "raw_server_metrics_20220101_to_20251231.parquet"  # default from module_01

    logger.info(f"Loading raw data from {'S3' if use_s3 else 'local'} → {target_filename}")

    # ────────────────────────────────────────────────
    # 2. Load the Parquet file
    # ────────────────────────────────────────────────
    df = load_from_s3_or_local(
        config=config,
        prefix=raw_prefix,
        filename=target_filename
    )

    if df is None or df.empty:
        raise FileNotFoundError(f"Failed to load raw data file: {target_filename}")

    # ────────────────────────────────────────────────
    # 3. Schema & dtype validation
    # ────────────────────────────────────────────────
    expected_columns = [
        'server_id', 'timestamp', 'business_unit', 'region', 'criticality',
        'cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95'
    ]

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Convert timestamp to datetime (robust)
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # ────────────────────────────────────────────────
    # 4. Basic quality checks
    # ────────────────────────────────────────────────
    num_servers = df['server_id'].nunique()
    expected_servers = data_cfg.get('num_servers', 120)
    if num_servers != expected_servers:
        logger.warning(f"Server count mismatch: expected {expected_servers}, got {num_servers}")

    # Date continuity per server (pseudo-logic):
    #   for each server:
    #       dates = df[df.server_id==srv]['timestamp'].sort_values()
    #       diff = dates.diff().dt.days
    #       if any(diff > 1): gaps exist
    date_gaps = df.groupby('server_id')['timestamp'].apply(lambda x: (x.sort_values().diff().dt.days > 1).any()).sum()
    if date_gaps > 0:
        logger.warning(f"Found date gaps in {date_gaps} servers")

    # Missing rate validation
    max_missing_allowed = validation.get('max_missing_rate', 5.0)
    numeric_cols = ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
    missing_rates = df[numeric_cols].isna().mean() * 100

    for col, rate in missing_rates.items():
        if rate > max_missing_allowed:
            raise ValueError(f"Missing rate for {col} too high: {rate:.2f}% (max allowed {max_missing_allowed}%)")

    # ────────────────────────────────────────────────
    # 5. Final cleaning & sorting
    # ────────────────────────────────────────────────
    df = df.sort_values(['server_id', 'timestamp']).reset_index(drop=True)

    return df


def main_process(config: dict):
    """Orchestrates loading, validation, summary generation and optional staging."""
    logger.info("=== MODULE 02 : Data Loading & Validation ===")

    df = load_raw_data(config)

    # ─── Generate quality summary ───
    summary = {
        "module": "02_data_load",
        "loaded_at": datetime.now().isoformat(),
        "rows": len(df),
        "unique_servers": df['server_id'].nunique(),
        "date_range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
        "missing_rates": {col: round(rate, 3) for col, rate in (df[['cpu_p95','mem_p95','disk_p95','net_in_p95','net_out_p95']].isna().mean()*100).items()},
        "validation_passed": True
    }

    # ─── Optional: save validated/staged copy ───
    staged_filename = f"validated_server_metrics_{df['timestamp'].min().strftime('%Y%m%d')}_to_{df['timestamp'].max().strftime('%Y%m%d')}.parquet"
    staged_path = save_processed_data(
        df=df,
        config=config,
        prefix="staged/",
        filename=staged_filename
    )

    # ─── Save summary ───
    summary_path = save_processed_data(
        content=json.dumps(summary, indent=2),
        config=config,
        prefix="reports/summaries/",
        filename="module_02_summary.json"
    )

    # ─── Final logging ───
    logger.info(f"Loaded & validated shape: {df.shape}")
    logger.info(f"Servers: {df['server_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info("Missing rates:\n" + pd.Series(summary['missing_rates']).round(3).to_string())
    logger.info(f"Staged file: {staged_path}")
    logger.info(f"Summary: {summary_path}")
    logger.info("✔ Module 02 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 02 — Load & validate raw synthetic data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')

    config = load_config(args.config, env=args.env)
    validate_config(config)

    main_process(config)
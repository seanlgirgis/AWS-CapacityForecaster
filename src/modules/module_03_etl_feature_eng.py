# =============================================================================
# src/modules/module_03_etl_feature_eng.py
#
# PURPOSE:
#   THIRD module in the AWS-CapacityForecaster pipeline.
#   Transforms the clean validated panel data from module_02 into a rich,
#   ML-ready feature matrix that captures Citi-style capacity telemetry patterns.
#
#   Implements the exact feature engineering playbook that drove 20-30% accuracy
#   gains in the original Citi Prophet + scikit-learn ensemble models.
#
# KEY FEATURES IMPLEMENTED:
#   • Linear imputation per server (group-by interpolate)
#   • Z-score-based outlier capping (Winsorization using 30-day rolling stats)
#   • Lags (configurable: default [1,3,7,14,30] days)
#   • Rolling statistics (mean, std, min, max) — configurable windows
#   • Calendar & business-cycle flags (weekend, EOQ window, US holidays)
#   • Metadata one-hot encoding (business_unit, region, criticality)
#   • Trend features (days_since_start, cumulative sums per metric)
#   • Precise per-server warm-up NaN removal (first max(lags,windows) rows)
#   • Full config-driven behavior with sensible defaults
#   • Persistent logging to file + console
#
# ROLE IN PIPELINE:
#   module_02 (staged/validated_*.parquet) → THIS → processed_features_*.parquet → module_04_model_training
#
# OUTPUT GUARANTEES:
#   - Clean DataFrame with no NaNs (after warm-up drop & final cleanup)
#   - ~90–120 columns depending on config
#   - Saved to processed/processed_features_YYYYMMDD_to_YYYYMMDD.parquet
#   - JSON summary with feature breakdown
#   - Log file: logs/module_03_etl_feature_eng.log (rotates daily)
#
# USAGE:
#   python -m src.modules.module_03_etl_feature_eng --env local
#
# =============================================================================

import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

import pandas as pd
import numpy as np
import holidays

from src.utils.config import load_config, validate_config
from src.utils.data_utils import load_from_s3_or_local, save_to_s3_or_local, save_processed_data

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler (always on)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(console_handler)

# File handler — persistent log with daily rotation (keeps last 7 days)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "module_03_etl_feature_eng.log"

file_handler = TimedRotatingFileHandler(
    filename=log_file,
    when='midnight',        # Rotate at midnight
    interval=1,
    backupCount=7,          # Keep 7 days of logs
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(file_handler)

logger.info("Logging initialized — console + file (logs/module_03_etl_feature_eng.log)")

# =============================================================================
# Feature Engineering Core
# =============================================================================

def perform_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Core feature engineering function.
    Applies imputation, outlier handling, lags, rolling stats, calendar flags,
    metadata encoding, trend features, and precise warm-up cleanup.

    All major behaviors are driven by the 'feature_engineering' section of config.
    """
    feat_cfg = config.get('feature_engineering', {})
    numeric_metrics = ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']

    logger.info("Starting Feature Engineering...")
    logger.info(f"Input shape: {df.shape}. Columns: {len(df.columns)}")

    # 1. Defensive sort (ensure correct order for groupby & time-based ops)
    df = df.sort_values(['server_id', 'timestamp']).reset_index(drop=True)

    # 2. Imputation — per-server to respect time-series continuity
    method = feat_cfg.get('impute_method', 'linear')
    logger.info(f"Imputing missing values using method: {method}")
    if method == 'linear':
        df[numeric_metrics] = df.groupby('server_id')[numeric_metrics].transform(
            lambda g: g.interpolate(method='linear', limit_direction='both')
        )
    elif method == 'forward_fill':
        df[numeric_metrics] = df.groupby('server_id')[numeric_metrics].transform(
            lambda g: g.fillna(method='ffill').fillna(method='bfill')
        )
    else:
        # Fallback — conservative fill
        df[numeric_metrics] = df[numeric_metrics].fillna(method='ffill').fillna(0)

    # 3. Outlier capping — rolling z-score Winsorization (prevents extreme values skewing models)
    if feat_cfg.get('handle_outliers', True):
        z_thresh = feat_cfg.get('outlier_z_threshold', 4.0)
        logger.info(f"Capping outliers using rolling z-score (threshold={z_thresh})...")
        for col in numeric_metrics:
            grouped = df.groupby('server_id')[col]
            roll_mean = grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())
            roll_std  = grouped.transform(lambda x: x.rolling(30, min_periods=1).std())
            z_score = (df[col] - roll_mean) / roll_std
            extreme_mask = abs(z_score) > z_thresh
            df.loc[extreme_mask, col] = roll_mean[extreme_mask] + np.sign(z_score[extreme_mask]) * z_thresh * roll_std[extreme_mask]

    # 4. Calendar & business-cycle features — critical for banking seasonality
    logger.info("Generating calendar features (EOQ, Holidays, DayOfWeek)...")
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']      = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # EOQ window: last 7 days of Mar/Jun/Sep/Dec — strong signal in financial data
    if feat_cfg.get('add_eoq_flag', True):
        days_in_month = df['timestamp'].dt.days_in_month
        is_q_end_month = df['month'].isin([3, 6, 9, 12])
        is_end_window  = df['timestamp'].dt.day >= (days_in_month - 6)
        df['is_eoq_window'] = (is_q_end_month & is_end_window).astype(int)

    # US holidays — banking often quieter on federal holidays
    if feat_cfg.get('add_holiday_flag', True):
        years = range(df['timestamp'].dt.year.min(), df['timestamp'].dt.year.max() + 1)
        us_holidays = holidays.US(years=years)
        df['is_holiday'] = df['timestamp'].dt.date.isin(us_holidays).astype(int)

    # 5. Lag features — capture autocorrelation (very important for time-series)
    lags = feat_cfg.get('lags_days', [1, 3, 7, 14, 30])
    if lags:
        logger.info(f"Generating lags: {lags}")
        for col in numeric_metrics:
            for lag in lags:
                df[f"{col}_lag_{lag}d"] = df.groupby('server_id')[col].shift(lag)

    # 6. Rolling statistics — capture recent trends & volatility
    windows = feat_cfg.get('rolling_windows_days', [7, 30, 90])
    if windows:
        logger.info(f"Generating rolling stats (mean, std, min, max): {windows}")
        for col in numeric_metrics:
            grouped = df.groupby('server_id')[col]
            for w in windows:
                df[f"{col}_roll_mean_{w}d"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
                df[f"{col}_roll_std_{w}d"]  = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
                df[f"{col}_roll_min_{w}d"]  = grouped.transform(lambda x: x.rolling(w, min_periods=1).min())
                df[f"{col}_roll_max_{w}d"]  = grouped.transform(lambda x: x.rolling(w, min_periods=1).max())

    # 7. Trend features — capture long-term growth/drift
    if feat_cfg.get('add_trend_features', True):
        min_date = df['timestamp'].min()
        df['days_since_start'] = (df['timestamp'] - min_date).dt.days

        # Cumulative sums — proxy for cumulative load / wear
        for col in numeric_metrics:
            df[f"{col}_cumulative"] = df.groupby('server_id')[col].cumsum()

    # 8. Metadata one-hot encoding — server heterogeneity
    if feat_cfg.get('encode_metadata', True):
        logger.info("One-hot encoding metadata...")
        meta_cols = ['business_unit', 'region', 'criticality']
        df = pd.get_dummies(df, columns=meta_cols, prefix=['bu', 'reg', 'crit'], dtype=int)

    # 9. Precise warm-up NaN removal — drop initial rows with NaN features per server
    # We take the max lookback from lags + rolling windows
    all_lookbacks = feat_cfg.get('lags_days', []) + feat_cfg.get('rolling_windows_days', [])
    max_lookback = max(all_lookbacks) if all_lookbacks else 1
    logger.info(f"Dropping first {max_lookback} rows per server to remove warm-up NaNs...")
    df['row_num'] = df.groupby('server_id').cumcount() + 1
    df = df[df['row_num'] > max_lookback].drop(columns='row_num').reset_index(drop=True)

    # 10. Final safety cleanup — remove any remaining inf/NaN (should be none)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"Final shape after cleanup: {df.shape}")
    return df


def main_process(config: dict):
    logger.info("=== MODULE 03 : ETL & Feature Engineering ===")

    # Load validated data from module_02
    source_filename = "validated_server_metrics_20220101_to_20251231.parquet"
    logger.info(f"Loading validated data: {source_filename}")
    df = load_from_s3_or_local(config, prefix="staged/", filename=source_filename)

    if df is None or df.empty:
        raise FileNotFoundError(f"Could not load staged file: {source_filename}. Run Module 02 first.")

    # Perform feature engineering
    df_features = perform_feature_engineering(df, config)

    # Save processed features
    min_d = df_features['timestamp'].min().strftime('%Y%m%d')
    max_d = df_features['timestamp'].max().strftime('%Y%m%d')
    output_filename = f"processed_features_{min_d}_to_{max_d}.parquet"

    save_path = save_processed_data(
        df=df_features,
        config=config,
        prefix="processed/",
        filename=output_filename
    )

    # Generate & save summary
    summary = {
        "module": "03_etl_feature_eng",
        "processed_at": datetime.now().isoformat(),
        "input_rows": len(df),
        "output_rows": len(df_features),
        "columns": len(df_features.columns),
        "numeric_metrics": ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95'],
        "lag_features_count": len([c for c in df_features.columns if '_lag_' in c]),
        "rolling_features_count": len([c for c in df_features.columns if '_roll_' in c]),
        "source_file": source_filename
    }

    summary_path = save_to_s3_or_local(
        content=json.dumps(summary, indent=2),
        config=config,
        prefix="reports/summaries/",
        filename="module_03_summary.json"
    )

    logger.info(f"Features saved: {save_path}")
    logger.info(f"Summary saved: {summary_path}")
    logger.info("✔ Module 03 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 03 — ETL & Feature Engineering")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    # Logging is already configured at module level (console + file)

    config = load_config(Path(args.config))
    if args.env:
        if 'execution' not in config:
            config['execution'] = {}
        config['execution']['mode'] = args.env
    validate_config(config)

    main_process(config)
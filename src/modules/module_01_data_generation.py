# =============================================================================
# src/modules/module_01_data_generation.py
#
# PURPOSE:
#   This is the FIRST module in the modular pipeline of AWS-CapacityForecaster.
#   It generates completely synthetic but highly realistic enterprise server
#   performance metrics — mimicking the kind of daily P95 telemetry data
#   that would come from thousands of servers in a large financial institution
#   (inspired by Citi's TrueSight / BMC monitoring feeds during 2017–2025).
#
#   The goal is to create training data that contains:
#   • Realistic utilization patterns (CPU, memory, disk, network)
#   • Weekly business rhythm (lower weekends)
#   • Annual seasonality
#   • End-of-quarter (EOQ) spikes — very characteristic of banking workloads
#   • US banking holidays effect (usually lower utilization)
#   • Gradual long-term growth
#   • Server-specific bias
#   • Random missing values & rare outliers — just like real monitoring data
#
# ROLE IN THE PIPELINE:
#   → Produces raw data → saved as Parquet in S3/local "raw/" prefix
#   → module_02_data_load.py will read exactly this file/format
#
# OUTPUT GUARANTEES:
#   • Parquet file: raw_server_metrics_YYYYMMDD_to_YYYYMMDD.parquet
#   • Columns: server_id, timestamp, business_unit, region, criticality,
#              cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95
#   • DataFrame is sorted by server_id + timestamp
#   • Missing values are introduced randomly (configurable rate)
#   • Values clipped to realistic business ranges
#
# CONFIGURATION DRIVEN:
#   Almost everything is controlled via config.yaml → makes experiments easy
#   (number of servers, date range, seasonality strength, EOQ parameters, etc.)
#
# DESIGN DECISIONS:
#   • Vectorized numpy/pandas operations → fast even for 500+ servers × 5 years
#   • Normal distribution for base noise → more realistic than uniform
#   • Per-server random bias → simulates hardware/usage differences
#   • Holidays via python-holidays library → accurate US federal + banking-relevant
#   • EOQ window uses last N calendar days (simplified but effective)
#
# USAGE:
#   cd C:\pyproj\AWS-CapacityForecaster
#   python src\modules\module_01_data_generation.py --env local
#
# =============================================================================

import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime

import pandas as pd
import numpy as np
import holidays

from src.utils.config import load_config, validate_config
from src.utils.data_utils import save_processed_data, save_to_s3_or_local

logger = logging.getLogger(__name__)


def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """
    Core function: generates the synthetic Citi-like server metrics DataFrame.

    Uses configuration to control scale, patterns, realism elements.
    Returns clean panel data ready for ETL / feature engineering.
    """
    # ────────────────────────────────────────────────
    #  Extract configuration sections
    # ────────────────────────────────────────────────
    data_cfg       = config.get('data', {})
    seasonality    = data_cfg.get('seasonality', {})
    p95_ranges     = data_cfg.get('p95_ranges', {})
    metadata_cfg   = data_cfg.get('metadata', {})
    aws_cfg        = config.get('aws', {})
    exec_cfg       = config.get('execution', {})

    # Core scale parameters
    num_servers    = data_cfg.get('num_servers', 120)
    start_date     = data_cfg.get('start_date', '2022-01-01')
    end_date       = data_cfg.get('end_date', '2025-12-31')
    granularity    = data_cfg.get('granularity', 'daily')      # 'daily' or 'hourly'
    random_seed    = exec_cfg.get('random_seed', 42)

    # Metadata lists & probabilities
    business_units     = metadata_cfg.get('business_units', ['Retail', 'Investment', 'Wealth', 'Corporate'])
    regions            = metadata_cfg.get('regions', ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'])
    criticalities      = metadata_cfg.get('criticalities', ['High', 'Medium', 'Low'])
    criticality_probs  = metadata_cfg.get('criticality_probs', [0.2, 0.5, 0.3])

    # Seasonality & realism controls
    amplitude          = seasonality.get('amplitude', 0.10)
    growth_rate        = seasonality.get('growth_rate', 0.10)
    eoq_multiplier     = seasonality.get('eoq_multiplier', 1.25)
    eoq_window_days    = seasonality.get('eoq_window_days', 7)
    use_holidays       = seasonality.get('use_holidays', True)
    holiday_multiplier = seasonality.get('holiday_multiplier', 0.85)   # usually dip
    missing_rate       = seasonality.get('missing_rate', 0.01)
    outlier_rate       = seasonality.get('outlier_rate', 0.005)

    # Set seed for reproducibility across runs
    np.random.seed(random_seed)

    logger.info(f"Generating synthetic data | {num_servers} servers | {start_date} → {end_date} | seed={random_seed}")

    # ────────────────────────────────────────────────
    # 1. Create server metadata table (one row per server)
    # ────────────────────────────────────────────────
    servers = pd.DataFrame({
        'server_id':      [f"SRV-{i:04d}" for i in range(1, num_servers + 1)],
        'business_unit':  np.random.choice(business_units, num_servers),
        'region':         np.random.choice(regions, num_servers),
        'criticality':    np.random.choice(criticalities, num_servers, p=criticality_probs)
    })

    # Small random offset per server → simulates different baseline load / hardware variance
    server_biases = np.random.uniform(-8, 8, num_servers)

    # ────────────────────────────────────────────────
    # 2. Create time index
    # ────────────────────────────────────────────────
    freq = 'D' if granularity == 'daily' else 'H'
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    df_dates = pd.DataFrame({'timestamp': dates})

    # ────────────────────────────────────────────────
    # 3. Build full panel (server × time) using cross join
    # ────────────────────────────────────────────────
    df = df_dates.assign(key=1).merge(servers.assign(key=1), on='key').drop(columns='key')
    df = df.sort_values(['server_id', 'timestamp']).reset_index(drop=True)

    # ────────────────────────────────────────────────
    # 4. Generate each P95 metric independently
    # ────────────────────────────────────────────────
    metrics = ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']

    for metric in metrics:
        rng = p95_ranges.get(metric, {'min': 0, 'max': 100})
        min_val, max_val = rng['min'], rng['max']
        mean_val = (min_val + max_val) / 2
        std_val  = (max_val - min_val) / 6   # ~99.7% within range for normal

        # Base signal: normal distribution centered in realistic range
        base = np.random.normal(loc=mean_val, scale=std_val, size=len(df))

        # ───── Weekly business pattern ─────
        day_of_week = df['timestamp'].dt.dayofweek
        weekly_factor = np.where(day_of_week >= 5, 0.80, 1.00)          # weekends lower

        # ───── Annual seasonality (sine wave) ─────
        day_of_year = df['timestamp'].dt.dayofyear
        seasonal = 1 + amplitude * np.sin(2 * np.pi * day_of_year / 365.25)

        # ───── Long-term linear growth ─────
        days_from_start = (df['timestamp'] - pd.to_datetime(start_date)).dt.days
        growth = 1 + growth_rate * (days_from_start / days_from_start.max())

        # ───── End-of-Quarter spikes ─────
        eoq_factor = np.ones(len(df))
        if seasonality.get('use_eoq_spikes', True):
            month = df['timestamp'].dt.month
            day   = df['timestamp'].dt.day
            q_ends = [3,6,9,12]
            for m in q_ends:
                # Rough approximation: last N days of quarter-end month
                mask = (month == m) & (day >= (31 - eoq_window_days + 1))
                eoq_factor = np.where(mask, eoq_multiplier, eoq_factor)

        # ───── Holiday effect ─────
        holiday_factor = np.ones(len(df))
        if use_holidays:
            us_holidays = holidays.US(years=range(int(start_date[:4]), int(end_date[:4])+1))
            is_holiday = df['timestamp'].dt.date.isin(us_holidays)
            holiday_factor = np.where(is_holiday, holiday_multiplier, 1.0)

        # ───── Combine all factors ─────
        utilization = base * weekly_factor * seasonal * growth * eoq_factor * holiday_factor

        # Apply per-server bias
        server_idx = df.index // len(dates)   # which server this row belongs to
        utilization += server_biases[server_idx]

        # Final clipping to valid business range
        utilization = np.clip(utilization, min_val, max_val)

        # ───── Introduce realism defects ─────
        # Random missing values (typical in monitoring feeds)
        missing_mask = np.random.rand(len(df)) < missing_rate
        utilization = np.where(missing_mask, np.nan, utilization)

        # Rare extreme outliers (spikes/dips)
        outlier_mask = np.random.rand(len(df)) < outlier_rate
        utilization = np.where(outlier_mask, np.clip(utilization * 1.60, min_val, max_val), utilization)

        df[metric] = utilization

    # Keep only final columns (raw data should not have engineered features yet)
    final_cols = ['server_id', 'timestamp', 'business_unit', 'region', 'criticality'] + metrics
    df = df[final_cols]

    return df


def main_process(config: dict):
    """Orchestrates data generation, quality checks, saving & summary creation."""
    logger.info("=== MODULE 01 : Synthetic Data Generation ===")

    df = generate_synthetic_data(config)

    # ─── Basic quality gates ───
    if df.empty:
        raise ValueError("Generated DataFrame is empty — check config")

    expected_rows = len(pd.date_range(
        config['data']['start_date'],
        config['data']['end_date'],
        freq='D' if config['data'].get('granularity','daily')=='daily' else 'H'
    )) * config['data']['num_servers']

    if abs(len(df) - expected_rows) > 100:
        logger.warning(f"Row count off: expected ~{expected_rows:,}, got {len(df):,}")

    missing_pct = df['cpu_p95'].isna().mean() * 100
    if missing_pct > 5:
        logger.warning(f"High missing rate on cpu_p95: {missing_pct:.2f}%")

    # ─── Logging summary statistics ───
    logger.info(f"Generated shape: {df.shape}")
    logger.info(f"Missing rate (cpu_p95): {missing_pct:.2f}%")
    logger.info("\n" + df.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(2).to_string())

    # ─── Save main output ───
    start_str = config['data']['start_date'].replace('-','')
    end_str   = config['data']['end_date'].replace('-','')
    filename  = f"raw_server_metrics_{start_str}_to_{end_str}.parquet"

    raw_dir = config.get('paths', {}).get('raw_dir', 'raw/')
    output_path = save_processed_data(
        df=df,
        config=config,
        prefix=raw_dir,
        filename=filename
    )

    # ─── Create JSON summary for debugging / reporting ───
    summary = {
        "module": "01_data_generation",
        "generated_at": datetime.now().isoformat(),
        "rows": len(df),
        "unique_servers": df['server_id'].nunique(),
        "date_range": f"{start_str} → {end_str}",
        "granularity": config['data'].get('granularity','daily'),
        "missing_rate_cpu": round(missing_pct, 2),
        "mean_cpu_p95": round(df['cpu_p95'].mean(), 2),
        "mean_mem_p95": round(df['mem_p95'].mean(), 2),
        "p95_cpu_p95": round(df['cpu_p95'].quantile(0.95), 2)
    }

    summaries_dir = config.get('paths', {}).get('summaries_dir', 'reports/summaries/')
    summary_path = save_to_s3_or_local(
        content=json.dumps(summary, indent=2),
        config=config,
        prefix=summaries_dir,
        filename="module_01_summary.json"
    )

    # ─── Optional quick-look sample ───
    samples_dir = config.get('paths', {}).get('samples_dir', 'samples/')
    sample_path = save_to_s3_or_local(
        content=df.head(200).to_csv(index=False),
        config=config,
        prefix=samples_dir,
        filename="module_01_sample_200rows.csv"
    )

    logger.info("✔ Module 01 completed successfully")
    logger.info(f"   Main output → {output_path}")
    logger.info(f"   Summary    → {summary_path}")
    logger.info(f"   Sample CSV → {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 01 — Generate synthetic Citi-like server metrics")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--env",    default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    config = load_config(Path(args.config))
    
    # Manually override env via args since load_config doesn't take it directly anymore
    if args.env:
        if 'execution' not in config:
            config['execution'] = {}
        config['execution']['mode'] = args.env
        
    validate_config(config)

    main_process(config)
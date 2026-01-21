# src/utils/data_utils.py
"""
Data Utilities Module

This module provides reusable helper functions for data preparation, synthetic data generation,
loading, basic transformation, and quality checks in the AWS-CapacityForecaster project.

Key Focus Areas (aligned with project priorities):
- Data Science/ML: Enables clean, feature-rich datasets for time-series forecasting (e.g., Prophet, scikit-learn models).
- Capacity Planning: Generates realistic enterprise server metrics simulating Citi-style monitoring data (P95 utilization, seasonal peaks).
- Python + AWS: Includes S3 loading via boto3 for cloud integration, keeping logic modular for SageMaker notebooks or Lambda.

All functions are designed to be efficient with large datasets (millions of rows) using pandas and numpy vectorization.
Functions are unit-testable (see tests/test_data_utils.py).

Usage Example (from a notebook):
    from src.utils.data_utils import generate_synthetic_server_metrics, load_from_s3, validate_capacity_df

    df = generate_synthetic_server_metrics(n_servers=100)
    # ... upload to S3 via boto3 in notebook ...
    df_s3 = load_from_s3(bucket='your-bucket', key='data/metrics.csv')
    is_valid, issues = validate_capacity_df(df_s3)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Tuple, Dict, List
import logging
import holidays  # For US banking holidays; install via requirements.txt if needed

# Set up logging for debugging (configurable via environment vars in production)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants for synthetic data generation (customizable for Citi-like scenarios)
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_FREQ = "D"  # Daily metrics, as per Citi monthly/seasonal reporting
METRIC_COLUMNS = ['cpu_p95', 'memory_p95', 'disk_p95', 'network_out_p95']
SERVER_METADATA_COLUMNS = ['server_id', 'app_name', 'business_unit', 'criticality', 'region']

def generate_synthetic_server_metrics(
    n_servers: int = 80,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    freq: str = DEFAULT_FREQ,
    seed: int = 42,
    add_seasonality: bool = True,
    add_eoq_spikes: bool = True,
    noise_level: float = 0.08,
    base_utilization: Dict[str, float] = None,
    trend_factor: float = 0.0005,  # Slight upward trend over time (e.g., growing load)
    eoq_spike_multiplier: float = 1.5,  # End-of-quarter spike factor
) -> pd.DataFrame:
    """
    Generate synthetic daily P95 server metrics simulating enterprise monitoring data (e.g., BMC TrueSight style).

    This function creates realistic time-series data for capacity forecasting:
    - Includes trends, seasonality, noise, and banking-specific events (EOQ spikes, holidays).
    - Outputs a long-format DataFrame ready for ETL/feature engineering.

    Parameters:
    - n_servers: Number of simulated servers (e.g., 80 for medium-scale testing).
    - start_date: Start date string (YYYY-MM-DD).
    - end_date: End date string (YYYY-MM-DD).
    - freq: Pandas frequency string ('D' for daily, 'H' for hourly).
    - seed: Random seed for reproducibility.
    - add_seasonality: If True, add weekly/annual patterns (e.g., lower weekends).
    - add_eoq_spikes: If True, spike utilization at end-of-quarter (banking cycles).
    - noise_level: Standard deviation for Gaussian noise.
    - base_utilization: Dict of base means for each metric (default: cpu=0.5, etc.).
    - trend_factor: Daily upward trend multiplier.
    - eoq_spike_multiplier: Multiplier for EOQ days.

    Returns:
    - pd.DataFrame: Columns include 'server_id', 'date', METRIC_COLUMNS.

    Aligns with Capacity Planning: Simulates P95 metrics for risk analysis/forecasting.
    """
    np.random.seed(seed)
    
    if base_utilization is None:
        base_utilization = {'cpu_p95': 0.5, 'memory_p95': 0.6, 'disk_p95': 0.7, 'network_out_p95': 0.4}
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_days = len(dates)
    
    # Generate server IDs and metadata (for realism)
    server_ids = [f'server_{i:04d}' for i in range(1, n_servers + 1)]
    
    # Expand to long format: one row per server-date
    df = pd.DataFrame({
        'server_id': np.repeat(server_ids, n_days),
        'date': np.tile(dates, n_servers)
    }).sort_values(['server_id', 'date']).reset_index(drop=True)
    
    # Add time index for trends
    df['time_idx'] = df.groupby('server_id')['date'].cumcount()
    
    # Base trends per server (random variation)
    server_bases = {col: np.random.uniform(base_utilization[col] - 0.1, base_utilization[col] + 0.1, n_servers)
                    for col in METRIC_COLUMNS}
    
    for col in METRIC_COLUMNS:
        # Assign base per server
        bases = np.repeat(server_bases[col], n_days)
        
        # Linear trend
        trend = trend_factor * df['time_idx']
        
        # Seasonality (sinusoidal weekly + annual)
        seasonality = 0
        if add_seasonality:
            day_of_week = df['date'].dt.dayofweek / 7 * 2 * np.pi
            day_of_year = df['date'].dt.dayofyear / 365 * 2 * np.pi
            seasonality = 0.1 * np.sin(day_of_week) + 0.05 * np.sin(day_of_year)
        
        # EOQ spikes
        eoq_spike = 0
        if add_eoq_spikes:
            is_eoq = df['date'].dt.is_quarter_end
            eoq_spike = is_eoq.astype(float) * (eoq_spike_multiplier - 1) * bases
        
        # Noise
        noise = np.random.normal(0, noise_level, len(df))
        
        # Combine and clip to [0,1] for utilization
        df[col] = np.clip(bases + trend + seasonality + eoq_spike + noise, 0, 1)
    
    df.drop(columns=['time_idx'], inplace=True)
    
    logger.info(f"Generated synthetic metrics for {n_servers} servers over {n_days} periods.")
    return df

def generate_server_metadata(
    n_servers: int = 80,
    business_units: List[str] = ['Trading', 'Retail', 'Compliance', 'IT'],
    criticalities: List[str] = ['High', 'Medium', 'Low'],
    regions: List[str] = ['US-East', 'US-West', 'EU', 'Asia'],
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate metadata lookup for servers (e.g., for joining with metrics).

    Parameters:
    - n_servers: Number of servers.
    - business_units: List of possible business units.
    - criticalities: List of criticality levels.
    - regions: List of regions.
    - seed: Random seed.

    Returns:
    - pd.DataFrame: Columns SERVER_METADATA_COLUMNS.

    Useful for Capacity Prioritization: Weight forecasts by criticality/business unit.
    """
    np.random.seed(seed)
    server_ids = [f'server_{i:04d}' for i in range(1, n_servers + 1)]
    df = pd.DataFrame({
        'server_id': server_ids,
        'app_name': [f'app_{i:04d}' for i in range(1, n_servers + 1)],
        'business_unit': np.random.choice(business_units, n_servers),
        'criticality': np.random.choice(criticalities, n_servers),
        'region': np.random.choice(regions, n_servers)
    })
    return df

def merge_metrics_with_metadata(
    metrics_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge metrics DataFrame with server metadata.

    Assumes 'server_id' as join key.
    """
    return pd.merge(metrics_df, metadata_df, on='server_id', how='left')

def load_from_s3(
    bucket: str,
    key: str,
    file_type: str = 'csv',  # 'csv' or 'parquet'
    profile_name: Optional[str] = None,
    region: str = "us-east-1"
) -> pd.DataFrame:
    """
    Load data from S3 into a pandas DataFrame (for AWS integration).

    Handles CSV or Parquet formats. Uses boto3 for secure access.

    Parameters:
    - bucket: S3 bucket name.
    - key: Object key (path).
    - file_type: 'csv' or 'parquet'.
    - profile_name: AWS profile (optional).
    - region: AWS region.

    Returns:
    - pd.DataFrame: Loaded data.

    Raises:
    - ValueError: If file_type invalid.
    - ClientError: If S3 access fails.

    Aligns with AWS/Cloud: Enables pulling data from S3 for SageMaker/Athena pipelines.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region)
    s3 = session.client('s3')
    
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        if file_type == 'csv':
            df = pd.read_csv(obj['Body'])
        elif file_type == 'parquet':
            df = pd.read_parquet(obj['Body'])
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        logger.info(f"Loaded {file_type.upper()} from s3://{bucket}/{key}")
        return df
    except ClientError as e:
        logger.error(f"S3 load error: {e}")
        raise

def load_local_csv(path: str) -> pd.DataFrame:
    """
    Load local CSV for development/testing.

    Parameters:
    - path: Local file path (e.g., 'data/raw/metrics.csv').

    Returns:
    - pd.DataFrame.
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded local CSV from {path}")
    return df

def validate_capacity_df(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    max_missing_pct: float = 0.05,
    allow_negative: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate capacity metrics DataFrame for quality issues.

    Checks:
    - Required columns present.
    - Date column is datetime.
    - Missing values < max_missing_pct.
    - No negatives in metrics (if not allowed).
    - No extreme outliers (>1 for utilization).

    Parameters:
    - df: Input DataFrame.
    - expected_columns: List of required columns (default: 'server_id', 'date' + METRIC_COLUMNS).
    - max_missing_pct: Max allowed missing percentage per column.
    - allow_negative: If True, allow negative values (rare).

    Returns:
    - (bool, list): (is_valid, issues_list)

    Aligns with Data Cleansing: Mirrors Citi QA for reliable ML inputs.
    """
    if expected_columns is None:
        expected_columns = ['server_id', 'date'] + METRIC_COLUMNS
    
    issues = []
    
    # Check columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check date type
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        issues.append("'date' column is not datetime")
    
    # Missing values
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if high_missing:
        issues.append(f"High missing (> {max_missing_pct*100}%): {high_missing}")
    
    # Negatives
    if not allow_negative:
        for col in METRIC_COLUMNS:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
    
    # Utilization >1
    for col in METRIC_COLUMNS:
        if col in df.columns and (df[col] > 1).any():
            issues.append(f"Values >1 in {col} (clip recommended)")
    
    is_valid = len(issues) == 0
    if not is_valid:
        logger.warning(f"Validation issues: {issues}")
    else:
        logger.info("DataFrame validated successfully")
    
    return is_valid, issues

def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = 'date',
    include_holidays: bool = True
) -> pd.DataFrame:
    """
    Add calendar-based features for time-series ML (e.g., Prophet regressors).

    Features: year, month, quarter, dayofweek, is_weekend, is_eoq, is_holiday.

    Parameters:
    - df: Input DataFrame with date column.
    - date_col: Name of date column.
    - include_holidays: Add US holidays (banking-relevant).

    Returns:
    - pd.DataFrame: With added features.

    Aligns with ML/Feature Engineering: Enhances forecasting accuracy with seasonality.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['dayofweek'] = dt.dt.dayofweek
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['is_eoq'] = dt.dt.is_quarter_end.astype(int)
    
    if include_holidays:
        us_holidays = holidays.US(years=dt.dt.year.unique())
        df['is_holiday'] = df[date_col].apply(lambda x: x in us_holidays).astype(int)
    
    return df

def clean_numerical_columns(
    df: pd.DataFrame,
    columns: List[str] = METRIC_COLUMNS,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    fill_method: str = 'ffill'  # 'ffill', 'bfill', 'interpolate'
) -> pd.DataFrame:
    """
    Clean numerical metric columns: clip, handle missing.

    Parameters:
    - df: Input DataFrame.
    - columns: List of columns to clean.
    - clip_min/max: Clip bounds.
    - fill_method: Method for NaN filling.

    Returns:
    - pd.DataFrame: Cleaned.

    Aligns with Data Cleansing: Prepares for ML without anomalies.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.clip(df[col], clip_min, clip_max)
            if fill_method == 'interpolate':
                df[col] = df.groupby('server_id')[col].transform(lambda x: x.interpolate())
            elif fill_method in ['ffill', 'bfill']:
                df[col] = df.groupby('server_id')[col].transform(fill_method)
    return df

def resample_to_daily(
    df: pd.DataFrame,
    date_col: str = 'date',
    group_col: str = 'server_id',
    agg_method: str = 'max'  # 'max' for P95 approximation
) -> pd.DataFrame:
    """
    Resample time-series to daily (if hourly input).

    Parameters:
    - df: Input DataFrame.
    - date_col: Date column.
    - group_col: Grouping column (e.g., server_id).
    - agg_method: Aggregation ('max' for P95).

    Returns:
    - pd.DataFrame: Resampled.
    """
    df = df.copy()
    df.set_index(date_col, inplace=True)
    df = df.groupby(group_col).resample('D').agg(agg_method).reset_index()
    return df

# Additional helpers can be added as project evolves (e.g., detect_anomalies with scipy)

"""
module_01_data_generation.py

Reliable starting point of the modular pipeline.
Responsibility: Generate high-fidelity synthetic enterprise server metrics.
Patterns: Config-driven, Citi-style telemetry (P95, seasonality, holidays, metadata).
Output: Raw Parquet file saved to Local or S3.
"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root is in path to import utils
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, validate_config
from src.utils.data_utils import (
    save_processed_data,
    generate_server_metadata
)
# We might need to import more specific generation functions if they aren't fully modularized in data_utils yet.
# For this module, we will implement the core generation logic here, using config as the driver, 
# similar to how src/data_generation.py did it, but cleaner and focused on the module contract.

logger = logging.getLogger(__name__)

def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """
    Core logic to generate synthetic server metrics based on configuration.
    Returns a pandas DataFrame.
    """
    logger.info("Starting synthetic data generation...")
    start_time = time.time()

    # 1. Unpack Config
    data_config = config.get('data', {})
    num_servers = data_config.get('num_servers', 120)
    start_date = data_config.get('start_date', '2022-01-01')
    end_date = data_config.get('end_date', '2025-12-31')
    granularity = data_config.get('granularity', 'daily')
    random_seed = config.get('execution', {}).get('random_seed', 42)
    
    np.random.seed(random_seed)

    logger.info(f"Configuration: {num_servers} servers, {start_date} to {end_date}, {granularity}")

    # 2. Generate Server Metadata (Server IDs, BU, Region, etc.)
    # We can reuse the utility function or implement simplified logic here if dependency logic is complex.
    # For robust modularity, we'll keep the core panel generation self-contained but use helpers where appropriate.
    
    # Generate Server IDs
    server_ids = [f"srv-{i:04d}" for i in range(1, num_servers + 1)]
    
    # Metadata (Simplified for module independence, or call data_utils)
    # Let's simple-gen here to ensure self-containment or use metadata config if detailed
    business_units = ['Retail', 'Investment', 'Wealth', 'Corporate']
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    criticalities = ['High', 'Medium', 'Low']
    
    server_meta = []
    for sid in server_ids:
        server_meta.append({
            'server_id': sid,
            'business_unit': np.random.choice(business_units),
            'region': np.random.choice(regions),
            'criticality': np.random.choice(criticalities, p=[0.2, 0.5, 0.3])
        })
    df_meta = pd.DataFrame(server_meta)

    # 3. Create Time Dimension
    if granularity == 'daily':
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    elif granularity == 'hourly':
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")
    
    logger.info(f"Time dimension created: {len(dates)} periods.")

    # 4. Cross Join (Servers x Time) -> Base Panel
    # Using efficient multi-index creation
    idx = pd.MultiIndex.from_product([server_ids, dates], names=['server_id', 'timestamp'])
    df = pd.DataFrame(index=idx).reset_index()
    
    # Merge Metadata to Panel
    df = df.merge(df_meta, on='server_id', how='left')
    
    # 5. Generate Metrics with Seasonality & Patterns
    # Base ranges from config
    p95_ranges = data_config.get('p95_ranges', {
        'cpu': [10.0, 95.0],
        'memory': [15.0, 92.0],
        'disk': [5.0, 85.0],
        'network_in': [20.0, 500.0]
    })
    
    # Pre-compute time features for vectorization
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Helper to generate metric
    def generate_metric(name, min_val, max_val):
        # Base random noise
        base = np.random.uniform(min_val, max_val, size=len(df))
        
        # Weekly Pattern (dip on weekends)
        if data_config.get('seasonality', {}).get('weekly', True):
            weekend_factor = np.where(df['is_weekend'], 0.8, 1.0)
            base = base * weekend_factor
            
        # Annual Seasonality (Sine wave)
        # Peak around mid-year or end-year depending on "banking cycles" simulation
        # Let's simulate a simple sine wave
        seasonality_factor = 1 + 0.1 * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        base = base * seasonality_factor
        
        # Trend (Linear growth)
        # Small random trend per server to simulate growth
        # For simplicity in vectorization, we apply a global slight trend
        # Real-world: merge server-specific trend slopes. 
        # Here: (date_index / total_days) * growth_factor
        # Simplified:
        total_days = (dates[-1] - dates[0]).days
        days_from_start = (df['timestamp'] - pd.to_datetime(start_date)).dt.days
        trend = 1 + (days_from_start / total_days) * 0.1 # 10% growth over period
        base = base * trend

        # EOQ Spikes (March, June, Sept, Dec)
        if data_config.get('seasonality', {}).get('quarterly_peaks', True):
            is_eoq_month = df['month'].isin([3, 6, 9, 12])
            # Last 7 days of EOQ months
            is_eoq_window = is_eoq_month & (df['timestamp'].dt.day > 23)
            base = np.where(is_eoq_window, base * 1.25, base) # 25% spike

        # Clipping
        return np.clip(base, min_val, max_val)

    logger.info("Generating metrics (CPU, Memory, Disk, Net)...")
    
    df['cpu_p95'] = generate_metric('cpu', p95_ranges['cpu'][0], p95_ranges['cpu'][1])
    df['mem_p95'] = generate_metric('memory', p95_ranges['memory'][0], p95_ranges['memory'][1])
    df['disk_p95'] = generate_metric('disk', p95_ranges['disk'][0], p95_ranges['disk'][1])
    
    # Network might have higher variance
    df['net_in_p95'] = generate_metric('network_in', p95_ranges['network_in'][0], p95_ranges['network_in'][1])
    df['net_out_p95'] = generate_metric('network_out', p95_ranges.get('network_out', [10,300])[0], p95_ranges.get('network_out', [10,300])[1])

    # Cleanup temporary cols if desired, or keep them as basic features
    # Keeping them helps module 03 (ETL) know context, but usually we strictly generate raw metrics here.
    # We'll drop derived features to keep it "Raw"
    drop_cols = ['day_of_week', 'day_of_year', 'month', 'is_weekend']
    df.drop(columns=drop_cols, inplace=True)

    generation_time = time.time() - start_time
    logger.info(f"Data generation complete. Rows: {len(df)}. Time: {generation_time:.2f}s")
    
    return df

def main(config):
    logger.info("=== Module 01: Data Generation ===")
    
    # 1. Generate
    df = generate_synthetic_data(config)
    
    # 2. Quality Checks
    if df.empty:
        logger.error("Generated dataframe is empty!")
        sys.exit(1)
        
    required_cols = ['server_id', 'timestamp', 'cpu_p95']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
        sys.exit(1)
        
    logger.info("Sample generated data:")
    logger.info("\n" + str(df.head()))

    # 3. Save Output
    # Filename pattern: raw_server_metrics_{YYYYMMDD}.parquet
    timestamp_str = datetime.now().strftime("%Y%m%d")
    filename = f"raw_server_metrics_{timestamp_str}.parquet"
    
    # Use config-defined prefix or default "raw/"
    raw_prefix = config.get('aws', {}).get('raw_prefix', 'raw/')
    
    # Using shared utility to save (handles S3 vs Local logic)
    output_path = save_processed_data(df, config, prefix=raw_prefix, filename=filename)
    
    if output_path:
        logger.info(f"SUCCESS: Data saved to {output_path}")
    else:
        logger.error("Failed to save data.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 01: Data Generation")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--env", type=str, default="local", choices=["local", "sagemaker", "lambda"], help="Execution environment")
    
    args = parser.parse_args()
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Load Config
        # Config loader doesn't take 'env' direct arg, so we load base then override
        config = load_config(Path(args.config))
        
        # Override execution mode from CLI if provided
        if args.env:
            if 'execution' not in config:
                config['execution'] = {}
            config['execution']['mode'] = args.env
            
        validate_config(config)
        
        # Run Module
        main(config)
        
    except Exception as e:
        logger.exception(f"Module execution failed: {e}")
        sys.exit(1)

"""
# Run command: python demos/demo_data_utils.py
Demo: Data Utilities (src/utils/data_utils.py)

This script demonstrates the capabilities of the data_utils module:
1. Generating synthetic enterprise server metrics ("Citi-style").
2. Validating data quality (missing values, negatives, etc.).
3. Detecting anomalies using simple Z-score analysis.
4. Adding calendar features (holidays, EOQ) for ML.
5. Simulating S3 loading (mocked).
"""

import logging
import pandas as pd
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.data_utils import (
    generate_synthetic_server_metrics,
    generate_server_metadata,
    merge_metrics_with_metadata,
    validate_capacity_df,
    add_calendar_features,
    detect_anomalies_simple,
    clean_numerical_columns,
    resample_to_daily
)
from src.utils.config import CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_demo():
    print("\n" + "="*80)
    print(" 🚀 AWS-CapacityForecaster: Data Utilities Demo")
    print("="*80)

    # 1. Synthetic Data Generation
    print("\n[1] Generating Synthetic Enterprise Metrics...")
    print(f"    - Simulating logic defined in Config (Project Root: {CONFIG.get('_project_root', 'Unknown')})")
    
    metrics_df = generate_synthetic_server_metrics(
        n_servers=5, 
        start_date="2024-01-01", 
        end_date="2024-06-30", 
        freq="D",
        add_eoq_spikes=True,
        noise_level=0.02
    )
    
    print(f"    ✅ Generated {len(metrics_df)} rows for 5 servers.")
    print("    - Preview (Head):")
    print(metrics_df.head(3).to_string())

    # 2. Metadata Generation & Merging
    print("\n[2] Generating & Merging Metadata...")
    meta_df = generate_server_metadata(n_servers=5)
    full_df = merge_metrics_with_metadata(metrics_df, meta_df)
    
    print("    - Merged DataFrame Columns:", list(full_df.columns))
    print("    - Sample Row:")
    print(full_df.iloc[0].to_frame().T.to_string())

    # 3. Data Validation
    print("\n[3] Running Data Validation...")
    is_valid, issues = validate_capacity_df(full_df)
    
    if is_valid:
        print("    ✅ Data Validation Passed!")
    else:
        print(f"    ❌ Validation Issues Found: {issues}")

    # Injecting a fault to test validation
    print("    > Injecting invalid data (negative CPU)...")
    full_df.loc[0, 'cpu_p95'] = -0.05
    is_valid_bad, issues_bad = validate_capacity_df(full_df)
    print(f"    - Re-validation result: {issues_bad}")
    
    # Fix it
    full_df = clean_numerical_columns(full_df, clip_min=0.0, clip_max=1.0)
    print("    ✅ Cleaned numerical columns (clipped negative values).")

    # 4. Feature Engineering (Calendar)
    print("\n[4] Adding Calendar Features (Holidays, EOQ)...")
    full_df = add_calendar_features(full_df, include_holidays=True)
    
    holidays = full_df[full_df['is_holiday'] == 1]
    eoq = full_df[full_df['is_eoq'] == 1]
    
    print(f"    - Found {len(holidays['date'].unique())} unique holidays in range.")
    print(f"    - Found {len(eoq['date'].unique())} end-of-quarter days.")
    print("    - Sample Holiday Row:")
    if not holidays.empty:
        print(holidays[['date', 'is_holiday', 'is_eoq', 'dayofweek']].head(1).to_string())

    # 5. Anomaly Detection
    print("\n[5] Detecting Anomalies (Z-Score)...")
    # Inject a spike
    print("    > Injecting a massive CPU spike (Anomalous)...")
    full_df.loc[10, 'cpu_p95'] = 0.99  # High value
    
    anomalies = detect_anomalies_simple(full_df, threshold=2.5)
    n_outliers = anomalies.sum().sum()
    print(f"    - Detected {n_outliers} potential anomalies (Z > 2.5).")
    
    if n_outliers > 0:
        print("    - Anomaly Matrix Sample:")
        print(anomalies.iloc[10:12].to_string())

    print("\n" + "="*80)
    print(" ✅ Demo Complete")
    print("="*80)

if __name__ == "__main__":
    run_demo()

"""
# Run command: python demos/demo_ml_utils.py
Demo: ML Utilities (src/utils/ml_utils.py)

This script demonstrates the complete machine learning workflow provided by the ml_utils module.
It simulates a real-world scenario by:
1.  Generating synthetic server metrics (using data_utils).
2.  Engineering features (lags, rolling stats) for ML models.
3.  Training forecasting models (Prophet and RandomForest).
4.  Measuring model performance (MAE, RMSE).
5.  Forecasting future capacity needs.
6.  Analyzing risks (P95 bandwidth) and clustering underutilized servers.

The demo integrates with the centralized configuration to pull model parameters and thresholds.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress joblib/loky warning on Windows (must be before sklearn import)
if os.name == 'nt' and 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

from src.utils.data_utils import generate_synthetic_server_metrics
from src.utils.ml_utils import (
    engineer_features,
    check_stationarity,
    train_prophet_model,
    train_sklearn_model,
    generate_forecast,
    evaluate_model,
    compare_models,
    flag_risks,
    cluster_utilization
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')  # Simplified for demo output
logger = logging.getLogger(__name__)

def run_demo():
    print("\n" + "="*80)
    print("    AWS-CapacityForecaster: ML Utilities Demo")
    print("="*80)

    # 1. Data Generation
    print("\n[1] Generating Synthetic Data...")
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    # Generate data for 5 servers
    # Generate data for 5 servers
    raw_df = generate_synthetic_server_metrics(
        n_servers=5, 
        start_date=start_date, 
        end_date=end_date, 
        freq='D'
    )
    
    # Focus on one server for detailed forecasting demo
    target_server = "server_0001" if "server_0001" in raw_df['server_id'].values else "server_001" # Handle padding differences?
    # Actually data_utils uses f'server_{i:04d}', so server_0001
    target_server = "server_0001"
    
    server_df = raw_df[raw_df['server_id'] == target_server].copy()
    server_df.set_index('date', inplace=True) # data_utils returns 'date', not 'timestamp'
    server_df.index.name = 'timestamp' # ml_utils expects 'timestamp' or DatetimeIndex
    server_df.sort_index(inplace=True)
    
    print(f"    Generated {len(server_df)} records for {target_server} ({start_date} to {end_date})")
    print(f"    Sample metrics:\n{server_df[['cpu_p95', 'memory_p95']].head(3)}")

    # 2. Feature Engineering
    print("\n[2] Feature Engineering...")
    # Add lags and rolling stats for 'cpu_utilization'
    # Utilizes defaults from config (e.g. windows=[7, 30])
    # Manually specifying here for clarity, but logic uses config defaults if args omitted
    metrics_to_engineer = ['cpu_p95']
    feat_df = engineer_features(
        server_df, 
        metrics=metrics_to_engineer,
        lag_periods=[1, 7],
        rolling_windows=[7, 30]
    )
    
    print(f"    Engineered features. Columns now include:")
    print(f"       {list(feat_df.columns)}")

    # 3. Stationarity Check
    print("\n[3] Checking Stationarity (ADF Test)...")
    adf_result = check_stationarity(feat_df['cpu_p95'])
    status = "Stationary" if adf_result['is_stationary'] else "Non-Stationary"
    print(f"    Info: CPU P95 is {status} (p-value: {adf_result['p_value']:.4f})")

    # 4. Model Training & Validation Split
    print("\n[4] Training Models (Train/Test Split)...")
    
    # 80/20 Split
    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    
    print(f"    Train size: {len(train_df)}, Test size: {len(test_df)}")

    # A) Prophet Model
    print("    Training Prophet Model...")
    # Prophet requires 'ds' and 'y', ml_utils handles this mapping
    # We can add external regressors like 'is_quarter_end'
    prophet_model = train_prophet_model(
        train_df, 
        target='cpu_p95',
        regressors=['is_quarter_end']
    )

    # B) Scikit-Learn RandomForest
    print("    Training Random Forest Model...")
    feature_cols = [c for c in train_df.columns if 'lag' in c or 'rolling' in c or 'is_' in c]
    rf_model = train_sklearn_model(
        train_df,
        target='cpu_p95',
        features=feature_cols,
        model_type='random_forest'
    )

    # 5. Forecasting & Evaluation
    print("\n[5] Evaluating Performance on Test Set...")
    
    # Prophet Forecast
    future_p = prophet_model.make_future_dataframe(periods=len(test_df))
    
    # We need to add 'is_quarter_end' for the future dates. 
    # In this demo, since we split train/test chronologically, test_df covers the future period.
    # We can join test_df's features to future_p.
    
    future_p = future_p.set_index('ds')
    # Join with features, ensuring we match on the datetime index. feat_df index is 'timestamp'.
    # We need to make sure the indices align. future_p index is date (no time) if freq='D', feat_df might specify time?
    # Both should be Daily.
    
    # Map feat_df index to 'ds' name for join if needed, or just join.
    # Note: feat_df contains ALL history (train + test).
    future_p = future_p.join(feat_df[['is_quarter_end']], how='left')
    
    # Reset index to get 'ds' back as a column
    future_p = future_p.reset_index()
    # If the index name was lost or different, rename it back to 'ds'
    if 'ds' not in future_p.columns:
        # Check if 'index' or 'timestamp' exists
        if 'timestamp' in future_p.columns:
            future_p.rename(columns={'timestamp': 'ds'}, inplace=True)
        elif 'index' in future_p.columns:
            future_p.rename(columns={'index': 'ds'}, inplace=True)
            
    # Fill NAs if any (e.g. if future extends beyond available features, shouldn't happen here)
    future_p['is_quarter_end'] = future_p['is_quarter_end'].fillna(0)

    prophet_pred = prophet_model.predict(future_p)[['ds', 'yhat']].set_index('ds').loc[test_df.index]
    
    # RandomForest Forecast
    rf_pred = generate_forecast(rf_model, len(test_df), future_df=test_df[feature_cols])
    rf_pred.index = test_df.index # Align index
    
    # Calculate Metrics
    p_metrics = evaluate_model(test_df['cpu_p95'], prophet_pred['yhat'])
    rf_metrics = evaluate_model(test_df['cpu_p95'], rf_pred['yhat'])
    
    # Compare
    comparison = compare_models([
        {**p_metrics, 'Model': 'Prophet'},
        {**rf_metrics, 'Model': 'RandomForest'}
    ])
    print("\n    Model Comparison Table:")
    print(comparison.set_index('Model'))

    # 6. Risk Analysis
    print("\n[6] Risk Analysis...")
    # Flag risks where CPU > 90% (or config default) based on forecast
    risks = flag_risks(
        train_df, # Historical context
        rf_pred,  # Forecast
        metric='cpu_p95'
    )
    risk_count = risks['risk_flag'].sum()
    print(f"    Found {risk_count} potential high-risk days in the forecast period.")
    if risk_count > 0:
        print(f"       Example Risks:\n{risks[['cpu_p95', 'risk_flag']].head(3)}")

    # 7. Clustering Optimization
    print("\n[7] Clustering for Optimization (Underutilized Servers)...")
    # Using the original dataset with multiple servers
    # Aggregating metrics to find stable patterns
    agg_df = raw_df.groupby('server_id')[['cpu_p95', 'memory_p95']].mean().rename(
        columns={'cpu_p95': 'cpu_mean', 'memory_p95': 'mem_mean'}
    )
    
    clustered_df = cluster_utilization(
        agg_df,
        features=['cpu_mean', 'mem_mean'] # Defaults from config usually
    )
    
    print("    Servers clustered by utilization profile:")
    print(clustered_df.sort_values('cluster'))

    print("\n" + "="*80)
    print(" ML Demo Complete")
    print("="*80)

if __name__ == "__main__":
    run_demo()

# =============================================================================
# src/modules/module_05_risk_capacity_analysis.py
#
# PURPOSE:
#   FIFTH module in the AWS-CapacityForecaster pipeline.
#   Analyzes forecasts from module_04 to identify risks (threshold exceedance,
#   seasonal peaks), underutilized servers (via clustering), and cost-saving
#   recommendations. Recreates Citi-style capacity planning: at-risk flagging,
#   EOQ/holiday analysis, optimization insights.
#
# ROLE IN PIPELINE:
#   module_04 (forecasts/*.parquet + metrics/*.json) → THIS → risk_analysis/*.parquet + *.json + *.csv + viz/*.png
#
# OUTPUT GUARANTEES:
#   - Risk-flagged forecasts Parquet (added flags/levels)
#   - Optimization recommendations JSON (savings, counts)
#   - Server clusters CSV (K-Means results)
#   - Visualizations (PNG: heatmap, clusters)
#   - Summary JSON + detailed logging
#
# USAGE:
#   python -m src.modules.module_05_risk_capacity_analysis --env local
#
# =============================================================================

import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
import os

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

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
log_file = log_dir / "module_05_risk_capacity_analysis.log"

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

logger.info("Logging initialized — console + file (logs/module_05_risk_capacity_analysis.log)")

# =============================================================================
# Helper Functions
# =============================================================================
def calculate_risks(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag risks based on predicted exceedance of thresholds."""
    thresholds = config.get('risk_analysis', {}).get('thresholds', {'cpu': 80.0})  # Adapt for metrics
    # Assuming 'predicted' is the key forecast column; adjust for multi-metric if needed
    df['risk_flag'] = df['predicted'] > list(thresholds.values())[0]  # Example for cpu
    df['risk_level'] = np.where(df['predicted'] > list(thresholds.values())[0] * 1.2, 'high',
                                np.where(df['risk_flag'], 'medium', 'low'))
    logger.info(f"Risk flags computed: {df['risk_flag'].sum()} flagged rows")
    return df

def seasonal_analysis(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add seasonal markers and compute peak risks."""
    eoq_months = config.get('risk_analysis', {}).get('eoq_months', [3, 6, 9, 12])
    df['month'] = df['timestamp'].dt.month
    df['is_eoq'] = (df['month'].isin(eoq_months)) & df['timestamp'].dt.is_quarter_end.astype(int)
    
    years = range(df['timestamp'].dt.year.min(), df['timestamp'].dt.year.max() + 1)
    us_holidays = holidays.US(years=years)
    df['is_holiday'] = df['timestamp'].dt.date.isin(us_holidays).astype(int)
    
    seasonal_df = df[df['is_future'] & (df['is_eoq'] | df['is_holiday'])]
    if not seasonal_df.empty:
        peak_risks = seasonal_df.groupby('server_id')['predicted'].max().rename('seasonal_peak_risk')
        df = df.merge(peak_risks, on='server_id', how='left')
        logger.info(f"Seasonal peaks computed for {len(peak_risks)} servers")
    return df

def detect_underutilized(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag underutilized servers based on average future predicted."""
    under_thresh = config.get('risk_analysis', {}).get('underutilized_threshold', 30.0)
    future_df = df[df['is_future']]
    avg_util = future_df.groupby('server_id')['predicted'].mean()
    under_servers = avg_util[avg_util < under_thresh].index
    df['is_underutilized'] = df['server_id'].isin(under_servers)
    logger.info(f"Detected {len(under_servers)} underutilized servers")
    return df

def cluster_and_optimize(df: pd.DataFrame, config: dict) -> tuple:
    """Cluster servers on utilization features and estimate savings."""
    n_clusters = config.get('risk_analysis', {}).get('clustering_n_clusters', 3)
    features = df.groupby('server_id').agg({'predicted': ['mean', 'std']})
    features.columns = ['_'.join(col) for col in features.columns]  # Flatten
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    server_clusters = pd.DataFrame({'server_id': features.index, 'cluster': kmeans.labels_})
    
    cost_per_instance = config.get('risk_analysis', {}).get('cost_per_instance', 100.0)
    # Assume lowest mean cluster is underutilized; 50% savings via consolidation
    cluster_centers = kmeans.cluster_centers_[:, 0]  # Mean util
    low_util_cluster = np.argmin(cluster_centers)
    low_util_servers = server_clusters[server_clusters['cluster'] == low_util_cluster]
    savings = len(low_util_servers) * cost_per_instance * 0.5
    optim_summary = {'total_savings_usd': round(savings, 2), 'underutilized_count': len(low_util_servers)}
    logger.info(f"Clustering complete: Savings estimate ${optim_summary['total_savings_usd']}")
    return server_clusters, optim_summary

def generate_visualizations(df: pd.DataFrame, server_clusters: pd.DataFrame, features: pd.DataFrame, config: dict):
    """Generate and save risk heatmap and cluster plots."""
    risk_dir = Path(config['paths']['local_data_dir']) / config['paths']['risk_analysis_dir']
    risk_dir.mkdir(parents=True, exist_ok=True)
    
    # Risk Heatmap
    pivot = df.pivot_table(index='server_id', columns='timestamp', values='risk_level', aggfunc='first')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot.replace({'low': 0, 'medium': 1, 'high': 2}).fillna(0), cmap='Reds')
    plt.title('Risk Level Heatmap')
    plt.savefig(risk_dir / 'risk_heatmap.png')
    plt.close()
    
    # Cluster Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(features['predicted_mean'], features['predicted_std'], c=server_clusters['cluster'], cmap='viridis')
    plt.title('Server Utilization Clusters')
    plt.xlabel('Mean Predicted'); plt.ylabel('Std Predicted')
    plt.savefig(risk_dir / 'utilization_clusters.png')
    plt.close()
    logger.info("Visualizations saved to risk_analysis/")

# =============================================================================
# Main Logic
# =============================================================================
def main_process(config):
    logger.info("=== MODULE 05 : Risk & Capacity Analysis ===")
    
    # Load latest forecasts dynamically
    forecasts_base = Path(config['paths']['local_data_dir']) / config['paths']['forecasts_dir']
    parquet_files = list(forecasts_base.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No forecasts parquet files found.")
    latest_file = max(parquet_files, key=os.path.getmtime)
    logger.info(f"Loading latest forecasts: {latest_file.name}")
    
    df = load_from_s3_or_local(config, prefix=config['paths']['forecasts_dir'], filename=latest_file.name)
    if df is None:
        raise FileNotFoundError(f"Failed to load {latest_file.name}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['server_id', 'timestamp'])
    
    # Optional: Load metrics for context
    metrics_filename = config['paths'].get('metrics_summary_filename', 'model_comparison.json')
    metrics_content = load_from_s3_or_local(config, prefix=config['paths']['metrics_dir'], filename=metrics_filename, file_type='json')
    if metrics_content:
        metrics = json.loads(metrics_content)
        logger.info(f"Best model from metrics: {metrics.get('best_model')}")
    
    # Output directories
    local_base = Path(config['paths']['local_data_dir'])
    risk_dir = local_base / config['paths'].get('risk_analysis_dir', 'risk_analysis')
    risk_dir.mkdir(parents=True, exist_ok=True)
    
    # Optional debug limit
    servers = df['server_id'].unique()
    debug_max = config.get('risk_analysis', {}).get('debug_max_servers', None)
    if debug_max:
        servers = servers[:debug_max]
        logger.info(f"DEBUG MODE: Limiting to first {debug_max} servers")
    df = df[df['server_id'].isin(servers)]
    
    logger.info(f"Analyzing {len(servers)} servers...")
    
    # Enrich with risks and seasonal analysis
    df = calculate_risks(df, config)
    df = seasonal_analysis(df, config)
    df = detect_underutilized(df, config)
    
    # Clustering and optimization
    features = df.groupby('server_id').agg({'predicted': ['mean', 'std']})
    features.columns = ['_'.join(col) for col in features.columns]
    clusters, optim_summary = cluster_and_optimize(df, config)
    
    # Visualizations (if enabled)
    if config.get('risk_analysis', {}).get('generate_viz', True):
        generate_visualizations(df, clusters, features, config)
    
    # Save enriched forecasts
    forecasts_filename = 'risk_flagged_forecasts.parquet'
    save_path_f = save_processed_data(
        df, config,
        prefix=config['paths']['risk_analysis_dir'],
        filename=forecasts_filename
    )
    logger.info(f"Saved risk-flagged forecasts to {save_path_f}")
    
    # Save optimizations and clusters
    save_to_s3_or_local(
        json.dumps(optim_summary, indent=2),
        config,
        prefix=config['paths']['risk_analysis_dir'],
        filename='optimization_recommendations.json'
    )
    save_to_s3_or_local(
        clusters.to_csv(index=False),
        config,
        prefix=config['paths']['risk_analysis_dir'],
        filename='server_clusters.csv'
    )
    
    # Save summary
    summary = {
        "analyzed_at": datetime.now().isoformat(),
        "total_servers": len(servers),
        "at_risk_servers": int(df['risk_flag'].sum()),
        "underutilized_servers": int(df['is_underutilized'].sum()),
        "optimization_savings_usd": optim_summary['total_savings_usd']
    }
    save_to_s3_or_local(
        json.dumps(summary, indent=2),
        config,
        prefix=config['paths']['summaries_dir'],
        filename='module_05_summary.json'
    )
    
    logger.info("✔ Module 05 completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 05 — Risk & Capacity Analysis")
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
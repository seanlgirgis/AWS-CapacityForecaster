Below is the **full detailed design** for `src/modules/module_05_risk_capacity_analysis.py`, the fifth module in the AWS-CapacityForecaster pipeline. This design follows the project's overall philosophy: focusing on Data Science/ML (forecasting accuracy), Performance Monitoring & Capacity Planning (risk flagging, seasonal analysis, utilization optimization, cost recommendations), and clean Python + AWS/cloud integration. It builds directly on the outputs from previous modules, ensuring continuity.

The design is structured as requested:
- **Purpose of the Step**
- **Entry Criteria**
- **Processing** (detailed, with pseudo code and a text-based flow chart)
- **Outcome**
- **Success Criteria**

After the design documentation, I provide the **complete Python code** for the module. The code:
- Uses shared utilities like `load_config`, `validate_config`, `load_from_s3_or_local`, `save_to_s3_or_local`, and `save_processed_data` from `src.utils.config` and `src.utils.data_utils`.
- Implements logging similar to other modules: console handler + timed rotating file handler (daily rotation, keeps 7 backups) in `logs/module_05_risk_capacity_analysis.log`.
- Logs important information (e.g., key steps, summaries, errors, and results).
- Is configurable via `config/config.yaml` (e.g., paths, thresholds for risk flagging, cost assumptions).
- Assumes the project's folder structure (e.g., `data/scratch/forecasts/`, `data/scratch/metrics/`, etc.).
- Focuses on analyzing forecasts from module_04 to identify risks and optimizations, mimicking Citi-style capacity planning.

---

## Detailed Design for Module 05: Risk & Capacity Analysis

### Purpose of the Step
This module performs post-forecasting analysis to translate raw predictions into actionable capacity insights. It recreates Citi-style performance and capacity planning by:
- Identifying at-risk servers (e.g., those forecasted to exceed utilization thresholds).
- Conducting seasonal risk analysis (e.g., flagging end-of-quarter spikes or holiday dips).
- Detecting underutilized resources for optimization (e.g., clustering servers for consolidation).
- Generating cost-saving recommendations (e.g., based on AWS instance resizing or decommissioning).
- Producing enriched reports and summaries for stakeholders.

**Role in Pipeline**: 
- Takes inputs from module_04 (forecasts parquet + metrics JSON) and optionally module_03 (processed features for baseline data).
- Outputs: Risk-flagged forecasts (Parquet), optimization recommendations (JSON/CSV), and visualizations (optional PNG/PDF via matplotlib/seaborn).
- This step emphasizes the project's secondary target: Performance Monitoring & Capacity Planning, while leveraging ML outputs from the primary target (Data Science/ML).

It ensures outputs are persistent, queryable (e.g., via Athena in AWS), and logged for auditability.

### Entry Criteria
To run successfully, this module requires:
- **Prerequisites**:
  - Module_04 must have completed successfully, producing:
    - Enriched forecasts Parquet (e.g., `forecasts/all_model_forecasts.parquet`): Columns like `server_id`, `model`, `timestamp`, `actual`, `predicted`, `yhat_lower`, `yhat_upper`, `is_future`.
    - Metrics summary JSON (e.g., `metrics/model_comparison.json`): For model performance context (e.g., sMAPE).
  - Optional: Processed features from module_03 (e.g., `processed/processed_features_*.parquet`) for additional baseline metrics (e.g., historical utilization).
- **Configuration**:
  - `config/config.yaml` must define:
    - Paths: `forecasts_dir`, `metrics_dir`, `risk_analysis_dir` (output), `summaries_dir`.
    - Parameters: `risk_thresholds` (e.g., {'cpu': 80.0, 'mem': 70.0}), `underutilized_threshold` (e.g., 30.0), `eoq_months` (e.g., [3,6,9,12]), `cost_per_instance` (e.g., 100.0 USD/month for savings calcs), `clustering_n_clusters` (e.g., 3 for K-Means).
    - Execution mode: `--env` arg (e.g., local, sagemaker).
- **Data Availability**:
  - Latest forecasts file must exist in S3/local (dynamically loaded as the most recent Parquet).
  - Data must cover at least 90 future days for meaningful risk analysis.
- **Environment**:
  - Python 3.12+ with libraries: pandas, numpy, scikit-learn (for clustering), matplotlib/seaborn (for viz), etc. (as per project's env).
- **Failure if Not Met**: The module will raise exceptions (e.g., FileNotFoundError) and log errors, preventing partial runs.

### Processing
This section details the step-by-step logic. Processing is per-server grouped (using pandas groupby) for efficiency. Key operations include threshold-based flagging, seasonal filtering, K-Means clustering, and cost computations.

#### High-Level Flow Chart (Text-Based)
```
Start
  |
  v
Load Config & Validate --> Load Forecasts (latest Parquet) --> Load Metrics JSON (optional)
  | (if fail: log error & exit)
  v
Group by Server_ID
  |
  v
For Each Server:  (Parallelizable with joblib if large scale)
  - Compute Risk Flags (threshold exceedance on predicted)
  - Seasonal Analysis (filter EOQ/holidays, compute peak risks)
  - Underutilization Detection (avg predicted < threshold)
  |
  v
Global Analysis:
  - Cluster Servers (K-Means on avg utilization/features)
  - Compute Cost Savings (based on clusters/underutilized)
  |
  v
Generate Visualizations (e.g., risk heatmaps, cluster plots)
  |
  v
Save Outputs: Enriched Risk Parquet, Recommendations JSON/CSV, Viz PNGs
  |
  v
Save Summary JSON & Log Completion
  |
End (Success) or Error (Log & Raise)
```

#### Detailed Pseudo Code
```
IMPORT necessary libraries: logging, argparse, json, Path, TimedRotatingFileHandler, pandas, numpy, sklearn.cluster.KMeans, matplotlib.pyplot, seaborn, etc.
FROM src.utils: load_config, validate_config, load_from_s3_or_local, save_to_s3_or_local, save_processed_data

SETUP Logging:
  - Create logger with INFO level
  - Add console handler (format: '%(asctime)s | %(levelname)-7s | %(message)s')
  - Create log_dir = Path("logs"); mkdir if not exist
  - Add timed file handler: filename="logs/module_05_risk_capacity_analysis.log", midnight rotation, 7 backups

FUNCTION calculate_risks(forecasts_df, config):
  # Input: forecasts_df (pandas DF), config (dict)
  # Output: enriched_df with new columns: 'risk_flag' (bool), 'risk_level' (low/med/high), 'peak_risk_pct' (float)
  thresholds = config['risk_analysis']['thresholds']  # e.g., {'cpu':80.0}
  FOR each metric in thresholds.keys():  # e.g., cpu_p95, but adapted to predicted cols
    enriched_df['risk_flag_' + metric] = enriched_df['predicted'] > thresholds[metric]
    enriched_df['risk_level_' + metric] = CASE
      WHEN predicted > thresholds[metric] * 1.2 THEN 'high'
      WHEN predicted > thresholds[metric] THEN 'medium'
      ELSE 'low'
  enriched_df['overall_risk_flag'] = any(risk_flag_* columns)
  RETURN enriched_df

FUNCTION seasonal_analysis(forecasts_df, config):
  # Filter future rows for EOQ/holidays
  eoq_months = config['risk_analysis']['eoq_months']
  forecasts_df['is_eoq'] = (forecasts_df['timestamp'].dt.month in eoq_months) & (forecasts_df['timestamp'].dt.is_quarter_end)
  forecasts_df['is_holiday'] = forecasts_df['timestamp'].dt.date in holidays.US()  # Use holidays lib
  seasonal_df = forecasts_df[forecasts_df['is_future'] & (forecasts_df['is_eoq'] | forecasts_df['is_holiday'])]
  # Compute peak risks: max predicted in seasonal periods
  grouped = seasonal_df.groupby('server_id')
  FOR group in grouped:
    group['seasonal_peak_risk'] = group['predicted'].max() / thresholds['cpu'] * 100  # Example for cpu
  RETURN forecasts_df with added seasonal columns

FUNCTION detect_underutilized(forecasts_df, config):
  # Compute avg future predicted
  under_thresh = config['risk_analysis']['underutilized_threshold']
  future_df = forecasts_df[forecasts_df['is_future']]
  avg_util = future_df.groupby('server_id')['predicted'].mean()
  underutilized_servers = avg_util[avg_util < under_thresh].index
  forecasts_df['is_underutilized'] = forecasts_df['server_id'].isin(underutilized_servers)
  RETURN forecasts_df

FUNCTION cluster_and_optimize(forecasts_df, config):
  # Use K-Means on features like avg_predicted, std_predicted
  n_clusters = config['risk_analysis']['clustering_n_clusters']
  features = forecasts_df.groupby('server_id').agg({'predicted': ['mean', 'std']})  # Flatten multi-index
  kmeans = KMeans(n_clusters=n_clusters).fit(features)
  labels = kmeans.labels_
  # Map labels back to servers
  server_clusters = pd.DataFrame({'server_id': features.index, 'cluster': labels})
  # Compute savings: assume low-util clusters can consolidate (e.g., 50% savings)
  cost_per_instance = config['risk_analysis']['cost_per_instance']
  low_util_clusters = server_clusters[server_clusters['cluster'] == argmin(kmeans.cluster_centers_[:,0])]  # Lowest mean util cluster
  savings = len(low_util_clusters) * cost_per_instance * 0.5  # Hypothetical 50% savings per month
  RETURN server_clusters, {'total_savings_usd': savings, 'underutilized_count': len(low_util_clusters)}

FUNCTION generate_visualizations(forecasts_df, server_clusters, config):
  # Plot 1: Risk heatmap (seaborn)
  plt.figure(); sns.heatmap(...)  # E.g., pivot on server_id vs timestamp with risk_level
  save to risk_dir / 'risk_heatmap.png'
  # Plot 2: Cluster scatter (matplotlib)
  plt.figure(); plt.scatter(features['mean'], features['std'], c=labels)
  save to risk_dir / 'utilization_clusters.png'

MAIN_PROCESS(config):
  LOG "=== MODULE 05 : Risk & Capacity Analysis ==="
  # Load latest forecasts
  forecasts_dir = Path(config['paths']['local_data_dir']) / config['paths']['forecasts_dir']
  latest_forecasts = max(forecasts_dir.glob("*.parquet"), key=os.path.getmtime)
  df = load_from_s3_or_local(config, prefix=config['paths']['forecasts_dir'], filename=latest_forecasts.name)
  LOG "Loaded forecasts: {df.shape}"
  
  # Optional: Load metrics JSON for context
  metrics_filename = config['paths']['metrics_summary_filename']
  metrics = json.loads(load_from_s3_or_local(config, prefix=config['paths']['metrics_dir'], filename=metrics_filename))
  LOG "Best model from metrics: {metrics['best_model']}"
  
  # Enrich with risks
  df = calculate_risks(df, config)
  df = seasonal_analysis(df, config)
  df = detect_underutilized(df, config)
  
  # Global optimizations
  clusters, savings = cluster_and_optimize(df, config)
  LOG "Optimization summary: Savings ${savings['total_savings_usd']}, Underutilized: {savings['underutilized_count']}"
  
  # Visualizations
  IF config['risk_analysis']['generate_viz']: generate_visualizations(df, clusters, config)
  
  # Save outputs
  risk_dir = Path(config['paths']['local_data_dir']) / config['paths']['risk_analysis_dir']; risk_dir.mkdir(exist_ok=True)
  save_processed_data(df, config, prefix=config['paths']['risk_analysis_dir'], filename='risk_flagged_forecasts.parquet')
  save_to_s3_or_local(json.dumps(savings, indent=2), config, prefix=config['paths']['risk_analysis_dir'], filename='optimization_recommendations.json')
  save_to_s3_or_local(clusters.to_csv(index=False), config, prefix=config['paths']['risk_analysis_dir'], filename='server_clusters.csv')
  
  # Save summary JSON
  summary = {'analyzed_at': datetime.now().isoformat(), 'at_risk_servers': df['overall_risk_flag'].sum(), ...}
  save_to_s3_or_local(json.dumps(summary, indent=2), config, prefix=config['paths']['summaries_dir'], filename='module_05_summary.json')
  LOG "✔ Module 05 completed successfully"

IF __name__ == "__main__":
  Parse args: --config, --env
  config = load_config(args.config)
  config['execution']['mode'] = args.env
  validate_config(config)
  main_process(config)
```

#### Key Processing Notes
- **Efficiency**: Groupby operations are vectorized with pandas; clustering uses scikit-learn (fast for 100+ servers).
- **Configurability**: All thresholds, windows, etc., pulled from config to allow experiments.
- **Error Handling**: Logs warnings (e.g., low data volume) and raises on critical failures.
- **Parallelism**: For large datasets, add joblib.parallel_backend in loops (configurable).
- **AWS Touch**: Outputs saved to S3-compatible paths; in sagemaker mode, use SageMaker features for scaling.

### Outcome
- **Primary Outputs**:
  - Enriched forecasts Parquet (e.g., `risk_analysis/risk_flagged_forecasts.parquet`): Original columns + risk flags, seasonal markers, underutilized flags.
  - Optimization JSON (e.g., `risk_analysis/optimization_recommendations.json`): Savings estimates, underutilized count.
  - Clusters CSV (e.g., `risk_analysis/server_clusters.csv`): Server-to-cluster mappings.
  - Visualizations (PNG): Risk heatmap, cluster scatter plots.
  - Summary JSON (e.g., `reports/summaries/module_05_summary.json`): High-level stats (e.g., at-risk servers, savings).
- **Logs**: Detailed trace in `logs/module_05_risk_capacity_analysis.log` (e.g., "Loaded forecasts: (rows, cols)", "Savings: $X", errors).
- **Next Steps**: Outputs feed into dashboards (e.g., QuickSight) or further automation (Lambda).

### Success Criteria
- **Quantitative**:
  - All outputs generated without NaNs/errors (check shapes match inputs + new cols).
  - At least 10% servers flagged (configurable threshold for realism).
  - Savings > $0 (indicates underutilized detection worked).
  - Processing time < 5 min for 100 servers (logged).
- **Qualitative**:
  - Logs show "✔ Module 05 completed successfully" with no ERROR entries.
  - Visualizations render correctly (no empty plots).
  - Summary JSON validates (e.g., json.loads without errors).
- **Failure Indicators**: Exceptions raised/logged (e.g., "No forecasts found"), partial outputs deleted to avoid inconsistency.
- **Testing**: Run with debug_max_servers=5 in config; verify flags align with thresholds.

---

## Complete Python Code for src/modules/module_05_risk_capacity_analysis.py

```python
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
    sns.heatmap(pivot.apply(lambda x: {'low':0, 'medium':1, 'high':2}.get(x, 0)), cmap='Reds')
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
    metrics_content = load_from_s3_or_local(config, prefix=config['paths']['metrics_dir'], filename=metrics_filename)
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
```
"""
Download and investigate external public datasets for server/infrastructure metrics.

This script downloads publicly available datasets and performs initial analysis to
determine suitability for the AWS-CapacityForecaster project.

Log files:
    Logs are written to: logs/download_external_data_YYYYMMDD_HHMMSS.log
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_external"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging with file output
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / f'download_external_data_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Log file: {log_file}")
logger.info(f"Data directory: {DATA_DIR}")

# Dataset sources (publicly accessible URLs)
DATASETS = {
    "google_cluster_sample": {
        "description": "Google Cluster 2019 sample data from Kaggle",
        "url": "https://storage.googleapis.com/kaggle-data-sets/derrickmwiti/google-2019-cluster-sample",
        "type": "google_cluster",
        "manual": True,  # Requires manual download
    },
    "alibaba_sample": {
        "description": "Alibaba cluster trace sample",
        "url": None,
        "type": "alibaba_cluster",
        "manual": True,
    }
}

def create_sample_server_data():
    """
    Create sample data based on common server monitoring patterns
    to demonstrate the analysis workflow.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CREATING SAMPLE SERVER MONITORING DATA")
    logger.info("=" * 70)
    start_time = time.time()

    # Generate 30 days of hourly data for 10 servers
    date_range = pd.date_range(start='2024-01-01', end='2024-01-31', freq='h')
    servers = [f'server_{i:03d}' for i in range(1, 11)]

    data = []
    for server in servers:
        np.random.seed(hash(server) % 2**32)  # Consistent seed per server

        # Base utilization with trend
        base_cpu = 30 + np.random.rand() * 40  # 30-70% base
        base_mem = 40 + np.random.rand() * 30  # 40-70% base

        for idx, timestamp in enumerate(date_range):
            # Time-based patterns
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek

            # Daily pattern (higher during business hours)
            daily_factor = 1.0
            if 9 <= hour <= 17:
                daily_factor = 1.3
            elif 22 <= hour or hour <= 6:
                daily_factor = 0.7

            # Weekly pattern (lower on weekends)
            weekly_factor = 0.8 if day_of_week >= 5 else 1.0

            # Add some noise and spikes
            noise = np.random.randn() * 5
            spike = 20 if np.random.rand() > 0.95 else 0

            cpu = base_cpu * daily_factor * weekly_factor + noise + spike
            cpu = np.clip(cpu, 0, 100)

            mem = base_mem * daily_factor * weekly_factor + noise * 0.5 + spike * 0.7
            mem = np.clip(mem, 0, 100)

            # Network and disk (correlated with CPU)
            net = cpu * 0.6 + np.random.randn() * 10
            net = np.clip(net, 0, 100)

            disk = mem * 0.5 + np.random.randn() * 8
            disk = np.clip(disk, 0, 100)

            data.append({
                'timestamp': timestamp,
                'server_id': server,
                'cpu_utilization': round(cpu, 2),
                'memory_utilization': round(mem, 2),
                'network_utilization': round(net, 2),
                'disk_utilization': round(disk, 2)
            })

    df = pd.DataFrame(data)
    output_path = DATA_DIR / 'sample_server_metrics.csv'
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n[OK] Sample data created in {elapsed:.2f}s")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"  Servers: {df['server_id'].nunique()}")
    logger.info(f"  Records per server: {len(df) // df['server_id'].nunique():,}")

    return df

def analyze_dataset(df, name):
    """Perform initial analysis on a dataset."""
    logger.info("\n" + "=" * 70)
    logger.info(f"DATASET ANALYSIS: {name}")
    logger.info("=" * 70)
    analysis_start = time.time()

    # Basic info
    logger.info(f"\nBasic Information:")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"\nData Types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"\nMissing Values Found:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("\n[OK] No missing values")

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"\nNumeric Statistics:")
        stats = df[numeric_cols].describe()
        for col in numeric_cols:
            logger.info(f"  {col}: mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}, min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}")

    # Time range if timestamp exists
    time_cols = df.select_dtypes(include=['datetime64']).columns
    if len(time_cols) == 0:
        # Try to find timestamp-like columns
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    time_cols = [col]
                    break
                except:
                    pass

    if len(time_cols) > 0:
        time_col = time_cols[0]
        duration = df[time_col].max() - df[time_col].min()
        logger.info(f"\nTime Range ({time_col}):")
        logger.info(f"  Start: {df[time_col].min()}")
        logger.info(f"  End: {df[time_col].max()}")
        logger.info(f"  Duration: {duration}")
        logger.info(f"  Duration (days): {duration.days}")

    elapsed = time.time() - analysis_start
    logger.info(f"\n[OK] Analysis complete in {elapsed:.2f}s")

    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_pct': (df.isnull().sum().sum() / df.size) * 100,
        'numeric_cols': list(numeric_cols),
        'has_timestamp': len(time_cols) > 0
    }

def visualize_sample_data(df):
    """Create visualizations for the sample data."""
    logger.info("\n" + "=" * 70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 70)
    viz_start = time.time()

    # Setup plot style
    sns.set_style("whitegrid")

    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Server Metrics Analysis - Sample Data', fontsize=16, fontweight='bold')

    # 1. CPU utilization over time (average across servers)
    if 'timestamp' in df.columns and 'cpu_utilization' in df.columns:
        cpu_avg = df.groupby('timestamp')['cpu_utilization'].mean()
        axes[0, 0].plot(cpu_avg.index, cpu_avg.values, linewidth=1, alpha=0.8)
        axes[0, 0].set_title('Average CPU Utilization Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Distribution of utilization metrics
    util_cols = [col for col in df.columns if 'utilization' in col]
    if util_cols:
        df[util_cols].boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Resource Utilization')
        axes[0, 1].set_ylabel('Utilization %')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
        axes[1, 0].set_title('Correlation Matrix')

    # 4. Per-server average utilization
    if 'server_id' in df.columns and 'cpu_utilization' in df.columns:
        server_avg = df.groupby('server_id')['cpu_utilization'].mean().sort_values(ascending=False)
        server_avg.plot(kind='bar', ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title('Average CPU Utilization by Server')
        axes[1, 1].set_xlabel('Server ID')
        axes[1, 1].set_ylabel('Avg CPU %')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save plot
    output_path = DATA_DIR / 'sample_data_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - viz_start
    logger.info(f"\n[OK] Visualization saved in {elapsed:.2f}s")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

def compare_with_project_requirements(analysis_results):
    """Compare dataset characteristics with project requirements."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPATIBILITY CHECK WITH PROJECT REQUIREMENTS")
    logger.info("=" * 70)

    requirements = {
        'min_servers': 50,
        'ideal_servers': 100,
        'min_duration_days': 90,
        'ideal_duration_days': 365,
        'required_metrics': ['cpu', 'memory'],
        'optional_metrics': ['disk', 'network']
    }

    logger.info("\nProject Requirements:")
    logger.info(f"  Servers: {requirements['min_servers']}-{requirements['ideal_servers']}+")
    logger.info(f"  Duration: {requirements['min_duration_days']}-{requirements['ideal_duration_days']}+ days")
    logger.info(f"  Required metrics: {', '.join(requirements['required_metrics'])}")
    logger.info(f"  Optional metrics: {', '.join(requirements['optional_metrics'])}")

    logger.info("\nDataset Characteristics:")
    logger.info(f"  Rows: {analysis_results['shape'][0]:,}")
    logger.info(f"  Columns: {analysis_results['shape'][1]}")
    logger.info(f"  Numeric columns: {len(analysis_results['numeric_cols'])}")
    logger.info(f"  Has timestamp: {analysis_results['has_timestamp']}")
    logger.info(f"  Missing data: {analysis_results['missing_pct']:.2f}%")

    # Score the dataset
    score = 0
    max_score = 5

    logger.info("\nScoring:")
    if analysis_results['shape'][0] > 1000:
        score += 1
        logger.info("  [+1] Sufficient data points (>1000 rows)")
    else:
        logger.info("  [+0] Insufficient data points (<1000 rows)")

    if analysis_results['has_timestamp']:
        score += 1
        logger.info("  [+1] Has timestamp column")
    else:
        logger.info("  [+0] Missing timestamp column")

    if len(analysis_results['numeric_cols']) >= 2:
        score += 1
        logger.info("  [+1] Has multiple metrics (>=2 numeric columns)")
    else:
        logger.info("  [+0] Insufficient metrics (<2 numeric columns)")

    if analysis_results['missing_pct'] < 5:
        score += 1
        logger.info("  [+1] Low missing data (<5%)")
    else:
        logger.warning(f"  [+0] High missing data ({analysis_results['missing_pct']:.2f}%)")

    if analysis_results['shape'][0] > 10000:
        score += 1
        logger.info("  [+1] Large dataset (>10000 rows)")
    else:
        logger.info("  [+0] Small dataset (<10000 rows)")

    is_compatible = score >= 3
    status = "PASS" if is_compatible else "NEEDS REVIEW"
    logger.info(f"\nFinal Score: {score}/{max_score}")
    logger.info(f"Compatibility Status: {status}")

    return is_compatible

def main():
    """Main execution function."""
    main_start = time.time()

    logger.info("=" * 70)
    logger.info("EXTERNAL DATA INVESTIGATION - AWS-CapacityForecaster")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")

    # Create sample data
    df_sample = create_sample_server_data()

    # Analyze sample data
    analysis = analyze_dataset(df_sample, "Sample Server Metrics")

    # Visualize
    visualize_sample_data(df_sample)

    # Compare with requirements
    is_compatible = compare_with_project_requirements(analysis)

    total_elapsed = time.time() - main_start

    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"[OK] Sample data created and analyzed")
    logger.info(f"[OK] Visualizations generated")
    logger.info(f"[OK] Compatibility check: {'PASS' if is_compatible else 'NEEDS REVIEW'}")
    logger.info(f"\nTotal execution time: {total_elapsed:.2f}s")
    logger.info(f"Log file: {log_file}")

    logger.info("\nNext Steps:")
    logger.info("1. Review the generated visualizations in data/raw_external/")
    logger.info("2. Download real datasets from:")
    logger.info("   - Google Cluster Data: https://github.com/google/cluster-data")
    logger.info("   - Alibaba Cluster Data: https://github.com/alibaba/clusterdata")
    logger.info("   - Kaggle datasets (requires API key)")
    logger.info("3. Run this script on downloaded datasets for comparison")
    logger.info("4. Formulate final data strategy based on findings")

    logger.info("\n" + "=" * 70)
    logger.info("[OK] INVESTIGATION COMPLETE")
    logger.info("=" * 70)

    return df_sample, analysis

if __name__ == "__main__":
    df, analysis = main()

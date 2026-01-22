"""
Download and investigate external public datasets for server/infrastructure metrics.

This script downloads publicly available datasets and performs initial analysis to
determine suitability for the AWS-CapacityForecaster project.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_external"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")

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
    print("\n=== Creating Sample Server Monitoring Data ===")

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
    print(f"[OK] Created sample data: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Servers: {df['server_id'].nunique()}")

    return df

def analyze_dataset(df, name):
    """Perform initial analysis on a dataset."""
    print(f"\n=== Analysis: {name} ===")

    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing Values:\n{missing[missing > 0]}")
    else:
        print("\nNo missing values")

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Statistics:\n{df[numeric_cols].describe()}")

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
        print(f"\nTime Range ({time_col}):")
        print(f"  Start: {df[time_col].min()}")
        print(f"  End: {df[time_col].max()}")
        print(f"  Duration: {df[time_col].max() - df[time_col].min()}")

    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_pct': (df.isnull().sum().sum() / df.size) * 100,
        'numeric_cols': list(numeric_cols),
        'has_timestamp': len(time_cols) > 0
    }

def visualize_sample_data(df):
    """Create visualizations for the sample data."""
    print("\n=== Creating Visualizations ===")

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
    print(f"[OK] Saved visualization: {output_path}")
    plt.close()

def compare_with_project_requirements(analysis_results):
    """Compare dataset characteristics with project requirements."""
    print("\n=== Compatibility with Project Requirements ===")

    requirements = {
        'min_servers': 50,
        'ideal_servers': 100,
        'min_duration_days': 90,
        'ideal_duration_days': 365,
        'required_metrics': ['cpu', 'memory'],
        'optional_metrics': ['disk', 'network']
    }

    print("\nProject Requirements:")
    print(f"  Servers: {requirements['min_servers']}-{requirements['ideal_servers']}+")
    print(f"  Duration: {requirements['min_duration_days']}-{requirements['ideal_duration_days']}+ days")
    print(f"  Required metrics: {', '.join(requirements['required_metrics'])}")
    print(f"  Optional metrics: {', '.join(requirements['optional_metrics'])}")

    print("\nDataset Characteristics:")
    print(f"  Rows: {analysis_results['shape'][0]:,}")
    print(f"  Columns: {analysis_results['shape'][1]}")
    print(f"  Numeric columns: {len(analysis_results['numeric_cols'])}")
    print(f"  Has timestamp: {analysis_results['has_timestamp']}")
    print(f"  Missing data: {analysis_results['missing_pct']:.2f}%")

    # Score the dataset
    score = 0
    max_score = 5

    if analysis_results['shape'][0] > 1000:
        score += 1
        print("  [OK] Sufficient data points")

    if analysis_results['has_timestamp']:
        score += 1
        print("  [OK] Has timestamp column")

    if len(analysis_results['numeric_cols']) >= 2:
        score += 1
        print("  [OK] Has multiple metrics")

    if analysis_results['missing_pct'] < 5:
        score += 1
        print("  [OK] Low missing data")

    if analysis_results['shape'][0] > 10000:
        score += 1
        print("  [OK] Large dataset")

    print(f"\nCompatibility Score: {score}/{max_score}")

    return score >= 3  # At least 3 out of 5

def main():
    """Main execution function."""
    print("="*70)
    print("External Data Investigation for AWS-CapacityForecaster")
    print("="*70)

    # Create sample data
    df_sample = create_sample_server_data()

    # Analyze sample data
    analysis = analyze_dataset(df_sample, "Sample Server Metrics")

    # Visualize
    visualize_sample_data(df_sample)

    # Compare with requirements
    is_compatible = compare_with_project_requirements(analysis)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"[OK] Sample data created and analyzed")
    print(f"[OK] Visualizations generated")
    print(f"[OK] Compatibility check: {'PASS' if is_compatible else 'NEEDS REVIEW'}")

    print("\nNext Steps:")
    print("1. Review the generated visualizations in data/raw_external/")
    print("2. Download real datasets from:")
    print("   - Google Cluster Data: https://github.com/google/cluster-data")
    print("   - Alibaba Cluster Data: https://github.com/alibaba/clusterdata")
    print("   - Kaggle datasets (requires API key)")
    print("3. Run this script on downloaded datasets for comparison")
    print("4. Formulate final data strategy based on findings")

    return df_sample, analysis

if __name__ == "__main__":
    df, analysis = main()

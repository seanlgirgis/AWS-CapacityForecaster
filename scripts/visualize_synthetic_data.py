"""
Visualize generated synthetic dataset to verify quality and patterns.

Log files:
    Logs are written to: logs/visualize_synthetic_data_YYYYMMDD_HHMMSS.log
"""

import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging with file output
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / f'visualize_synthetic_data_{timestamp_str}.log'

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

# Start timing
script_start = time.time()

logger.info("=" * 70)
logger.info("SYNTHETIC DATA VISUALIZATION - AWS-CapacityForecaster")
logger.info("=" * 70)
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Log file: {log_file}")

# Setup
sns.set_style("whitegrid")
DATA_PATH = PROJECT_ROOT / "data/synthetic/server_metrics_full.csv.gz"
OUTPUT_DIR = PROJECT_ROOT / "reports/data_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"\n[PHASE 1/3] Loading data...")
load_start = time.time()
logger.info(f"  Source: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, compression='gzip', parse_dates=['timestamp'], index_col='timestamp')

load_elapsed = time.time() - load_start
logger.info(f"  [OK] Data loaded in {load_elapsed:.2f}s")
logger.info(f"  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
logger.info(f"  Unique servers: {df['server_id'].nunique()}")
logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

logger.info(f"\n[PHASE 2/3] Creating visualizations...")
viz_start = time.time()

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
logger.info("  Building 7-panel visualization dashboard...")

# 1. Time series: Average CPU across all servers
ax1 = fig.add_subplot(gs[0, :])
cpu_daily = df.groupby(df.index)['cpu_p95'].mean()
ax1.plot(cpu_daily.index, cpu_daily.values, linewidth=1, alpha=0.8, color='steelblue')
ax1.set_title('Average CPU P95 Utilization Over Time (All Servers)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('CPU %')
ax1.grid(True, alpha=0.3)

# 2. Distribution by server type
ax2 = fig.add_subplot(gs[1, 0])
type_means = df.groupby('server_type')['cpu_p95'].mean().sort_values(ascending=False)
ax2.bar(range(len(type_means)), type_means.values, color='steelblue')
ax2.set_xticks(range(len(type_means)))
ax2.set_xticklabels(type_means.index, rotation=45)
ax2.set_title('Avg CPU by Server Type')
ax2.set_xlabel('Server Type')
ax2.set_ylabel('Avg CPU %')
ax2.grid(axis='y', alpha=0.3)

# 3. Correlation heatmap
ax3 = fig.add_subplot(gs[1, 1])
metrics = ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
corr = df[metrics].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax3,
            cbar_kws={'label': 'Correlation'})
ax3.set_title('Metric Correlations')

# 4. Quarterly pattern
ax4 = fig.add_subplot(gs[1, 2])
df_sample = df[df['server_id'] == 'server_000'].copy()
df_sample['month'] = df_sample.index.month
monthly_avg = df_sample.groupby('month')['cpu_p95'].mean()
ax4.bar(monthly_avg.index, monthly_avg.values, color='coral')
ax4.set_title('Monthly CPU Pattern (Sample Server)')
ax4.set_xlabel('Month')
ax4.set_ylabel('Avg CPU %')
ax4.axhline(y=monthly_avg.mean(), color='red', linestyle='--', label='Annual Avg')
ax4.legend()

# 5. Business unit distribution
ax5 = fig.add_subplot(gs[2, 0])
bu_counts = df.groupby('business_unit')['server_id'].nunique()
ax5.pie(bu_counts.values, labels=bu_counts.index, autopct='%1.1f%%', startangle=90)
ax5.set_title('Servers by Business Unit')

# 6. Criticality vs Utilization
ax6 = fig.add_subplot(gs[2, 1])
crit_means = df.groupby('criticality')['mem_p95'].mean()
ax6.bar(range(len(crit_means)), crit_means.values, color='coral')
ax6.set_xticks(range(len(crit_means)))
ax6.set_xticklabels(crit_means.index)
ax6.set_title('Avg Memory by Criticality')
ax6.set_xlabel('Criticality')
ax6.set_ylabel('Avg Memory %')
ax6.grid(axis='y', alpha=0.3)

# 7. Weekly pattern
ax7 = fig.add_subplot(gs[2, 2])
df_week = df[df['server_type'] == 'web'].copy()
weekly_pattern = df_week.groupby('dayofweek')['cpu_p95'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
if len(weekly_pattern) > 0:
    ax7.plot(weekly_pattern.index, weekly_pattern.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax7.set_xticks(range(7))
    ax7.set_xticklabels(days)
    ax7.set_title('Weekly Pattern (Web Servers)')
    ax7.set_xlabel('Day of Week')
    ax7.set_ylabel('Avg CPU %')
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax7.transAxes)

plt.suptitle('Synthetic Server Metrics - Data Quality Analysis', fontsize=16, fontweight='bold', y=0.995)

output_path = OUTPUT_DIR / 'synthetic_data_overview.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

viz_elapsed = time.time() - viz_start
logger.info(f"  [OK] Visualization saved in {viz_elapsed:.2f}s")
logger.info(f"  Output file: {output_path}")
logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

# Generate statistics report
logger.info(f"\n[PHASE 3/3] Generating statistics report...")
logger.info("\n" + "=" * 70)
logger.info("STATISTICS REPORT")
logger.info("=" * 70)

logger.info(f"\nDataset Overview:")
logger.info(f"  Total records: {len(df):,}")
logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
logger.info(f"  Duration: {(df.index.max() - df.index.min()).days} days")
logger.info(f"  Unique servers: {df['server_id'].nunique()}")
logger.info(f"  Columns: {df.shape[1]}")

logger.info(f"\nServer Type Distribution:")
for stype, count in df.groupby('server_type')['server_id'].nunique().items():
    logger.info(f"  {stype}: {count} servers")

logger.info(f"\nMetric Statistics:")
for metric in metrics:
    logger.info(f"  {metric}: mean={df[metric].mean():.2f}, std={df[metric].std():.2f}, min={df[metric].min():.2f}, max={df[metric].max():.2f}, P95={df[metric].quantile(0.95):.2f}")

logger.info(f"\nData Quality:")
missing_count = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()
logger.info(f"  Missing values: {missing_count}")
logger.info(f"  Duplicate rows: {duplicate_count}")
logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if missing_count == 0 and duplicate_count == 0:
    logger.info("  [OK] Data quality checks passed")
else:
    logger.warning(f"  [!] Data quality issues detected")

total_elapsed = time.time() - script_start

logger.info("\n" + "=" * 70)
logger.info("EXECUTION SUMMARY")
logger.info("=" * 70)
logger.info(f"Total execution time: {total_elapsed:.2f}s")
logger.info(f"Output visualization: {output_path}")
logger.info(f"Log file: {log_file}")
logger.info("\n" + "=" * 70)
logger.info("[OK] VISUALIZATION COMPLETE")
logger.info("=" * 70)

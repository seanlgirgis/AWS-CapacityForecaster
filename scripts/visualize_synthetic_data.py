"""
Visualize generated synthetic dataset to verify quality and patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup
sns.set_style("whitegrid")
DATA_PATH = Path("data/synthetic/server_metrics_full.csv.gz")
OUTPUT_DIR = Path("reports/data_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, compression='gzip', parse_dates=['timestamp'], index_col='timestamp')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Servers: {df['server_id'].nunique()}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

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
print(f"\n[OK] Saved visualization: {output_path}")

# Generate statistics report
print("\n" + "="*70)
print("STATISTICS REPORT")
print("="*70)

print(f"\nDataset Overview:")
print(f"  Total records: {len(df):,}")
print(f"  Date range: {df.index.min()} to {df.index.max()}")
print(f"  Duration: {(df.index.max() - df.index.min()).days} days")
print(f"  Unique servers: {df['server_id'].nunique()}")
print(f"  Columns: {df.shape[1]}")

print(f"\nServer Type Distribution:")
for stype, count in df.groupby('server_type')['server_id'].nunique().items():
    print(f"  {stype}: {count} servers")

print(f"\nMetric Statistics:")
for metric in metrics:
    print(f"\n  {metric}:")
    print(f"    Mean: {df[metric].mean():.2f}")
    print(f"    Std: {df[metric].std():.2f}")
    print(f"    Min: {df[metric].min():.2f}")
    print(f"    Max: {df[metric].max():.2f}")
    print(f"    P95: {df[metric].quantile(0.95):.2f}")

print(f"\nData Quality:")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*70)
print("[OK] Analysis complete!")

# Data Generation Running Guide

## Overview

This guide covers running the data generation and visualization pipeline for the AWS-CapacityForecaster project.

**Pipeline Components:**
1. **Data Generation** (`src/data_generation.py`) - Generates synthetic server metrics
2. **Data Visualization** (`scripts/visualize_synthetic_data.py`) - Creates quality analysis visualizations
3. **External Data Investigation** (`scripts/download_external_data.py`) - Sample data for comparison

---

## Prerequisites

```powershell
# Navigate to project root
cd C:\pyproj\AWS-CapacityForecaster

# Activate virtual environment (if using one)
# .\.venv\Scripts\Activate.ps1

# Verify Python and dependencies
python --version
pip list | Select-String "pandas|numpy|matplotlib|seaborn"
```

---

## Step 1: Generate Synthetic Data

### Basic Command
```powershell
python src/data_generation.py --output data/synthetic/server_metrics_full.csv --compress
```

### Full Options
```powershell
python src/data_generation.py `
    --output data/synthetic/server_metrics_full.csv `
    --servers 120 `
    --granularity daily `
    --format csv `
    --compress `
    --metadata `
    --calendar-features
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output` | Required | Output file path |
| `--servers` | 120 | Number of servers (50-200) |
| `--granularity` | daily | `daily` or `hourly` |
| `--format` | csv | `csv` or `parquet` |
| `--compress` | False | Enable gzip compression |
| `--metadata` | True | Include business metadata |
| `--calendar-features` | True | Include calendar features |

### Expected Output
- **File:** `data/synthetic/server_metrics_full.csv.gz`
- **Size:** ~5-7 MB (compressed)
- **Records:** ~175,000 (120 servers x 1461 days)
- **Log:** `logs/data_generation_YYYYMMDD_HHMMSS.log`

### Sample Console Output
```
======================================================================
SYNTHETIC DATA GENERATION - AWS-CapacityForecaster
======================================================================
Start time: 2026-01-23 18:30:00
Log file: C:\pyproj\AWS-CapacityForecaster\logs\data_generation_20260123_183000.log

[PHASE 1/6] Loading configuration...
[PHASE 2/6] Generating timestamp range...
[PHASE 3/6] Assigning server archetypes...
[PHASE 4/6] Generating time-series metrics...
[PHASE 5/6] Building DataFrame...
[PHASE 6/6] DATA GENERATION COMPLETE
```

---

## Step 2: Visualize Generated Data

### Basic Command
```powershell
python scripts/visualize_synthetic_data.py --input data/synthetic/server_metrics_full.csv.gz
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `data/synthetic/server_metrics_full.csv.gz` | Input data file |

### Expected Output
- **Visualization:** `reports/data_quality/synthetic_data_overview.png`
- **Log:** `logs/visualize_synthetic_data_YYYYMMDD_HHMMSS.log`

### Generated Charts
1. Average CPU P95 Utilization Over Time (time series)
2. Average CPU by Server Type (bar chart)
3. Metric Correlations (heatmap)
4. Monthly CPU Pattern (seasonal)
5. Servers by Business Unit (pie chart)
6. Average Memory by Criticality (bar chart)
7. Weekly Pattern for Web Servers (line chart)

---

## Step 3: External Data Investigation (Optional)

Creates sample server data for comparison with real datasets.

### Command
```powershell
python scripts/download_external_data.py
```

### Expected Output
- **Sample Data:** `data/raw_external/sample_server_metrics.csv`
- **Visualization:** `data/raw_external/sample_data_analysis.png`
- **Log:** `logs/download_external_data_YYYYMMDD_HHMMSS.log`

---

## Complete Pipeline (Copy & Paste)

Run the full data generation and visualization pipeline:

```powershell
# Navigate to project
cd C:\pyproj\AWS-CapacityForecaster

# Step 1: Generate synthetic data
Write-Host "`n=== STEP 1: Generating Synthetic Data ===" -ForegroundColor Cyan
python src/data_generation.py --output data/synthetic/server_metrics_full.csv --compress

# Step 2: Visualize the data
Write-Host "`n=== STEP 2: Creating Visualizations ===" -ForegroundColor Cyan
python scripts/visualize_synthetic_data.py --input data/synthetic/server_metrics_full.csv.gz

# Step 3: (Optional) Generate external data samples
Write-Host "`n=== STEP 3: External Data Investigation ===" -ForegroundColor Cyan
python scripts/download_external_data.py

# Show results
Write-Host "`n=== RESULTS ===" -ForegroundColor Green
Write-Host "Data files:"
Get-ChildItem data/synthetic/*.gz | Format-Table Name, Length, LastWriteTime
Write-Host "`nVisualization:"
Get-ChildItem reports/data_quality/*.png | Format-Table Name, Length, LastWriteTime
Write-Host "`nLog files:"
Get-ChildItem logs/*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | Format-Table Name, LastWriteTime
```

---

## Log Files

All scripts create timestamped log files in the `logs/` directory:

| Script | Log File Pattern |
|--------|-----------------|
| `data_generation.py` | `logs/data_generation_YYYYMMDD_HHMMSS.log` |
| `visualize_synthetic_data.py` | `logs/visualize_synthetic_data_YYYYMMDD_HHMMSS.log` |
| `download_external_data.py` | `logs/download_external_data_YYYYMMDD_HHMMSS.log` |

### View Recent Logs
```powershell
# List recent logs
Get-ChildItem logs/*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 10

# View latest data generation log
Get-Content (Get-ChildItem logs/data_generation_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1)

# Tail a log file (watch in real-time)
Get-Content logs/data_generation_*.log -Wait -Tail 20
```

---

## Output Directory Structure

After running the complete pipeline:

```
AWS-CapacityForecaster/
├── data/
│   ├── synthetic/
│   │   └── server_metrics_full.csv.gz    # Main synthetic dataset
│   └── raw_external/
│       ├── sample_server_metrics.csv     # Sample comparison data
│       └── sample_data_analysis.png      # Sample data visualization
├── reports/
│   └── data_quality/
│       └── synthetic_data_overview.png   # 7-panel quality dashboard
└── logs/
    ├── data_generation_20260123_183000.log
    ├── visualize_synthetic_data_20260123_183500.log
    └── download_external_data_20260123_184000.log
```

---

## Dataset Schema

### Generated Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Date (index) |
| `server_id` | string | Server identifier (server_000 to server_119) |
| `cpu_p95` | float | 95th percentile CPU utilization (%) |
| `mem_p95` | float | 95th percentile memory utilization (%) |
| `disk_p95` | float | 95th percentile disk utilization (%) |
| `net_in_p95` | float | 95th percentile network in (Mbps) |
| `net_out_p95` | float | 95th percentile network out (Mbps) |
| `server_type` | string | Archetype: web, database, application, batch |
| `business_unit` | string | Business unit assignment |
| `criticality` | string | high, medium, low |
| `environment` | string | production, staging, development |
| `region` | string | AWS region |
| `dayofweek` | int | Day of week (0=Monday) |
| `month` | int | Month (1-12) |
| `quarter` | int | Quarter (1-4) |
| `is_weekend` | bool | Weekend flag |
| `is_month_end` | bool | Month-end flag |
| `is_quarter_end` | bool | Quarter-end flag |

---

## Loading Data in Python

```python
import pandas as pd

# Load the compressed CSV directly
df = pd.read_csv(
    'data/synthetic/server_metrics_full.csv.gz',
    compression='gzip',
    parse_dates=['timestamp'],
    index_col='timestamp'
)

print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Servers: {df['server_id'].nunique()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

---

## Troubleshooting

### ModuleNotFoundError
```powershell
# Install in development mode
pip install -e .
```

### FileNotFoundError for data file
```powershell
# Check if data exists
Get-ChildItem data/synthetic/

# Re-run data generation
python src/data_generation.py --output data/synthetic/server_metrics_full.csv --compress
```

### Empty logs directory
Logs are created at runtime. Run a script first:
```powershell
python scripts/download_external_data.py
Get-ChildItem logs/
```

---

## Quick Reference

```powershell
# Generate data (full)
python src/data_generation.py --output data/synthetic/server_metrics_full.csv --compress

# Generate data (quick test - fewer servers)
python src/data_generation.py --output data/synthetic/test.csv --servers 10 --compress

# Visualize data
python scripts/visualize_synthetic_data.py --input data/synthetic/server_metrics_full.csv.gz

# External data investigation
python scripts/download_external_data.py

# Check latest log
Get-Content (Get-ChildItem logs/*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1) | Select-Object -Last 50
```

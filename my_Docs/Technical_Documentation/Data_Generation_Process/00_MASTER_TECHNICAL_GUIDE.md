# AWS-CapacityForecaster: Master Technical Guide
## Synthetic Data Generation System - Complete Documentation

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Author:** Sean L. Girgis
**System Status:** Production-Ready

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Generation Process Flow](#3-data-generation-process-flow)
4. [Component Deep Dive](#4-component-deep-dive)
5. [Configuration System](#5-configuration-system)
6. [Code Files Reference](#6-code-files-reference)
7. [Function Catalog](#7-function-catalog)
8. [Input/Output Specifications](#8-inputoutput-specifications)
9. [Success Criteria & Validation](#9-success-criteria--validation)
10. [Troubleshooting Guide](#10-troubleshooting-guide)

---

## 1. Executive Overview

### 1.1 Purpose

This document provides complete technical documentation for the **Synthetic Data Generation System** within the AWS-CapacityForecaster project. The system generates enterprise-realistic server capacity metrics for machine learning model training and validation.

### 1.2 What This System Does

**Input:**
- Configuration parameters (YAML)
- Date ranges, server counts, metrics specifications

**Process:**
- Assigns server archetypes (Web, Database, Application, Batch)
- Generates correlated time-series metrics (CPU, Memory, Disk, Network)
- Applies realistic patterns (business hours, quarterly peaks, holidays)
- Adds business metadata and calendar features

**Output:**
- CSV/Parquet files with 175,320+ records
- Visualizations and quality reports
- Production-ready datasets for ML training

### 1.3 Key Features

✅ **4 Server Archetypes** - Web, Database, Application, Batch with unique behaviors
✅ **Correlated Metrics** - Cholesky decomposition for realistic metric relationships
✅ **Banking Seasonality** - Quarterly peaks, holiday effects, weekly patterns
✅ **Metadata Integration** - Business unit, criticality, region, server type
✅ **Scalable** - 50-200 servers, 1-5 years of data
✅ **Reproducible** - Seeded random generation for consistent results

### 1.4 Generated Dataset Summary

| Attribute | Value |
|-----------|-------|
| **Records** | 175,320 (120 servers × 1,461 days) |
| **Date Range** | 2022-01-01 to 2025-12-31 (4 years) |
| **Granularity** | Daily (hourly optional) |
| **Metrics** | 5 (CPU, Memory, Disk, Network In/Out) |
| **Metadata Columns** | 13 (server info, business context, calendar) |
| **File Size** | 3.04 MB (compressed) |
| **Quality** | 100% complete, 0 missing values in metrics |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ config.yaml  │  │ Environment  │  │ CLI Args     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Archetype Assignment Layer                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  assign_archetypes_to_fleet()                            │  │
│  │  - Distributes archetypes across 120 servers             │  │
│  │  - Web: 35%, App: 40%, DB: 15%, Batch: 10%             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Time Series Generation Layer                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Web       │  │  Database  │  │  Application│               │
│  │  Archetype │  │  Archetype │  │  Archetype  │   Batch       │
│  └────────────┘  └────────────┘  └────────────┘   Archetype    │
│         ↓                ↓                ↓            ↓         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  generate_correlated_metrics()                           │  │
│  │  - Cholesky decomposition for correlation               │  │
│  │  - Time-based factors (business hours, weekends)        │  │
│  │  - Seasonal patterns (quarterly, holidays)              │  │
│  │  - Spike modeling                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Metadata Enrichment Layer                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  generate_server_metadata()                              │  │
│  │  - Business unit assignment                              │  │
│  │  - Criticality levels                                    │  │
│  │  - Geographic regions                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  add_calendar_features()                                 │  │
│  │  - Year, month, quarter, day of week                    │  │
│  │  - Weekend flags, end-of-quarter flags                  │  │
│  │  - US holiday indicators                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Output & Validation Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  CSV.gz File │  │  Visualizations│  │  Quality     │         │
│  │  175K records│  │  7 panels      │  │  Reports     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
START
  │
  ├─→ Load Config (config.yaml)
  │     └─→ Validate parameters
  │
  ├─→ Generate Date Range
  │     └─→ 2022-01-01 to 2025-12-31 (daily)
  │
  ├─→ Assign Server Archetypes
  │     ├─→ Web servers (42)
  │     ├─→ Application servers (48)
  │     ├─→ Database servers (18)
  │     └─→ Batch servers (12)
  │
  ├─→ FOR EACH Server:
  │     │
  │     ├─→ Create Archetype Instance
  │     │     └─→ Load profile (base metrics, correlations, patterns)
  │     │
  │     └─→ FOR EACH Timestamp:
  │           │
  │           ├─→ Calculate Time Factors
  │           │     ├─→ Business hours multiplier
  │           │     ├─→ Weekend adjustment
  │           │     ├─→ Quarterly peak factor
  │           │     └─→ Holiday effect
  │           │
  │           ├─→ Generate Correlated Metrics
  │           │     ├─→ Build correlation matrix (4×4)
  │           │     ├─→ Cholesky decomposition
  │           │     ├─→ Generate random vector
  │           │     ├─→ Transform to correlated values
  │           │     └─→ Scale by variance + add to base
  │           │
  │           ├─→ Apply Spike Logic
  │           │     └─→ Random spike if probability < threshold
  │           │
  │           ├─→ Clip to Valid Ranges
  │           │     ├─→ CPU/Mem/Disk: 0-100%
  │           │     └─→ Network: 0-1000 Mbps
  │           │
  │           └─→ Create Record
  │                 └─→ {timestamp, server_id, cpu, mem, disk, net_in, net_out}
  │
  ├─→ Build DataFrame (175,320 rows × 6 columns)
  │
  ├─→ Add Business Metadata
  │     ├─→ Generate 120 server metadata records
  │     └─→ Merge on server_id
  │
  ├─→ Add Calendar Features
  │     ├─→ Extract year, month, quarter from timestamp
  │     ├─→ Calculate day of week, weekend flags
  │     ├─→ Identify end-of-quarter dates
  │     └─→ Mark US holidays
  │
  ├─→ Validate Data Quality
  │     ├─→ Check for missing values
  │     ├─→ Verify ranges (0-100%)
  │     └─→ Log statistics
  │
  └─→ Save Output
        ├─→ Compress to CSV.gz (3.04 MB)
        └─→ Generate visualizations
              └─→ 7-panel dashboard

END
```

### 2.3 Module Dependency Graph

```
                    ┌────────────────┐
                    │  config.yaml   │
                    └────────┬───────┘
                             │
                    ┌────────▼───────┐
                    │  src/utils/    │
                    │  config.py     │
                    └────────┬───────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
  ┌────────▼─────────┐ ┌────▼──────┐ ┌───────▼────────┐
  │ src/utils/       │ │ src/utils/│ │ src/utils/     │
  │ server_          │ │ data_     │ │ ml_utils.py    │
  │ archetypes.py    │ │ utils.py  │ │ (validation)   │
  └────────┬─────────┘ └────┬──────┘ └────────────────┘
           │                 │
           └────────┬────────┘
                    │
           ┌────────▼──────────┐
           │  src/             │
           │  data_generation  │
           │  .py              │
           └────────┬──────────┘
                    │
           ┌────────┴────────┐
           │                 │
  ┌────────▼─────┐  ┌───────▼────────┐
  │ data/        │  │ scripts/       │
  │ synthetic/   │  │ visualize_     │
  │ *.csv.gz     │  │ synthetic_data │
  └──────────────┘  └────────────────┘
```

---

## 3. Data Generation Process Flow

### 3.1 End-to-End Process Timeline

```
Time: 0s
├─ START: python -m src.data_generation
├─ [0.1s] Load configuration from config.yaml
├─ [0.2s] Validate parameters (120 servers, 4 years)
├─ [0.3s] Generate date range (1,461 timestamps)
├─ [0.5s] Assign archetypes to 120 servers
│
Time: 1s
├─ [1-15s] Generate metrics for all servers
│   ├─ Progress: server_000... (0%)
│   ├─ Progress: server_020... (16%)
│   ├─ Progress: server_040... (33%)
│   ├─ Progress: server_060... (50%)
│   ├─ Progress: server_080... (66%)
│   ├─ Progress: server_100... (83%)
│   └─ Complete: 175,320 records generated
│
Time: 15s
├─ [15-16s] Build DataFrame (17.89 MB in memory)
├─ [16-17s] Generate metadata (120 records)
├─ [17-18s] Merge metadata with metrics
├─ [18-19s] Add calendar features (7 columns)
├─ [19-20s] Validate data quality
│   ├─ Missing values: 0
│   ├─ Valid ranges: 100%
│   └─ Statistics: MAE, std, min, max
│
Time: 20s
├─ [20-22s] Compress and save CSV.gz (3.04 MB)
└─ [22s] COMPLETE
```

### 3.2 Detailed Function Call Stack

```
main()
  │
  ├─→ parse_args()  # CLI argument parsing
  │     └─→ Returns: {output, servers, years, ...}
  │
  ├─→ generate_full_dataset()
  │     │
  │     ├─→ get_data_config()  # Load from config.yaml
  │     │     └─→ Returns: {num_servers: 120, start_date: '2022-01-01', ...}
  │     │
  │     ├─→ pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')
  │     │     └─→ Returns: DatetimeIndex with 1,461 dates
  │     │
  │     ├─→ assign_archetypes_to_fleet(num_servers=120)
  │     │     │
  │     │     ├─→ Calculate counts: web=42, app=48, db=18, batch=12
  │     │     └─→ Returns: {'server_000': 'web', 'server_001': 'web', ...}
  │     │
  │     ├─→ FOR each server_id in archetype_assignments:
  │     │     │
  │     │     ├─→ get_archetype(server_type='web', server_id='server_000')
  │     │     │     │
  │     │     │     ├─→ Create ServerArchetype instance
  │     │     │     ├─→ Load profile from ARCHETYPE_PROFILES
  │     │     │     └─→ Returns: ServerArchetype object
  │     │     │
  │     │     └─→ FOR each timestamp in date_range:
  │     │           │
  │     │           ├─→ archetype.get_time_factor(timestamp)
  │     │           │     │
  │     │           │     ├─→ hour = timestamp.hour (0-23)
  │     │           │     ├─→ dayofweek = timestamp.dayofweek (0-6)
  │     │           │     ├─→ IF 9 <= hour <= 17: bh_factor = profile.business_hours_factor
  │     │           │     ├─→ IF dayofweek >= 5: weekend_factor = profile.weekend_factor
  │     │           │     └─→ Returns: bh_factor * weekend_factor
  │     │           │
  │     │           ├─→ _get_quarterly_peak_factor(timestamp, config)
  │     │           │     │
  │     │           │     ├─→ IF month in [3,6,9,12] AND day > (days_in_month - 5):
  │     │           │     │     └─→ Returns: 1.0 + (0.3 * proximity_to_quarter_end)
  │     │           │     └─→ ELSE: Returns: 1.0
  │     │           │
  │     │           ├─→ _get_holiday_factor(timestamp, config)
  │     │           │     │
  │     │           │     ├─→ IF month==1 AND day==1: Returns: 0.5 (New Year)
  │     │           │     ├─→ IF month==12 AND day>=25: Returns: 0.6 (Xmas week)
  │     │           │     └─→ ELSE: Returns: 1.0
  │     │           │
  │     │           ├─→ combined_factor = time_factor * qtr_factor * holiday_factor
  │     │           │
  │     │           ├─→ trend_factor = timestamp_index / total_timestamps
  │     │           │
  │     │           ├─→ archetype.generate_correlated_metrics(timestamp, combined_factor, trend_factor)
  │     │           │     │
  │     │           │     ├─→ Build correlation matrix (4×4):
  │     │           │     │     [1.0,  cpu_mem_corr,  0.1,  cpu_net_corr]
  │     │           │     │     [cpu_mem_corr,  1.0,  mem_disk_corr,  0.2]
  │     │           │     │     [0.1,  mem_disk_corr,  1.0,  0.3]
  │     │           │     │     [cpu_net_corr,  0.2,  0.3,  1.0]
  │     │           │     │
  │     │           │     ├─→ np.linalg.cholesky(corr_matrix)
  │     │           │     │     └─→ Returns: Lower triangular matrix L
  │     │           │     │
  │     │           │     ├─→ z = self.rng.randn(4)  # Independent standard normal
  │     │           │     │
  │     │           │     ├─→ correlated = L @ z  # Matrix multiplication
  │     │           │     │
  │     │           │     ├─→ cpu = base_cpu * time_factor * (1 + trend) + correlated[0] * variance
  │     │           │     ├─→ memory = base_mem * time_factor * (1 + trend) + correlated[1] * variance
  │     │           │     ├─→ disk = base_disk * (1 + trend) + correlated[2] * variance
  │     │           │     ├─→ network = base_net * time_factor + correlated[3] * variance
  │     │           │     │
  │     │           │     ├─→ IF random() < spike_probability:
  │     │           │     │     ├─→ cpu *= spike_magnitude
  │     │           │     │     ├─→ memory *= (spike_magnitude * 0.7)
  │     │           │     │     └─→ network *= (spike_magnitude * 0.8)
  │     │           │     │
  │     │           │     ├─→ Clip all values to valid ranges (0-100%, 0-1000 Mbps)
  │     │           │     │
  │     │           │     └─→ Returns: {cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95}
  │     │           │
  │     │           └─→ Append record to all_data list
  │     │
  │     ├─→ df = pd.DataFrame(all_data)
  │     │     └─→ Returns: DataFrame with 175,320 rows × 6 columns
  │     │
  │     ├─→ generate_server_metadata(n_servers=120)
  │     │     │
  │     │     ├─→ FOR each server_id:
  │     │     │     ├─→ Assign random business_unit from ['Trading', 'Retail', 'Compliance', 'IT']
  │     │     │     ├─→ Assign random criticality from ['High', 'Medium', 'Low']
  │     │     │     ├─→ Assign random region from ['US-East', 'US-West', 'EU', 'Asia']
  │     │     │     └─→ Generate app_name: f"{bu}-app-{i}"
  │     │     │
  │     │     └─→ Returns: DataFrame with 120 rows × 5 columns
  │     │
  │     ├─→ metadata_df['server_type'] = metadata_df['server_id'].map(archetype_assignments)
  │     │
  │     ├─→ df = df.merge(metadata_df, on='server_id')
  │     │     └─→ Returns: DataFrame with 175,320 rows × 11 columns
  │     │
  │     ├─→ add_calendar_features(df, date_col='timestamp')
  │     │     │
  │     │     ├─→ df['year'] = df['timestamp'].dt.year
  │     │     ├─→ df['month'] = df['timestamp'].dt.month
  │     │     ├─→ df['quarter'] = df['timestamp'].dt.quarter
  │     │     ├─→ df['dayofweek'] = df['timestamp'].dt.dayofweek
  │     │     ├─→ df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
  │     │     ├─→ df['is_eoq'] = df['timestamp'].dt.is_quarter_end.astype(int)
  │     │     ├─→ df['is_holiday'] = df['timestamp'].apply(check_us_holiday)
  │     │     │
  │     │     └─→ Returns: DataFrame with 175,320 rows × 18 columns
  │     │
  │     └─→ Returns: Final DataFrame
  │
  ├─→ save_dataset(df, output_path='data/synthetic/server_metrics_full.csv', compress=True)
  │     │
  │     ├─→ df.to_csv(output_path + '.gz', compression='gzip')
  │     └─→ Log file size and completion
  │
  └─→ COMPLETE

```

---

## 4. Component Deep Dive

### 4.1 Server Archetype System (`src/utils/server_archetypes.py`)

#### 4.1.1 Purpose

The archetype system creates **heterogeneous infrastructure** by defining 4 distinct server types with unique:
- Resource utilization patterns
- Correlation structures
- Time-based behaviors
- Spike characteristics

#### 4.1.2 Archetype Profiles

**Profile Structure:**
```python
@dataclass
class ArchetypeProfile:
    name: str

    # Base metrics (mean utilization)
    base_cpu: float          # e.g., 45% for web servers
    base_memory: float       # e.g., 35% for web servers
    base_disk: float
    base_network: float

    # Variance (standard deviation)
    cpu_variance: float      # e.g., 15% std dev
    memory_variance: float
    disk_variance: float
    network_variance: float

    # Correlations (Pearson correlation coefficients)
    cpu_memory_correlation: float     # e.g., 0.5 for web
    cpu_network_correlation: float    # e.g., 0.8 for web (high!)
    memory_disk_correlation: float    # e.g., 0.7 for database

    # Time-based multipliers
    business_hours_factor: float      # e.g., 1.6x during 9-5
    weekend_factor: float             # e.g., 0.5x on weekends

    # Spike modeling
    spike_probability: float          # e.g., 0.03 (3% of time)
    spike_magnitude: float            # e.g., 1.8x (80% increase)

    # Growth trend
    monthly_growth_rate: float        # e.g., 0.5% per month
```

**Web Server Profile:**
```python
ServerType.WEB: ArchetypeProfile(
    name="Web Server",
    base_cpu=45.0,           # Moderate CPU for request processing
    base_memory=35.0,        # Lower memory (stateless)
    base_disk=20.0,          # Minimal disk I/O
    base_network=150.0,      # High network (HTTP traffic)

    cpu_variance=15.0,
    memory_variance=8.0,
    disk_variance=5.0,
    network_variance=50.0,

    cpu_memory_correlation=0.5,    # Moderate: CPU driven by requests
    cpu_network_correlation=0.8,   # STRONG: requests drive both CPU and network
    memory_disk_correlation=0.2,   # Weak: little caching

    business_hours_factor=1.6,     # HIGH sensitivity to business hours
    weekend_factor=0.5,            # 50% reduction on weekends

    spike_probability=0.03,        # 3% chance per hour
    spike_magnitude=1.8,           # 80% increase during spike

    monthly_growth_rate=0.5,       # 0.5% growth per month
)
```

**Database Server Profile:**
```python
ServerType.DATABASE: ArchetypeProfile(
    name="Database Server",
    base_cpu=35.0,           # Lower CPU (optimized queries)
    base_memory=70.0,        # HIGH memory (caching, buffer pools)
    base_disk=55.0,          # HIGH disk I/O
    base_network=100.0,      # Moderate network

    cpu_variance=12.0,
    memory_variance=10.0,
    disk_variance=15.0,
    network_variance=30.0,

    cpu_memory_correlation=0.6,    # Memory pressure affects CPU
    cpu_network_correlation=0.4,   # Moderate coupling
    memory_disk_correlation=0.7,   # STRONG: memory pressure → swapping → disk I/O

    business_hours_factor=1.3,     # Moderate sensitivity
    weekend_factor=0.7,            # Still active on weekends

    spike_probability=0.01,        # LOW: steady-state operation
    spike_magnitude=1.4,           # Smaller spikes

    monthly_growth_rate=1.0,       # Data grows steadily
)
```

**Application Server Profile:**
```python
ServerType.APPLICATION: ArchetypeProfile(
    name="Application Server",
    base_cpu=50.0,           # Balanced
    base_memory=55.0,        # Balanced
    base_disk=30.0,          # Balanced
    base_network=120.0,      # Balanced

    cpu_variance=18.0,       # Higher variability
    memory_variance=15.0,
    disk_variance=10.0,
    network_variance=40.0,

    cpu_memory_correlation=0.7,    # Strong coupling (stateful apps)
    cpu_network_correlation=0.6,   # Moderate
    memory_disk_correlation=0.4,   # Some caching

    business_hours_factor=1.5,     # Strong business hours pattern
    weekend_factor=0.6,            # 40% reduction on weekends

    spike_probability=0.02,        # Moderate spikes
    spike_magnitude=1.6,

    monthly_growth_rate=0.8,
)
```

**Batch Processing Server Profile:**
```python
ServerType.BATCH: ArchetypeProfile(
    name="Batch Processing Server",
    base_cpu=30.0,           # LOW baseline (idle between jobs)
    base_memory=45.0,        # Moderate
    base_disk=40.0,          # High I/O during processing
    base_network=80.0,       # Lower network

    cpu_variance=25.0,       # VERY HIGH variance (spiky workload)
    memory_variance=12.0,
    disk_variance=20.0,
    network_variance=35.0,

    cpu_memory_correlation=0.4,    # Weaker: batch jobs are diverse
    cpu_network_correlation=0.3,
    memory_disk_correlation=0.5,

    business_hours_factor=0.8,     # INVERSE: lower during business hours
    weekend_factor=1.2,            # HIGHER on weekends (batch windows)

    spike_probability=0.08,        # VERY HIGH: 8% (scheduled jobs)
    spike_magnitude=2.5,           # LARGE: 150% increase

    monthly_growth_rate=0.3,       # Slower growth
)
```

#### 4.1.3 Correlation Matrix Construction

For each archetype, we build a **4×4 correlation matrix** for [CPU, Memory, Disk, Network]:

```python
corr_matrix = np.array([
    [1.0, cpu_memory_corr, 0.1, cpu_network_corr],
    [cpu_memory_corr, 1.0, memory_disk_corr, 0.2],
    [0.1, memory_disk_corr, 1.0, 0.3],
    [cpu_network_corr, 0.2, 0.3, 1.0]
])
```

**Cholesky Decomposition:**
To generate correlated random variables:

1. Decompose correlation matrix: `L = cholesky(corr_matrix)`
2. Generate independent random vector: `z ~ N(0, 1)` (4 values)
3. Transform to correlated: `correlated = L @ z`
4. Scale by variance: `metric = base + correlated[i] * variance`

**Example for Web Server:**
```
corr_matrix = [
    [1.0,  0.5,  0.1,  0.8],  # CPU
    [0.5,  1.0,  0.2,  0.2],  # Memory
    [0.1,  0.2,  1.0,  0.3],  # Disk
    [0.8,  0.2,  0.3,  1.0]   # Network
]

Cholesky(corr_matrix) = L = [
    [1.0,   0.0,   0.0,   0.0],
    [0.5,   0.866, 0.0,   0.0],
    [0.1,   0.173, 0.970, 0.0],
    [0.8,  -0.260, 0.231, 0.503]
]

If z = [0.5, -0.3, 0.8, -0.2], then:
correlated = L @ z = [0.5, 0.11, -0.023, 0.45]

cpu = 45 * time_factor + 0.5 * 15 = 45 * time_factor + 7.5
mem = 35 * time_factor + 0.11 * 8 = 35 * time_factor + 0.88
disk = 20 + (-0.023) * 5 = 20 - 0.12
network = 150 * time_factor + 0.45 * 50 = 150 * time_factor + 22.5
```

This ensures that when CPU is high, Network is also likely high (0.8 correlation).

---

*[Document continues in next file due to length...]*

---

## Navigation

📄 **Current:** Master Technical Guide
📄 **Next:** [01_Configuration_System.md](01_Configuration_System.md)
📄 **Next:** [02_Code_Walkthrough.md](02_Code_Walkthrough.md)
📄 **Next:** [03_Function_Catalog.md](03_Function_Catalog.md)

---

**Document End - Part 1 of 5**

# Function Catalog - Complete Reference

**Part 3 of 5** | [◀ Back to Index](README.md) | [Next: Flowcharts ▶](04_Flowcharts_Diagrams.md)

---

## Table of Contents

1. [Core Generation Functions](#1-core-generation-functions)
2. [Archetype Functions](#2-archetype-functions)
3. [Configuration Functions](#3-configuration-functions)
4. [Utility Functions](#4-utility-functions)
5. [Validation Functions](#5-validation-functions)

---

## 1. Core Generation Functions

### 1.1 `generate_full_dataset()`

**Location:** `src/data_generation.py:40`

**Purpose:** Main orchestration function that generates complete synthetic dataset

**Signature:**
```python
def generate_full_dataset(
    num_servers: Optional[int] = None,
    years_of_data: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = 'daily',
    include_metadata: bool = True,
    include_calendar_features: bool = True
) -> pd.DataFrame
```

**Input Parameters:**

| Parameter | Type | Default | Description | Required |
|-----------|------|---------|-------------|----------|
| `num_servers` | int | None (from config) | Number of servers (50-200) | No |
| `years_of_data` | int | None (from config) | Years of historical data (1-5) | No |
| `start_date` | str | None (from config) | Start date 'YYYY-MM-DD' | No |
| `end_date` | str | None (from config) | End date 'YYYY-MM-DD' | No |
| `granularity` | str | 'daily' | 'daily' or 'hourly' | No |
| `include_metadata` | bool | True | Add business metadata | No |
| `include_calendar_features` | bool | True | Add calendar features | No |

**Output:**

| Attribute | Type | Description |
|-----------|------|-------------|
| **Return** | pd.DataFrame | DataFrame with generated data |
| **Shape** | (175320, 18) | 120 servers × 1461 days × 18 columns |
| **Index** | DatetimeIndex | 'timestamp' column |
| **Memory** | ~18 MB | In-memory size |

**Output Columns:**

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `server_id` | str | server_000 - server_119 | Unique identifier |
| `cpu_p95` | float | 0-100 | CPU P95 utilization (%) |
| `mem_p95` | float | 0-100 | Memory P95 utilization (%) |
| `disk_p95` | float | 0-100 | Disk P95 utilization (%) |
| `net_in_p95` | float | 0-1000 | Network inbound P95 (Mbps) |
| `net_out_p95` | float | 0-600 | Network outbound P95 (Mbps) |
| `app_name` | str | - | Application name |
| `business_unit` | str | Trading/Retail/Compliance/IT | Business unit |
| `criticality` | str | High/Medium/Low | Criticality level |
| `region` | str | US-East/US-West/EU/Asia | Geographic region |
| `server_type` | str | web/database/application/batch | Archetype |
| `year` | int | 2022-2025 | Year |
| `month` | int | 1-12 | Month |
| `quarter` | int | 1-4 | Quarter |
| `dayofweek` | int | 0-6 | Day of week (0=Mon) |
| `is_weekend` | int | 0/1 | Weekend flag |
| `is_eoq` | int | 0/1 | End-of-quarter flag |
| `is_holiday` | int | 0/1 | US holiday flag |

**Usage Example:**
```python
from src.data_generation import generate_full_dataset

# Generate with defaults
df = generate_full_dataset()

# Generate custom
df = generate_full_dataset(
    num_servers=50,
    start_date='2023-01-01',
    end_date='2024-12-31',
    granularity='daily',
    include_metadata=True
)

# Access data
print(f"Shape: {df.shape}")
print(f"Servers: {df['server_id'].nunique()}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

**Success Criteria:**
- ✅ No missing values in metric columns
- ✅ All metrics within valid ranges
- ✅ Correlations match archetype specifications
- ✅ Seasonal patterns detectable
- ✅ Execution time < 60 seconds for 120 servers × 4 years

**Failure Modes:**
- `ValueError`: Invalid configuration parameters
- `MemoryError`: Insufficient memory for large datasets (hourly, 200+ servers)
- `FileNotFoundError`: Config file missing

**Performance:**
- **Time:** ~20 seconds for 120 servers × 4 years (daily)
- **Memory:** 18 MB peak
- **Scalability:** Linear with num_servers × timestamps

---

### 1.2 `save_dataset()`

**Location:** `src/data_generation.py:278`

**Purpose:** Save generated dataset to file with optional compression

**Signature:**
```python
def save_dataset(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    compress: bool = False
) -> None
```

**Input Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pd.DataFrame | - | DataFrame to save |
| `output_path` | str | - | Output file path |
| `format` | str | 'csv' | 'csv' or 'parquet' |
| `compress` | bool | False | Enable compression |

**Output:**
- **File:** CSV or Parquet file at `output_path`
- **Size:** ~3 MB (CSV.gz), ~2 MB (Parquet), ~18 MB (uncompressed CSV)
- **Compression Ratio:** ~6:1 for CSV.gz

**Usage Example:**
```python
# Save as compressed CSV
save_dataset(df, 'data/output.csv', compress=True)
# Creates: data/output.csv.gz

# Save as Parquet
save_dataset(df, 'data/output.parquet', format='parquet')
```

**Success Criteria:**
- ✅ File created at specified path
- ✅ File size within expected range
- ✅ Data integrity preserved (can be reloaded)

---

### 1.3 `_get_quarterly_peak_factor()`

**Location:** `src/data_generation.py:203`

**Purpose:** Calculate quarterly peak multiplier for banking workloads

**Signature:**
```python
def _get_quarterly_peak_factor(timestamp: datetime, config: Dict) -> float
```

**Input Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `timestamp` | datetime | Current timestamp |
| `config` | Dict | Data configuration |

**Algorithm:**
```
IF config['seasonality']['quarterly_peaks'] is False:
    RETURN 1.0

IF month in [3, 6, 9, 12]:  # Quarter-end months
    days_in_month = get_days_in_month(month, year)
    IF day >= (days_in_month - 4):  # Last 5 days
        days_from_end = days_in_month - day
        intensity = 1.0 + (0.3 * (5 - days_from_end) / 5)
        RETURN intensity  # Range: 1.0 - 1.3
    ELSE:
        RETURN 1.0
ELSE:
    RETURN 1.0
```

**Output:**

| Value | Meaning |
|-------|---------|
| 1.0 | Normal load |
| 1.06 | 5 days before quarter end |
| 1.12 | 4 days before quarter end |
| 1.18 | 3 days before quarter end |
| 1.24 | 2 days before quarter end |
| 1.30 | Last day of quarter (30% increase) |

**Example:**
```python
# March 31, 2024 (last day of Q1)
factor = _get_quarterly_peak_factor(pd.Timestamp('2024-03-31'), config)
# Returns: 1.30

# March 27, 2024 (5 days before)
factor = _get_quarterly_peak_factor(pd.Timestamp('2024-03-27'), config)
# Returns: 1.06

# March 15, 2024 (mid-month)
factor = _get_quarterly_peak_factor(pd.Timestamp('2024-03-15'), config)
# Returns: 1.0
```

---

### 1.4 `_get_holiday_factor()`

**Location:** `src/data_generation.py:239`

**Purpose:** Calculate holiday effect multiplier (reduced load)

**Signature:**
```python
def _get_holiday_factor(timestamp: datetime, config: Dict) -> float
```

**Holiday Mapping:**

| Holiday | Date | Reduction | Multiplier |
|---------|------|-----------|------------|
| New Year's Day | Jan 1 | 50% | 0.5 |
| Christmas Week | Dec 25-31 | 40% | 0.6 |
| Independence Day | Jul 4 | 30% | 0.7 |
| Thanksgiving Week | Nov 22-28 (approx) | 30% | 0.7 |
| Christmas Eve | Dec 24 | 30% | 0.7 |

**Output:**
- **Range:** 0.5 - 1.0
- **Default:** 1.0 (no holiday)

**Example:**
```python
# Christmas Day
factor = _get_holiday_factor(pd.Timestamp('2024-12-25'), config)
# Returns: 0.6 (40% reduction)

# Regular workday
factor = _get_holiday_factor(pd.Timestamp('2024-03-15'), config)
# Returns: 1.0 (no reduction)
```

---

## 2. Archetype Functions

### 2.1 `get_archetype()`

**Location:** `src/utils/server_archetypes.py:283`

**Purpose:** Factory function to create server archetype instance

**Signature:**
```python
def get_archetype(server_type: str, server_id: str) -> ServerArchetype
```

**Input Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_type` | str | 'web', 'database', 'application', 'batch' |
| `server_id` | str | Unique server identifier |

**Output:**
- **Type:** `ServerArchetype` instance
- **Attributes:** profile, seed, rng (random number generator)

**Supported Types:**

| Input String | Maps To | Archetype |
|--------------|---------|-----------|
| 'web' | ServerType.WEB | Web Server |
| 'database', 'db' | ServerType.DATABASE | Database Server |
| 'application', 'app' | ServerType.APPLICATION | Application Server |
| 'batch' | ServerType.BATCH | Batch Processing Server |

**Usage Example:**
```python
from src.utils.server_archetypes import get_archetype

# Create web server archetype
web_server = get_archetype('web', 'server_001')

# Generate metrics for a timestamp
timestamp = pd.Timestamp('2024-01-15 10:00:00')
time_factor = web_server.get_time_factor(timestamp)
metrics = web_server.generate_correlated_metrics(timestamp, time_factor, 0.5)

print(metrics)
# {'cpu_p95': 62.4, 'mem_p95': 48.2, 'disk_p95': 21.3,
#  'net_in_p95': 198.5, 'net_out_p95': 119.1}
```

**Raises:**
- `ValueError`: If server_type is not recognized

---

### 2.2 `ServerArchetype.generate_correlated_metrics()`

**Location:** `src/utils/server_archetypes.py:107`

**Purpose:** Generate correlated metrics using Cholesky decomposition

**Signature:**
```python
def generate_correlated_metrics(
    self,
    timestamp: datetime,
    time_factor: float = 1.0,
    trend_factor: float = 0.0
) -> Dict[str, float]
```

**Input Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `timestamp` | datetime | - | Current timestamp (for spike logic) |
| `time_factor` | float | 0.5-2.0 | Combined time multiplier |
| `trend_factor` | float | 0.0-1.0 | Linear growth (0=start, 1=end) |

**Algorithm:**

```
1. Build Correlation Matrix (4×4):
   ┌                                      ┐
   │ 1.0   cpu_mem   0.1      cpu_net    │
   │ cpu_mem  1.0    mem_disk   0.2      │
   │ 0.1   mem_disk  1.0       0.3       │
   │ cpu_net  0.2    0.3       1.0       │
   └                                      ┘

2. Cholesky Decomposition:
   L = cholesky(corr_matrix)

3. Generate Independent Random Vector:
   z = random_normal(0, 1, size=4)

4. Transform to Correlated:
   correlated = L @ z

5. Scale and Add to Base:
   cpu = base_cpu * time_factor * (1 + trend * growth_rate) + correlated[0] * cpu_variance
   mem = base_mem * time_factor * (1 + trend * growth_rate) + correlated[1] * mem_variance
   disk = base_disk * (1 + trend * growth_rate) + correlated[2] * disk_variance
   net = base_net * time_factor + correlated[3] * net_variance

6. Apply Spikes (probabilistic):
   IF random() < spike_probability:
       cpu *= spike_magnitude
       mem *= (spike_magnitude * 0.7)
       net *= (spike_magnitude * 0.8)

7. Clip to Valid Ranges:
   cpu = clip(cpu, 0, 100)
   mem = clip(mem, 0, 100)
   disk = clip(disk, 0, 100)
   net_in = clip(net, 0, 1000)
   net_out = clip(net * 0.6, 0, 600)
```

**Output:**

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `cpu_p95` | float | 0-100 | CPU P95 % |
| `mem_p95` | float | 0-100 | Memory P95 % |
| `disk_p95` | float | 0-100 | Disk P95 % |
| `net_in_p95` | float | 0-1000 | Network In Mbps |
| `net_out_p95` | float | 0-600 | Network Out Mbps |

**Example:**
```python
# Web server during business hours
web = get_archetype('web', 'server_001')
timestamp = pd.Timestamp('2024-03-15 14:00:00')  # Friday 2 PM
time_factor = 1.6  # Business hours
trend_factor = 0.25  # 25% through dataset

metrics = web.generate_correlated_metrics(timestamp, time_factor, trend_factor)
# Expected: Higher CPU and network due to time_factor
# Possible spike if probability triggers
```

**Statistical Properties:**
- **Mean:** Approximately base_metric * time_factor
- **Std Dev:** Approximately variance
- **Correlation:** Matches archetype profile correlation matrix
- **Spikes:** Occur with probability = spike_probability

---

### 2.3 `ServerArchetype.get_time_factor()`

**Location:** `src/utils/server_archetypes.py:176`

**Purpose:** Calculate time-based adjustment multiplier

**Signature:**
```python
def get_time_factor(self, timestamp: datetime) -> float
```

**Algorithm:**
```
hour = timestamp.hour  # 0-23
day_of_week = timestamp.dayofweek  # 0 (Mon) - 6 (Sun)

# Business hours factor
IF 9 <= hour <= 17:
    bh_factor = profile.business_hours_factor  # e.g., 1.6 for web
ELSE:
    bh_factor = 1.0

# Weekend factor
IF day_of_week >= 5:  # Saturday or Sunday
    weekend_factor = profile.weekend_factor  # e.g., 0.5 for web
ELSE:
    weekend_factor = 1.0

RETURN bh_factor * weekend_factor
```

**Output Examples:**

| Scenario | hour | dayofweek | Web Server | Batch Server |
|----------|------|-----------|------------|--------------|
| Weekday 10 AM | 10 | 1 (Tue) | 1.6 × 1.0 = **1.6** | 0.8 × 1.0 = **0.8** |
| Weekday 8 PM | 20 | 1 (Tue) | 1.0 × 1.0 = **1.0** | 1.0 × 1.0 = **1.0** |
| Saturday 2 PM | 14 | 5 (Sat) | 1.6 × 0.5 = **0.8** | 0.8 × 1.2 = **0.96** |
| Sunday 11 PM | 23 | 6 (Sun) | 1.0 × 0.5 = **0.5** | 1.0 × 1.2 = **1.2** |

**Key Insight:** Batch servers have **inverse pattern** (higher on weekends/off-hours).

---

### 2.4 `assign_archetypes_to_fleet()`

**Location:** `src/utils/server_archetypes.py:308`

**Purpose:** Distribute server archetypes across fleet

**Signature:**
```python
def assign_archetypes_to_fleet(
    num_servers: int,
    distribution: Dict[str, float] = None
) -> Dict[str, str]
```

**Input Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_servers` | int | - | Total servers (e.g., 120) |
| `distribution` | Dict | None | Proportion per archetype |

**Default Distribution:**
```python
{
    'web': 0.35,          # 35%
    'application': 0.40,  # 40%
    'database': 0.15,     # 15%
    'batch': 0.10,        # 10%
}
```

**Output:**
```python
{
    'server_000': 'web',
    'server_001': 'web',
    ...
    'server_041': 'web',        # 42 web servers
    'server_042': 'application',
    ...
    'server_089': 'application',  # 48 app servers
    'server_090': 'database',
    ...
    'server_107': 'database',    # 18 database servers
    'server_108': 'batch',
    ...
    'server_119': 'batch',       # 12 batch servers
}
```

**Example:**
```python
# Default distribution
assignments = assign_archetypes_to_fleet(120)
print(len([v for v in assignments.values() if v == 'web']))  # 42

# Custom distribution
custom_dist = {'web': 0.5, 'application': 0.3, 'database': 0.2}
assignments = assign_archetypes_to_fleet(100, custom_dist)
# web: 50, application: 30, database: 20
```

**Validation:**
- Distribution must sum to 1.0
- Raises `ValueError` if sum ≠ 1.0

---

## 3. Configuration Functions

### 3.1 `get_data_config()`

**Location:** `src/utils/config.py:120`

**Purpose:** Retrieve data section from configuration

**Signature:**
```python
def get_data_config() -> Dict
```

**Output:**
```python
{
    'num_servers': 120,
    'years_of_data': 4,
    'granularity': 'daily',
    'start_date': '2022-01-01',
    'end_date': '2025-12-31',
    'seasonality': {
        'weekly': True,
        'quarterly_peaks': True,
        'holiday_effect': True
    },
    'p95_ranges': {
        'cpu': [10.0, 95.0],
        'memory': [15.0, 92.0],
        ...
    }
}
```

**Usage:**
```python
from src.utils.config import get_data_config

config = get_data_config()
num_servers = config['num_servers']
start_date = config['start_date']
```

**Auto-loads config.yaml on first call**

---

### 3.2 `get_feature_engineering_config()`

**Location:** `src/utils/config.py:126`

**Purpose:** Retrieve feature engineering section

**Output:**
```python
{
    'lags': [1, 7, 30],
    'rolling_windows': [7, 14, 30],
    'external_regressors': ['is_end_of_quarter', 'is_month_end'],
    'anomaly_detection': {
        'z_score_threshold': 3.0,
        'knn_neighbors': 5
    }
}
```

---

## 4. Utility Functions

### 4.1 `generate_server_metadata()`

**Location:** `src/utils/data_utils.py:147`

**Purpose:** Generate business metadata for servers

**Signature:**
```python
def generate_server_metadata(
    n_servers: int = 80,
    business_units: List[str] = ['Trading', 'Retail', 'Compliance', 'IT'],
    criticalities: List[str] = ['High', 'Medium', 'Low'],
    regions: List[str] = ['US-East', 'US-West', 'EU', 'Asia'],
    seed: int = 42
) -> pd.DataFrame
```

**Output Columns:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `server_id` | str | 'server_042' | Identifier |
| `app_name` | str | 'Trading-app-7' | Application name |
| `business_unit` | str | 'Trading' | Business unit |
| `criticality` | str | 'High' | Criticality level |
| `region` | str | 'US-East' | Geographic region |

**Example:**
```python
metadata = generate_server_metadata(n_servers=120)
print(metadata.head())
#   server_id       app_name  business_unit criticality   region
# 0 server_000  Trading-app-0        Trading        High  US-East
# 1 server_001   Retail-app-1         Retail      Medium  US-West
```

---

### 4.2 `add_calendar_features()`

**Location:** `src/utils/data_utils.py:319`

**Purpose:** Add calendar-based features to DataFrame

**Signature:**
```python
def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = 'date',
    include_holidays: bool = True
) -> pd.DataFrame
```

**Output Columns Added:**

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `year` | int | 2022-2025 | Year |
| `month` | int | 1-12 | Month |
| `quarter` | int | 1-4 | Quarter |
| `dayofweek` | int | 0-6 | Day of week (0=Mon) |
| `is_weekend` | int | 0/1 | Weekend flag |
| `is_eoq` | int | 0/1 | End-of-quarter flag |
| `is_holiday` | int | 0/1 | US holiday flag |

**Holidays Detected:**
- New Year's Day (Jan 1)
- Independence Day (Jul 4)
- Thanksgiving (4th Thu of Nov)
- Christmas (Dec 25)

**Example:**
```python
df_with_features = add_calendar_features(df, date_col='timestamp')
print(df_with_features[['timestamp', 'is_weekend', 'is_eoq', 'is_holiday']].head())
```

---

## 5. Validation Functions

### 5.1 `validate_capacity_df()`

**Location:** `src/utils/data_utils.py:180`

**Purpose:** Validate data quality of capacity DataFrame

**Signature:**
```python
def validate_capacity_df(
    df: pd.DataFrame,
    required_cols: List[str] = ['cpu_p95', 'mem_p95'],
    missing_threshold: float = 0.05
) -> bool
```

**Validation Checks:**

| Check | Criterion | Action if Failed |
|-------|-----------|------------------|
| Required columns | All present | Raise ValueError |
| Data types | Numeric for metrics | Raise TypeError |
| Missing values | < 5% | Raise ValueError |
| Value ranges | 0-100 for % | Log warning |
| Negative values | None | Raise ValueError |
| Duplicates | Check index | Log warning |

**Returns:**
- `True` if all checks pass
- `Raises exception` if critical checks fail

---

## Navigation

📄 [◀ Back to Index](README.md)
📄 [Back to Master Guide](00_MASTER_TECHNICAL_GUIDE.md)
📄 [Next: Flowcharts ▶](04_Flowcharts_Diagrams.md)

---

**Document End - Part 3 of 5**

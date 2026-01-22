# Configuration System Documentation

**Part 2 of 5** | [◀ Back to Master Guide](00_MASTER_TECHNICAL_GUIDE.md) | [Next: Code Walkthrough ▶](02_Code_Walkthrough.md)

---

## Table of Contents

1. [Configuration File Structure](#1-configuration-file-structure)
2. [Configuration Parameters](#2-configuration-parameters)
3. [Configuration Loading Process](#3-configuration-loading-process)
4. [Parameter Validation](#4-parameter-validation)
5. [Environment Overrides](#5-environment-overrides)

---

## 1. Configuration File Structure

### 1.1 File Location

**Primary Configuration:**
- **Path:** `C:\pyproj\AWS-CapacityForecaster\config\config.yaml`
- **Format:** YAML
- **Size:** 107 lines
- **Encoding:** UTF-8

### 1.2 Configuration Hierarchy

```
config.yaml
├── project (metadata)
│   ├── name
│   ├── version
│   └── description
│
├── aws (cloud configuration)
│   ├── region
│   ├── bucket_name
│   ├── prefixes (raw, processed, features)
│   ├── sagemaker_role_arn
│   └── athena_database
│
├── data (generation parameters) ★ USED BY DATA GENERATION
│   ├── num_servers
│   ├── years_of_data
│   ├── start_date / end_date
│   ├── seasonality settings
│   └── p95_ranges
│
├── ml (model parameters)
│   ├── forecast_horizon_months
│   ├── models (Prophet, RandomForest, etc.)
│   └── evaluation metrics
│
├── feature_engineering ★ USED BY DATA GENERATION
│   ├── lags
│   ├── rolling_windows
│   └── external_regressors
│
├── risk_analysis
│   ├── thresholds
│   └── clustering parameters
│
├── paths (local directories)
├── visualization (plot settings)
├── logging (log configuration)
└── execution (runtime settings)
```

---

## 2. Configuration Parameters

### 2.1 Data Section (Primary for Generation)

```yaml
data:
  num_servers: 120                    # Total servers to generate
  years_of_data: 4                    # Years of historical data
  granularity: daily                  # 'daily' or 'hourly'
  start_date: "2022-01-01"           # Start of time series
  end_date: "2025-12-31"             # End of time series

  seasonality:
    weekly: true                      # Enable weekly patterns
    quarterly_peaks: true             # End-of-quarter surges
    holiday_effect: true              # Holiday reductions

  p95_ranges:                         # Min-Max ranges for P95 metrics
    cpu: [10.0, 95.0]                # 10-95%
    memory: [15.0, 92.0]             # 15-92%
    disk: [5.0, 85.0]                # 5-85%
    network_in: [20.0, 500.0]        # 20-500 Mbps
    network_out: [10.0, 300.0]       # 10-300 Mbps
```

**Parameter Details:**

| Parameter | Type | Default | Valid Range | Used By |
|-----------|------|---------|-------------|---------|
| `num_servers` | int | 120 | 50-200 | `generate_full_dataset()` |
| `years_of_data` | int | 4 | 1-5 | Date range calculation |
| `granularity` | str | 'daily' | 'daily', 'hourly' | `pd.date_range()` |
| `start_date` | str | '2022-01-01' | YYYY-MM-DD | `pd.date_range()` |
| `end_date` | str | '2025-12-31' | YYYY-MM-DD | `pd.date_range()` |
| `seasonality.weekly` | bool | true | true/false | `get_time_factor()` |
| `seasonality.quarterly_peaks` | bool | true | true/false | `_get_quarterly_peak_factor()` |
| `seasonality.holiday_effect` | bool | true | true/false | `_get_holiday_factor()` |

**P95 Ranges Explanation:**
- **Purpose:** Define realistic min/max bounds for metrics
- **Usage:** Currently NOT enforced (archetypes define their own ranges)
- **Future Use:** Will be used for validation and anomaly detection

### 2.2 Feature Engineering Section

```yaml
feature_engineering:
  lags: [1, 7, 30]                   # Lag periods in days
  rolling_windows: [7, 14, 30]       # Rolling window sizes
  external_regressors:
    - is_end_of_quarter
    - is_month_end
  anomaly_detection:
    z_score_threshold: 3.0
    knn_neighbors: 5
```

**Used By:**
- `add_calendar_features()` - Creates `is_end_of_quarter`, `is_month_end` flags
- Future ML pipeline will use `lags` and `rolling_windows`

### 2.3 Execution Section

```yaml
execution:
  mode: local                         # 'local', 'sagemaker_processing', 'lambda'
  parallel_workers: 4                 # For joblib parallelization
  random_seed: 42                     # Reproducibility seed
```

**Random Seed:**
- **Value:** 42
- **Purpose:** Deterministic random number generation
- **Scope:** NOT currently used in data generation (archetypes use server_id-based seeds)
- **Future Use:** Will control train/test splits in ML pipeline

---

## 3. Configuration Loading Process

### 3.1 Load Flow Diagram

```
START: get_data_config()
  │
  ├─→ Check if CONFIG global variable is loaded
  │     │
  │     ├─→ IF loaded: Return CONFIG['data']
  │     │
  │     └─→ IF not loaded:
  │           │
  │           ├─→ Find config file
  │           │     │
  │           │     ├─→ Try: PROJECT_ROOT / 'config' / 'config.yaml'
  │           │     ├─→ Try: os.environ.get('CAPACITY_FORECASTER_CONFIG')
  │           │     └─→ Raise FileNotFoundError if not found
  │           │
  │           ├─→ Open and parse YAML
  │           │     │
  │           │     ├─→ with open(config_path) as f:
  │           │     ├─→ yaml.safe_load(f)
  │           │     └─→ Handle YAML syntax errors
  │           │
  │           ├─→ Apply environment variable overrides
  │           │     │
  │           │     ├─→ FOR each env var starting with 'CF_':
  │           │     │     │
  │           │     │     ├─→ Parse path: CF_DATA_NUM_SERVERS → data.num_servers
  │           │     │     ├─→ Parse value: Try int → float → bool → str
  │           │     │     └─→ Set in CONFIG dict
  │           │     │
  │           │     └─→ Example: CF_DATA_NUM_SERVERS=150 overrides num_servers
  │           │
  │           ├─→ Validate configuration
  │           │     │
  │           │     ├─→ Check required keys exist
  │           │     ├─→ Validate data types
  │           │     ├─→ Validate ranges (e.g., num_servers > 0)
  │           │     └─→ Raise ValueError if invalid
  │           │
  │           └─→ Store in global CONFIG variable
  │
  └─→ Return CONFIG['data']
```

### 3.2 Code Implementation

**File:** `src/utils/config.py`

```python
# Global configuration storage
CONFIG = {}
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary containing all configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file has invalid YAML syntax
        ValueError: If configuration validation fails
    """
    global CONFIG

    # Find config file
    if config_path is None:
        default_path = PROJECT_ROOT / 'config' / 'config.yaml'
        env_path = os.environ.get('CAPACITY_FORECASTER_CONFIG')

        if env_path and Path(env_path).exists():
            config_path = env_path
        elif default_path.exists():
            config_path = default_path
        else:
            raise FileNotFoundError(
                f"Config file not found. Tried:\n"
                f"  - {default_path}\n"
                f"  - Environment variable: CAPACITY_FORECASTER_CONFIG"
            )

    # Load YAML
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # Apply environment variable overrides
    _apply_env_overrides()

    # Validate
    _validate_config()

    return CONFIG


def _apply_env_overrides():
    """
    Override config values from environment variables.

    Environment variables should be prefixed with 'CF_' and use underscores
    for nested keys. Examples:
        CF_DATA_NUM_SERVERS=150
        CF_ML_FORECAST_HORIZON_MONTHS=12
    """
    for key, value in os.environ.items():
        if not key.startswith('CF_'):
            continue

        # Parse path: CF_DATA_NUM_SERVERS → ['data', 'num_servers']
        path_parts = key[3:].lower().split('_')

        # Navigate to correct location in CONFIG
        current = CONFIG
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Parse value (try int → float → bool → str)
        final_key = path_parts[-1]
        try:
            # Try integer
            parsed_value = int(value)
        except ValueError:
            try:
                # Try float
                parsed_value = float(value)
            except ValueError:
                # Try boolean
                if value.lower() in ['true', 'yes', '1']:
                    parsed_value = True
                elif value.lower() in ['false', 'no', '0']:
                    parsed_value = False
                else:
                    # Keep as string
                    parsed_value = value

        current[final_key] = parsed_value


def _validate_config():
    """Validate configuration parameters."""
    errors = []

    # Validate data section
    if 'data' not in CONFIG:
        errors.append("Missing 'data' section")
    else:
        data = CONFIG['data']

        if 'num_servers' not in data:
            errors.append("Missing data.num_servers")
        elif not isinstance(data['num_servers'], int) or data['num_servers'] <= 0:
            errors.append(f"data.num_servers must be positive integer, got {data['num_servers']}")

        if 'start_date' not in data:
            errors.append("Missing data.start_date")
        else:
            try:
                pd.to_datetime(data['start_date'])
            except Exception as e:
                errors.append(f"Invalid data.start_date: {e}")

        if 'end_date' not in data:
            errors.append("Missing data.end_date")
        else:
            try:
                pd.to_datetime(data['end_date'])
            except Exception as e:
                errors.append(f"Invalid data.end_date: {e}")

    # Validate AWS section (if present)
    if 'aws' in CONFIG:
        aws = CONFIG['aws']
        if 'region' in aws:
            valid_regions = ['us-east-1', 'us-west-1', 'us-west-2', 'eu-west-1', 'eu-central-1', 'ap-southeast-1']
            if aws['region'] not in valid_regions:
                errors.append(f"aws.region '{aws['region']}' not in valid regions")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def get_data_config() -> Dict:
    """Get data configuration section."""
    if not CONFIG:
        load_config()
    return CONFIG.get('data', {})


def get_feature_engineering_config() -> Dict:
    """Get feature engineering configuration section."""
    if not CONFIG:
        load_config()
    return CONFIG.get('feature_engineering', {})
```

---

## 4. Parameter Validation

### 4.1 Validation Rules

| Parameter | Validation Rule | Error Message |
|-----------|----------------|---------------|
| `num_servers` | Must be int > 0 | "data.num_servers must be positive integer" |
| `start_date` | Must be valid date string | "Invalid data.start_date: ..." |
| `end_date` | Must be valid date string | "Invalid data.end_date: ..." |
| `end_date` | Must be > start_date | "end_date must be after start_date" |
| `years_of_data` | Must be int 1-5 | "years_of_data must be between 1 and 5" |
| `granularity` | Must be 'daily' or 'hourly' | "granularity must be 'daily' or 'hourly'" |

### 4.2 Validation Process Flow

```
_validate_config()
  │
  ├─→ Validate 'data' section exists
  │     └─→ IF missing: Add error "Missing 'data' section"
  │
  ├─→ Validate num_servers
  │     ├─→ Check exists
  │     ├─→ Check type is int
  │     ├─→ Check value > 0
  │     └─→ IF fail: Add error message
  │
  ├─→ Validate dates
  │     ├─→ Parse start_date with pd.to_datetime()
  │     ├─→ Parse end_date with pd.to_datetime()
  │     ├─→ Check end_date > start_date
  │     └─→ IF fail: Add error message
  │
  ├─→ Validate AWS section (if present)
  │     ├─→ Check region in valid_regions list
  │     ├─→ Check bucket_name format
  │     └─→ IF fail: Add error message
  │
  ├─→ IF any errors collected:
  │     └─→ Raise ValueError with all error messages
  │
  └─→ ELSE: Validation passed
```

---

## 5. Environment Overrides

### 5.1 Override Mechanism

**Environment Variable Naming Convention:**
```
CF_<SECTION>_<SUBSECTION>_<PARAMETER>
```

**Examples:**
```bash
# Override number of servers
export CF_DATA_NUM_SERVERS=150

# Override start date
export CF_DATA_START_DATE="2020-01-01"

# Override forecast horizon
export CF_ML_FORECAST_HORIZON_MONTHS=12

# Override AWS region
export CF_AWS_REGION="us-west-2"
```

### 5.2 Type Parsing Logic

```
Parse environment variable value:
  │
  ├─→ Try parse as INTEGER
  │     └─→ int(value)
  │           ├─→ Success: Use as int
  │           └─→ Fail: Continue
  │
  ├─→ Try parse as FLOAT
  │     └─→ float(value)
  │           ├─→ Success: Use as float
  │           └─→ Fail: Continue
  │
  ├─→ Try parse as BOOLEAN
  │     └─→ IF value.lower() in ['true', 'yes', '1']:
  │           └─→ Use True
  │         IF value.lower() in ['false', 'no', '0']:
  │           └─→ Use False
  │         ELSE: Continue
  │
  └─→ Use as STRING
```

### 5.3 Override Examples

**Scenario 1: Generate more servers**
```bash
# Set environment variable
export CF_DATA_NUM_SERVERS=200

# Run generation
python -m src.data_generation

# Result: Generates 200 servers instead of 120
```

**Scenario 2: Change date range**
```bash
export CF_DATA_START_DATE="2023-01-01"
export CF_DATA_END_DATE="2026-12-31"

python -m src.data_generation

# Result: Generates 4 years from 2023-2026
```

**Scenario 3: Disable seasonality**
```bash
export CF_DATA_SEASONALITY_QUARTERLY_PEAKS=false
export CF_DATA_SEASONALITY_HOLIDAY_EFFECT=false

python -m src.data_generation

# Result: No quarterly peaks or holiday effects
```

---

## 6. Configuration Best Practices

### 6.1 Development vs. Production

**Development (config.yaml):**
```yaml
data:
  num_servers: 10          # Small for fast testing
  years_of_data: 1         # 1 year for quick iteration
  start_date: "2025-01-01"
```

**Production (environment overrides):**
```bash
export CF_DATA_NUM_SERVERS=120
export CF_DATA_YEARS_OF_DATA=4
export CF_DATA_START_DATE="2022-01-01"
```

### 6.2 Parameter Selection Guidelines

| Use Case | num_servers | years_of_data | granularity |
|----------|-------------|---------------|-------------|
| **Quick Test** | 10 | 1 | daily |
| **Development** | 50 | 2 | daily |
| **Production** | 120 | 4 | daily |
| **Large Scale** | 200 | 5 | daily |
| **Hourly Data** | 50 | 1 | hourly |

**Memory Considerations:**
- Daily, 120 servers, 4 years: ~18 MB in memory, 3 MB compressed
- Hourly, 120 servers, 4 years: ~432 MB in memory, ~50 MB compressed
- Hourly, 200 servers, 5 years: ~1.2 GB in memory, ~150 MB compressed

---

## 7. Configuration File Template

```yaml
# config/config.yaml
# AWS-CapacityForecaster Configuration

project:
  name: AWS-CapacityForecaster
  version: 1.0
  description: Enterprise server capacity forecasting system

data:
  # Generation parameters
  num_servers: 120                    # 50-200 recommended
  years_of_data: 4                    # 1-5 years
  granularity: daily                  # 'daily' or 'hourly'
  start_date: "2022-01-01"
  end_date: "2025-12-31"

  # Seasonality settings
  seasonality:
    weekly: true                      # Business hours vs. off-hours
    quarterly_peaks: true             # End-of-quarter surges
    holiday_effect: true              # Holiday reductions

  # P95 ranges (future use)
  p95_ranges:
    cpu: [10.0, 95.0]
    memory: [15.0, 92.0]
    disk: [5.0, 85.0]
    network_in: [20.0, 500.0]
    network_out: [10.0, 300.0]

feature_engineering:
  lags: [1, 7, 30]
  rolling_windows: [7, 14, 30]
  external_regressors:
    - is_end_of_quarter
    - is_month_end

execution:
  mode: local
  parallel_workers: 4
  random_seed: 42
```

---

## Navigation

📄 [◀ Back to Master Guide](00_MASTER_TECHNICAL_GUIDE.md)
📄 [Next: Code Walkthrough ▶](02_Code_Walkthrough.md)

---

**Document End - Part 2 of 5**

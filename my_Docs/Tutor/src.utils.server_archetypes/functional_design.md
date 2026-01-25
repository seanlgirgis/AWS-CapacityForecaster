# Data Generation Flow: How the Synthetic Data File Gets Created

## Overview

This document explains how the AWS-CapacityForecaster generates synthetic server capacity data across 4 years with realistic patterns, focusing on **randomness**, **continuity**, and **which function creates what**.

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│  │ config.yaml  │───►│ generate_full_   │───►│ Final DataFrame        │    │
│  │ (parameters) │    │ dataset()        │    │ (CSV/Parquet)          │    │
│  └──────────────┘    └────────┬─────────┘    └────────────────────────┘    │
│                               │                                             │
│         ┌─────────────────────┼─────────────────────┐                       │
│         ▼                     ▼                     ▼                       │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐              │
│  │ assign_      │    │ ServerArchetype  │    │ add_calendar │              │
│  │ archetypes_  │    │ .generate_       │    │ _features()  │              │
│  │ to_fleet()   │    │ correlated_      │    │ data_utils   │              │
│  │              │    │ metrics()        │    │              │              │
│  └──────────────┘    └──────────────────┘    └──────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Configuration Loading

**Source:** `config/config.yaml` → loaded by `get_data_config()`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_servers` | 120 | Number of servers to simulate |
| `start_date` | 2022-01-01 | First day of data |
| `end_date` | 2025-12-31 | Last day of data (~4 years) |
| `granularity` | daily | One record per server per day |
| `quarterly_peaks` | true | Banking EOQ load spikes |
| `holiday_effect` | true | Reduced load on US holidays |

---

## Phase 2: Timestamp Generation

**Function:** `pd.date_range()` in `generate_full_dataset()` (Line 99-102)

```python
date_range = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')
```

**Result:** ~1,461 daily timestamps (4 years)

---

## Phase 3: Server Archetype Assignment

**Function:** `assign_archetypes_to_fleet()` (server_archetypes.py:392-451)

### Default Distribution

| Archetype | Proportion | For 120 Servers |
|-----------|------------|-----------------|
| Web | 35% | 42 servers |
| Application | 40% | 48 servers |
| Database | 15% | 18 servers |
| Batch | 10% | 12 servers |

### How It Works

```python
for archetype, proportion in distribution.items():
    count = int(num_servers * proportion)
    for _ in range(count):
        server_id = f"server_{server_idx:03d}"
        assignments[server_id] = archetype  # e.g., "server_042" → "database"
```

**Output:** Dictionary `{'server_000': 'web', 'server_001': 'web', ... 'server_119': 'batch'}`

---

## Phase 4: Archetype Profile Definitions

**Location:** `ARCHETYPE_PROFILES` dictionary (server_archetypes.py:98-214)

Each archetype has a unique `ArchetypeProfile` dataclass:

### Web Server Profile

```python
ServerType.WEB: ArchetypeProfile(
    name="Web Server",
    base_cpu=45.0,          # Mean CPU %
    base_memory=35.0,       # Mean Memory %
    base_disk=20.0,         # Mean Disk %
    base_network=150.0,     # Mean Network Mbps

    cpu_variance=15.0,      # Std deviation for randomness
    memory_variance=8.0,
    disk_variance=5.0,
    network_variance=50.0,

    # Correlations (how metrics move together)
    cpu_memory_correlation=0.5,
    cpu_network_correlation=0.8,  # Strong: more requests = more CPU & network
    memory_disk_correlation=0.2,

    # Time-based factors
    business_hours_factor=1.6,  # 60% higher during 9AM-5PM
    weekend_factor=0.5,         # 50% lower on weekends

    # Random spikes
    spike_probability=0.03,     # 3% chance per data point
    spike_magnitude=1.8,        # 80% increase during spike

    monthly_growth_rate=0.5,    # 0.5% growth per month
)
```

### All Four Archetypes Compared

| Metric | Web | Database | Application | Batch |
|--------|-----|----------|-------------|-------|
| Base CPU | 45% | 35% | 50% | 30% |
| Base Memory | 35% | 70% | 55% | 45% |
| Base Disk | 20% | 55% | 30% | 40% |
| CPU Variance | 15 | 12 | 18 | 25 |
| Business Hours Factor | 1.6x | 1.3x | 1.5x | 0.8x |
| Weekend Factor | 0.5x | 0.7x | 0.6x | 1.2x |
| Spike Probability | 3% | 1% | 2% | 8% |
| Monthly Growth | 0.5% | 1.0% | 0.8% | 0.3% |

---

## Phase 5: Randomness & Reproducibility

### Per-Server Seed (Deterministic Randomness)

**Function:** `ServerArchetype.__init__()` (Line 222-236)

```python
def __init__(self, archetype_type: ServerType, server_id: str):
    # Create deterministic but unique seed per server
    self.seed = hash(server_id) % (2**32)
    self.rng = np.random.RandomState(self.seed)
```

**Why This Matters:**
- `hash("server_000")` → consistent seed (e.g., 1234567890)
- Same `server_id` always produces same random sequence
- Different servers get different patterns
- Re-running generation produces identical data

### Correlated Random Numbers (Cholesky Decomposition)

**Function:** `generate_correlated_metrics()` (Line 242-323)

```python
# Correlation matrix (CPU, Memory, Disk, Network)
corr_matrix = np.array([
    [1.0, 0.5, 0.1, 0.8],  # CPU correlates strongly with Network
    [0.5, 1.0, 0.7, 0.2],  # Memory correlates strongly with Disk
    [0.1, 0.7, 1.0, 0.3],
    [0.8, 0.2, 0.3, 1.0]
])

# Cholesky decomposition for correlated variables
L = np.linalg.cholesky(corr_matrix)

# Generate 4 independent random numbers
z = self.rng.randn(4)  # [z1, z2, z3, z4]

# Transform to correlated variables
correlated = L @ z  # Matrix multiplication produces correlated values
```

**Result:** When CPU goes up, Network also tends to go up (0.8 correlation)

---

## Phase 6: Time Continuity Across 4 Years

### Linear Trend Factor

**Location:** `generate_full_dataset()` (Line 152)

```python
for idx, timestamp in enumerate(date_range):
    trend_factor = idx / len(date_range)  # 0.0 → 1.0 over 4 years
```

| Day | idx | trend_factor | Effect |
|-----|-----|--------------|--------|
| 2022-01-01 | 0 | 0.000 | Base values |
| 2023-07-01 | 730 | 0.500 | +50% of growth applied |
| 2025-12-31 | 1460 | 1.000 | Full growth applied |

### Growth Calculation

```python
cpu = (
    self.profile.base_cpu
    * time_factor
    * (1 + trend_factor * self.profile.monthly_growth_rate / 100)
    + correlated[0] * self.profile.cpu_variance
)
```

**Example (Database Server):**
- Day 1: `35.0 * 1.0 * (1 + 0.0 * 1.0/100) = 35.0%`
- Day 1461: `35.0 * 1.0 * (1 + 1.0 * 1.0/100) = 35.35%`

---

## Phase 7: Time-Based Adjustments

### Business Hours Factor

**Function:** `get_time_factor()` (Line 328-354)

```python
def get_time_factor(self, timestamp) -> float:
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek

    # Business hours (9 AM - 5 PM)
    if 9 <= hour <= 17:
        bh_factor = self.profile.business_hours_factor  # e.g., 1.6 for web
    else:
        bh_factor = 1.0

    # Weekend
    if day_of_week >= 5:
        weekend_factor = self.profile.weekend_factor  # e.g., 0.5 for web
    else:
        weekend_factor = 1.0

    return bh_factor * weekend_factor
```

### Quarterly Peak Factor (Banking-Specific)

**Function:** `_get_quarterly_peak_factor()` (data_generation.py:249-281)

```python
# Quarter end months: March, June, September, December
if month in [3, 6, 9, 12]:
    days_in_month = pd.Timestamp(timestamp.year, month, 1).days_in_month

    # Last 5 days of quarter
    if day >= days_in_month - 4:
        days_from_end = days_in_month - day
        peak_intensity = 1.0 + (0.3 * (5 - days_from_end) / 5)  # Up to 1.3x
        return peak_intensity
```

**Example (March 2024):**

| Date | Days from End | Factor |
|------|---------------|--------|
| Mar 27 | 4 | 1.06x |
| Mar 28 | 3 | 1.12x |
| Mar 29 | 2 | 1.18x |
| Mar 30 | 1 | 1.24x |
| Mar 31 | 0 | 1.30x |

### Holiday Factor

**Function:** `_get_holiday_factor()` (data_generation.py:284-323)

| Date | Factor | Effect |
|------|--------|--------|
| Jan 1 (New Year) | 0.5x | 50% reduced load |
| Jul 4 (Independence) | 0.7x | 30% reduced load |
| Nov 22-28 (Thanksgiving) | 0.7x | 30% reduced load |
| Dec 24 (Christmas Eve) | 0.7x | 30% reduced load |
| Dec 25-31 (Holiday Week) | 0.6x | 40% reduced load |

---

## Phase 8: Random Spike Injection

**Location:** `generate_correlated_metrics()` (Line 304-308)

```python
# Add spike if probability triggers
if self.rng.rand() < self.profile.spike_probability:  # e.g., 3% for web
    spike_mult = self.profile.spike_magnitude  # e.g., 1.8 for web
    cpu *= spike_mult
    memory *= (spike_mult * 0.7)  # Memory spikes less than CPU
    network_base *= (spike_mult * 0.8)
```

**Spike Frequencies:**
- Web: 3% = ~44 spikes per year per server
- Database: 1% = ~15 spikes per year per server
- Batch: 8% = ~117 spikes per year per server

---

## Phase 9: Value Clipping

**Location:** `generate_correlated_metrics()` (Line 311-315)

```python
cpu = np.clip(cpu, 0, 100)
memory = np.clip(memory, 0, 100)
disk = np.clip(disk, 0, 100)
network_in = np.clip(network_base, 0, 1000)
network_out = np.clip(network_base * 0.6, 0, 600)
```

Ensures all values stay within realistic bounds.

---

## Complete Function Call Chain

```
main()
  └── generate_full_dataset()
        ├── get_data_config()                    # Load config.yaml
        ├── pd.date_range()                      # Create timestamps
        ├── assign_archetypes_to_fleet()         # Assign server types
        │     └── (uses default distribution)
        │
        ├── FOR each server_id, archetype_type:
        │     ├── get_archetype()                # Create ServerArchetype
        │     │     └── ServerArchetype.__init__()
        │     │           └── hash(server_id)    # Deterministic seed
        │     │
        │     └── FOR each timestamp:
        │           ├── archetype.get_time_factor()      # Business hours/weekend
        │           ├── _get_quarterly_peak_factor()     # EOQ spikes
        │           ├── _get_holiday_factor()            # Holiday reduction
        │           │
        │           └── archetype.generate_correlated_metrics()
        │                 ├── Build correlation matrix
        │                 ├── Cholesky decomposition
        │                 ├── rng.randn(4)               # Random values
        │                 ├── Apply trend_factor         # Growth over time
        │                 ├── Apply time_factor          # Combined factors
        │                 ├── Maybe apply spike          # Random spike
        │                 └── np.clip()                  # Bound values
        │
        ├── pd.DataFrame(all_data)               # Build DataFrame
        ├── generate_server_metadata()           # Add business metadata
        └── add_calendar_features()              # Add calendar columns
```

---

## Which Function Creates What

| Function | File | Creates |
|----------|------|---------|
| `generate_full_dataset()` | data_generation.py | Orchestrates entire pipeline |
| `pd.date_range()` | data_generation.py:100 | Timestamp array (1,461 days) |
| `assign_archetypes_to_fleet()` | server_archetypes.py:392 | Server→Archetype mapping |
| `get_archetype()` | server_archetypes.py:357 | ServerArchetype instance |
| `ServerArchetype.__init__()` | server_archetypes.py:222 | Per-server random seed |
| `get_time_factor()` | server_archetypes.py:328 | Business hours/weekend multiplier |
| `_get_quarterly_peak_factor()` | data_generation.py:249 | EOQ peak multiplier (1.0-1.3) |
| `_get_holiday_factor()` | data_generation.py:284 | Holiday reduction (0.5-1.0) |
| `generate_correlated_metrics()` | server_archetypes.py:242 | CPU, memory, disk, network values |
| `generate_server_metadata()` | data_utils.py:147 | business_unit, criticality, region |
| `add_calendar_features()` | data_utils.py:319 | year, month, quarter, is_weekend, is_holiday |
| `save_dataset()` | data_generation.py:326 | CSV or Parquet file output |

---

## Output Data Structure

| Column | Type | Source Function |
|--------|------|-----------------|
| timestamp | datetime | pd.date_range() |
| server_id | string | assign_archetypes_to_fleet() |
| cpu_p95 | float | generate_correlated_metrics() |
| mem_p95 | float | generate_correlated_metrics() |
| disk_p95 | float | generate_correlated_metrics() |
| net_in_p95 | float | generate_correlated_metrics() |
| net_out_p95 | float | generate_correlated_metrics() |
| server_type | string | archetype_assignments mapping |
| business_unit | string | generate_server_metadata() |
| criticality | string | generate_server_metadata() |
| region | string | generate_server_metadata() |
| year | int | add_calendar_features() |
| month | int | add_calendar_features() |
| quarter | int | add_calendar_features() |
| dayofweek | int | add_calendar_features() |
| is_weekend | int | add_calendar_features() |
| is_eoq | int | add_calendar_features() |
| is_holiday | int | add_calendar_features() |

---

## Summary: How Randomness + Continuity Work Together

1. **Reproducibility:** `hash(server_id)` creates deterministic seed per server
2. **Correlation:** Cholesky decomposition makes metrics move together realistically
3. **Continuity:** `trend_factor = idx / total_days` creates smooth growth
4. **Seasonality:** Time-based multipliers (business hours, weekends, EOQ, holidays)
5. **Noise:** `rng.randn()` adds realistic variance around base values
6. **Spikes:** Probability-based random events with magnitude multipliers
7. **Bounds:** `np.clip()` ensures values stay realistic

**Total Records:** 120 servers × 1,461 days = **175,320 records**

The plan for **`module_01_data_generation.py`** is to create a clean, standalone, configuration-driven module that acts as the **reliable starting point** of the entire modular pipeline.

Its sole responsibility: **Generate high-fidelity synthetic enterprise server metrics** that closely mimic Citi-style performance telemetry (daily P95 CPU, memory, disk, network utilization + metadata), incorporating realistic patterns: weekly patterns, annual seasonality, end-of-quarter (EOQ) spikes, US banking holidays, business unit/region/criticality variation.

**In / Process / Out structure** — kept deliberately tight and explicit so the output can be directly consumed by `module_02_data_load.py` (and later modules) without rework.

### High-Level Flow (In → Process → Out)

**In** (Inputs – all controlled via config)
- Configuration object (loaded from `config/config.yaml` + optional env overrides)
- Key config sections expected/used (based on your existing config.yaml style):
  - `data_generation`:
    - `num_servers`: int (e.g. 120)
    - `start_date`, `end_date`: YYYY-MM-DD (e.g. 2022-01-01 to 2025-12-31 → ~4 years daily)
    - `metrics`: list of dicts or subkeys for each metric (cpu_p95, mem_p95, disk_p95, net_p95) with:
      - `base_mean`, `base_std`, `min_clip`, `max_clip`
      - `seasonality_amplitude`, `weekly_pattern_factor`, `eoq_spike_multiplier`, `eoq_months` (e.g. [3,6,9,12])
      - `holiday_effect_multiplier` (if `use_holidays: true`)
    - `use_holidays`: bool
    - `use_eoq_spikes`: bool
    - `metadata`: settings for server attributes (business_units list, regions list, criticality_levels list, assignment probabilities)
    - `random_seed`: int (for reproducibility)
  - `storage`:
    - `bucket_name`
    - `raw_prefix`: e.g. "raw/"
    - `use_s3`: bool
    - `local_base_path`: fallback directory
  - `general`: `env` ("local" | "sagemaker" | "lambda")

**Process** (Core logic – deterministic + configurable)
1. **Load & validate config** → ensure required keys exist, value ranges sensible (e.g. dates valid, num_servers > 0)
2. **Generate server metadata** (one row per server):
   - server_id (e.g. "SRV-{0001..0120}")
   - business_unit, region, criticality (random choice with configurable weights)
3. **Create date range** → pd.date_range(start, end, freq="D")
4. **Cross join** → create full panel (server × date) using pandas
5. **Generate base utilization** for each metric (vectorized/numpy):
   - Base noise: normal distribution (mean + std per metric)
   - Add weekly pattern (e.g. lower weekends)
   - Add annual seasonality (sin/cos waves or Fourier terms)
   - Add holiday effects (using `holidays.US()` library if enabled)
   - Add EOQ spikes (multiply by factor on last 5–10 business days of Mar/Jun/Sep/Dec)
   - Clip to realistic min/max
6. **Add light realism**:
   - Small random server-specific offset (per-server bias)
   - Occasional missing values (configurable %)
   - Outliers/anomalies (rare spikes/dips, configurable)
7. **Quality & success checks** (before save):
   - Assert df shape ≈ (num_servers × days, expected columns)
   - Check missing % < threshold (e.g. 5%)
   - Verify P95 values in realistic range (e.g. 0–100 for % metrics)
   - Log summary stats (mean, std, min/max per metric)
   - Optional: small assertion on known patterns (e.g. higher utilization mid-week)

**Out** (Outputs – consistent & ready for next module)
- Primary artifact: **Parquet file** (efficient, preserves types, columnar)
  - Filename pattern: `raw_server_metrics_{start_date}_to_{end_date}.parquet`
  - Columns (minimum set – expandable):
    - `server_id`, `date`, `business_unit`, `region`, `criticality`
    - `cpu_p95`, `mem_p95`, `disk_p95`, `net_p95`
    - Optional: `is_weekend`, `is_holiday`, `is_eoq_window` (added here or in ETL?)
- Storage locations (controlled by config):
  - **Preferred**: S3 → `s3://{bucket}/{raw_prefix}/{filename}`
  - **Fallback**: Local → `C:\pyproj\AWS-CapacityForecaster\data\raw\{filename}`
- Side outputs (optional but recommended for debug):
  - JSON summary: row count, date range, server count, basic stats per metric
  - Small sample CSV (first 100 rows) for quick inspection

### Execution & Testing Pattern
- Standalone runnable:
  ```powershell
  cd C:\pyproj\AWS-CapacityForecaster
  python src\modules\module_01_data_generation.py --config config/config.yaml --env local
  ```
- In `__main__` block:
  - Parse args → load/validate config
  - Call `generate_synthetic_data(config)`
  - Save output
  - Run checks → log "SUCCESS" or raise/log error

### Integration Notes for Chainability
- Output format (parquet + known column names) must match exactly what `module_02_data_load.py` expects to read.
- Use same `save_to_s3_or_local()` helper from `utils/data_utils.py` (or create one if missing) → ensures consistent path logic across all modules.
- Keep metric names, dtypes, and folder structure identical to what future modules assume (prevents downstream breakage).
- Config keys should be prefixed under `data_generation:` so they don't clash with later modules (e.g. `data_generation.num_servers` vs `model_training.forecast_horizon`).

### Next Immediate Step
1. Create the file at  
   `C:\pyproj\AWS-CapacityForecaster\src\modules\module_01_data_generation.py`
2. Use your existing synthetic generation logic (likely in notebooks/01_data_generation.ipynb or utils/data_utils.py) → extract & refactor into this module.
3. Add the config-driven structure + checks as outlined.

Would you like me to provide a **detailed code skeleton** for this file now (with placeholders for your generation logic), or do you want to draft it first and then review/debug together? Just say "give me the skeleton" or "let's refine after I code the core generation part".
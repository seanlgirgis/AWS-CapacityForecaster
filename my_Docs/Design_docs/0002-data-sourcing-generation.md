# 02 – Data Sourcing & Generation Strategy

**Version**: 0.1 – Draft  
**Status**: In progress

## 1. Goals for Data
- Realistic enough to demonstrate Citi-like capacity planning (seasonality, peaks, P95 risks)
- Reproducible and fast to generate (portfolio-friendly)
- Suitable for time-series ML (clean timestamps, multiple features)
- Low friction (no huge downloads, no complex ETL upfront)

## 2. Decision: Primary → Synthetic Generation
- Why: Full control over patterns, seasonality (banking calendar), scale, and cleanliness
- How: pandas/numpy/scipy for time-series + custom functions for:
  - Base trend + weekly cycle
  - End-of-month/quarter spikes
  - Random server groups (high/low utilization)
  - Noise/outliers/missing values

## 3. Validation with Real-World Traces (Optional)
- Alibaba Cluster Trace v2018 (machine_usage table) – subset only
- TSBS benchmark data generator
- Purpose: Tune synthetic params, show comparison in notebooks

## 4. Target Data Specs
- 50–200 servers
- 3–5 years daily data (P95 CPU/mem/disk/network)
- Columns: timestamp, server_id, cpu_p95, mem_p95, disk_p95, ...
- ~100k–500k rows total (light for AWS + local)

## 5. Future Options
- Add hourly data later
- Incorporate Alibaba/Google patterns for advanced analysis

Next steps: Finalize synthetic data schema in next doc/notebook
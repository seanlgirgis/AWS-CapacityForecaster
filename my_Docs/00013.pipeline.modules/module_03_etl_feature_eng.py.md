```markdown
# Module Design Document: module_03_etl_feature_eng.py

**File Path**  
`src/modules/module_03_etl_feature_eng.py`

**Module Sequence Number**  
03 (immediately follows module_02_data_load.py → feeds module_04_model_training.py)

**Last Updated**  
January 2026

**Author / Maintainer**  
seanlgirgis (with Grok assistance)

## 1. Purpose & Business Context

This is the **heart of the data science value** in the AWS-CapacityForecaster pipeline.

It takes the clean, validated panel data from module_02 and transforms it into a **rich, ML-ready feature set** that captures every nuance of real Citi-style capacity telemetry:

- Short-term autocorrelation (lags)
- Medium-term trends & volatility (rolling windows)
- Calendar & business-cycle effects (EOQ spikes, holidays, weekends)
- Server heterogeneity (metadata encoding)
- Long-term drift (cumulative & time-index features)

These features are exactly what enabled 20–30% forecast accuracy gains in the original Citi Prophet + scikit-learn ensemble models.

**Citi Realism Alignment**  
In the real Citi environment:
- Raw P95 metrics had gaps/outliers → needed robust imputation
- Strong weekly + annual seasonality + EOQ spikes were critical regressors
- Server metadata (business unit, region, criticality) explained variance
- Lags & rolling stats were the #1 feature importance in RandomForest/GradientBoosting models

This module recreates that exact feature engineering playbook in a clean, configurable, reproducible way.

**Pipeline Hand-off Guarantee**  
- Input: validated panel (120 servers × ~1461 days daily)
- Output: processed feature matrix with ~80–120 columns
- Next module (04_model_training) can directly use this DataFrame for train/test split & modeling — no further cleaning required

## 2. In / Process / Out Structure

### In (Inputs – Strictly Config-Driven)

- **Validated DataFrame** from module_02  
  Location: `staged/validated_server_metrics_YYYYMMDD_to_YYYYMMDD.parquet` (local or S3)  
  Schema: exactly as module_02 output (10 raw columns + sorted + datetime timestamp)

- **Configuration sections** (config.yaml)
  ```yaml
  features:
    lags_days: [1, 3, 7, 14, 30]          # lag features per metric
    rolling_windows_days: [7, 30, 90]     # windows for mean/std/min/max
    impute_method: "linear"               # or "knn", "median", "forward_fill"
    outlier_z_threshold: 4.0              # flag or cap values > Z from rolling mean
    encode_metadata: true                 # one-hot business_unit/region/criticality
    add_eoq_flag: true                    # binary is_eoq_window
    add_holiday_flag: true                # binary is_holiday (US banking)
    add_trend_features: true              # days_since_start, cumulative_cpu etc.
  storage:
    processed_prefix: "processed/"        # where to save final feature parquet
  ```

### Process (Core Logic – Detailed Pseudo-Code)

```text
1. Load validated DataFrame using load_from_s3_or_local (prefix="staged/", latest or specific file)

2. Minor cleaning (if not already perfect from module_02)
   - Sort again by ['server_id', 'timestamp'] (defensive)
   - Set index: MultiIndex(server_id, timestamp) for groupby efficiency

3. Imputation of missing numeric values
   if config.features.impute_method == "linear":
       df[numeric_cols] = df.groupby('server_id')[numeric_cols].transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
   elif "knn":
       from sklearn.impute import KNNImputer → per-server KNN (n_neighbors=5)
   fallback: forward_fill + backward_fill

4. Outlier handling (optional cap or flag)
   for each metric:
       rolling_mean = group.rolling(30, min_periods=1).mean()
       rolling_std  = group.rolling(30, min_periods=1).std()
       z = (value - rolling_mean) / rolling_std
       if abs(z) > config.features.outlier_z_threshold:
           value = rolling_mean + sign(z) * threshold * rolling_std   # Winsorize

5. Lag features
   for metric in numeric_metrics:
       for lag in config.features.lags_days:
           df[f"{metric}_lag_{lag}d"] = df.groupby('server_id')[metric].shift(lag)

6. Rolling statistics
   for metric in numeric_metrics:
       for window in config.features.rolling_windows_days:
           roll = df.groupby('server_id')[metric].transform(lambda x: x.rolling(window, min_periods=1))
           df[f"{metric}_roll_mean_{window}d"] = roll.mean()
           df[f"{metric}_roll_std_{window}d"]  = roll.std()
           df[f"{metric}_roll_min_{window}d"] = roll.min()
           df[f"{metric}_roll_max_{window}d"] = roll.max()

7. Calendar & business cycle flags
   df['day_of_week']   = df['timestamp'].dt.dayofweek           # 0=Mon
   df['month']         = df['timestamp'].dt.month
   df['is_weekend']    = df['day_of_week'].isin([5,6])
   df['is_eoq_window'] = df['timestamp'].dt.day >= (df['timestamp'].dt.daysinmonth - 6) & df['timestamp'].dt.month.isin([3,6,9,12])
   if config.features.add_holiday_flag:
       us_holidays = holidays.US(years=range(start_year, end_year+1))
       df['is_holiday'] = df['timestamp'].dt.date.isin(us_holidays)

8. Server metadata encoding
   if config.features.encode_metadata:
       df = pd.get_dummies(df, columns=['business_unit', 'region', 'criticality'], prefix=['bu', 'reg', 'crit'])

9. Trend & cumulative features
   if config.features.add_trend_features:
       df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
       for metric in numeric_metrics:
           df[f"{metric}_cumulative"] = df.groupby('server_id')[metric].cumsum()

10. Final cleanup
    - Drop any rows with NaN in lag/rolling features from the beginning (first N rows per server)
    - Reset index, sort again
    - Assert no infinite values

11. Save processed Parquet
    filename = f"processed_features_{min_date}_to_{max_date}.parquet"
    save_to_s3_or_local(df, config, prefix="processed/", filename=filename)

12. Generate & save JSON summary
    - rows, columns, feature_counts per group, missing % after engineering, sample head
```

### Out (Outputs – Predictable & Rich)

- **Primary**: Processed feature Parquet  
  Location: `processed/processed_features_YYYYMMDD_to_YYYYMMDD.parquet`  
  Columns: ~80–120 (original 10 + lags + rolling + calendar + one-hot + trend)

- **Side artifacts**:
  - JSON summary: `reports/summaries/module_03_summary.json`
    Example keys: total_features, lag_features_count, rolling_windows, missing_after_etl, sample_head (5 rows)
  - Optional: feature importance preview (correlation with cpu_p95 or quick RandomForest on small sample)

## 3. How to Run Manual Checks (Post-Execution Verification)

```powershell
# 1. Quick inspection
python -c "
import pandas as pd
df = pd.read_parquet('data/scratch/processed/processed_features_20220101_to_20251231.parquet')
print('Shape:', df.shape)
print('Columns sample:', df.columns[:20].tolist())
print('Lag example:\n', df[['cpu_p95', 'cpu_p95_lag_7d']].head(10))
print('Missing after ETL:', df.isna().mean().mean() * 100, '%')
"

# 2. Verify EOQ flag works
python -c "
import pandas as pd
df = pd.read_parquet('data/scratch/processed/processed_features_20220101_to_20251231.parquet')
eoq_sample = df[df['is_eoq_window']][['timestamp', 'cpu_p95']].head(10)
print('EOQ periods sample:\n', eoq_sample)
"

# 3. Check JSON summary
type data\scratch\reports\summaries\module_03_summary.json
```

**Expected Success Signs**
- Shape: (175320 - some head rows dropped, ~100 columns)
- Lag columns present and shifted correctly
- Rolling stats non-null after first window
- is_eoq_window true only in Mar/Jun/Sep/Dec last week
- Missing % near 0% after imputation

## 4. Next Module Hand-off

**module_04_model_training.py** receives a **perfectly clean, feature-rich panel** ready for:
- Time-based train/test split (e.g., last 180 days as test)
- Prophet + scikit-learn ensemble training
- Cross-validation & metric comparison

---
**End of Design Document**
```

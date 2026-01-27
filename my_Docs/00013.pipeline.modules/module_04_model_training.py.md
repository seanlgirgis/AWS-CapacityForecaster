# =============================================================================
# src/modules/module_04_model_training.py
#
# PURPOSE:
#   FOURTH module in the AWS-CapacityForecaster pipeline.
#   Trains forecasting models on processed features from module_03 and
#   generates both backtest predictions (test period) and true forward forecasts.
#
#   Recreates Citi-style enterprise capacity forecasting:
#   - Chronological train/test split
#   - Day-of-week average baseline (stronger than naive repeat)
#   - Prophet with external regressors + uncertainty intervals
#   - Random Forest with recursive multi-step forecasting (calendar features only)
#   - Saves fitted models, enriched forecasts, metrics
#
# ROLE IN PIPELINE:
#   module_03 → THIS → models/*.pkl + forecasts/all_model_forecasts.parquet + metrics/
#
# OUTPUT GUARANTEES:
#   - Fitted models                → data/scratch/models/
#   - Enriched forecasts           → data/scratch/forecasts/all_model_forecasts.parquet
#     (columns: server_id, model, timestamp, actual, predicted, yhat_lower, yhat_upper, is_future)
#   - Metrics summary              → data/scratch/metrics/model_comparison.json
#   - Detailed logging             → logs/module_04_model_training.log (rotated)
#
# USAGE:
#   python -m src.modules.module_04_model_training --env local
#
# =============================================================================

import logging
import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

from src.utils.config import load_config, validate_config
from src.utils.data_utils import load_from_s3_or_local, save_to_s3_or_local, save_processed_data

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(console_handler)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "module_04_model_training.log"

file_handler = TimedRotatingFileHandler(
    filename=log_file,
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(file_handler)

logger.info("Logging initialized — console + file (logs/module_04_model_training.log)")


# =============================================================================
# Helpers
# =============================================================================
def generate_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Generate calendar regressors for any date range (used for future)."""
    df = pd.DataFrame({'timestamp': pd.to_datetime(dates)})
    df['ds'] = df['timestamp']
    df['eoq_flag']     = df['ds'].dt.is_quarter_end.astype(int)
    df['is_holiday']   = 0   # ← extend with real holidays if desired
    df['is_weekend']   = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['quarter']      = df['ds'].dt.quarter
    df['month']        = df['ds'].dt.month
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    return df.drop(columns=['ds'])


def calculate_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE with safe division."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# =============================================================================
# Models
# =============================================================================
def train_predict_naive_seasonal(train_df, test_df, horizon_days, target_col, period=7):
    """Day-of-week average baseline — separate test + future predictions."""
    train_df = train_df.copy()
    train_df['dow'] = train_df['timestamp'].dt.dayofweek
    dow_means = train_df.groupby('dow')[target_col].mean().to_dict()

    # Test period: use actual test dates' day of week
    test_dows = test_df['timestamp'].dt.dayofweek
    test_preds = np.array([dow_means.get(dow, np.nan) for dow in test_dows])

    # Future period
    last_date = train_df['timestamp'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    future_dows = future_dates.dayofweek
    future_preds = np.array([dow_means.get(dow, np.nan) for dow in future_dows])

    return test_preds, future_preds, future_dates


def train_predict_prophet(train_df, test_df, future_dates_df, target_col, config):
    """Prophet with regressors + uncertainty — returns test preds + future preds+intervals."""
    p_cfg = config.get('model_training', {}).get('prophet', {})
    regressors = p_cfg.get('regressor_columns', [])

    # Training data
    pf_train = train_df[['timestamp', target_col] + regressors].rename(columns={'timestamp': 'ds', target_col: 'y'})

    m = Prophet(
        growth=p_cfg.get('growth', 'linear'),
        yearly_seasonality=p_cfg.get('yearly_seasonality', True),
        weekly_seasonality=p_cfg.get('weekly_seasonality', True),
        daily_seasonality=p_cfg.get('daily_seasonality', False),
        changepoint_prior_scale=p_cfg.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=p_cfg.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=p_cfg.get('holidays_prior_scale', 10.0)
    )

    for reg in regressors:
        if reg in pf_train.columns:
            m.add_regressor(reg)

    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    m.fit(pf_train)

    # All dates to predict (test + future)
    all_ds = pd.concat([
        pf_train[['ds']],
        test_df[['timestamp']].rename(columns={'timestamp': 'ds'}),
        future_dates_df[['timestamp']].rename(columns={'timestamp': 'ds'})
    ]).drop_duplicates().sort_values('ds')

    future = all_ds.copy()
    future_reg = generate_calendar_features(future['ds'])
    for reg in regressors:
        if reg in future_reg.columns:
            future[reg] = future_reg[reg].values

    forecast = m.predict(future)

    # Extract test period
    test_mask = forecast['ds'].isin(test_df['timestamp'])
    test_preds = forecast.loc[test_mask, 'yhat'].values

    # Extract future period
    future_mask = forecast['ds'].isin(future_dates_df['timestamp'])
    future_df = forecast.loc[future_mask, ['yhat', 'yhat_lower', 'yhat_upper']]
    future_preds = future_df.values  # (horizon_days, 3)

    return test_preds, future_preds, m, forecast


def train_predict_rf(train_df, test_df, future_dates_df, target_col, config):
    """Random Forest — recursive forecasting using only exogenous/calendar features."""
    rf_cfg = config.get('model_training', {}).get('random_forest', {})

    safe_features = [
        'eoq_flag', 'is_holiday', 'is_weekend', 'quarter', 'month',
        'month_sin', 'month_cos'
        # add more non-lagged features if available
    ]
    feature_cols = [c for c in safe_features if c in train_df.columns]

    X_train = train_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y_train = train_df[target_col]

    model = RandomForestRegressor(
        n_estimators=rf_cfg.get('n_estimators', 150),
        max_depth=rf_cfg.get('max_depth', 12),
        min_samples_split=rf_cfg.get('min_samples_split', 5),
        random_state=rf_cfg.get('random_state', 42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Test period prediction
    X_test = test_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    test_preds = model.predict(X_test)

    # Recursive future forecasting
    future_preds = []
    current_features = future_dates_df[feature_cols].copy().fillna(0)

    for i in range(len(future_dates_df)):
        pred = model.predict(current_features.iloc[[i]])[0]
        future_preds.append(pred)

    return test_preds, np.array(future_preds), model


# =============================================================================
# Per-server processing
# =============================================================================
def process_server_group(server_id, df_group, config):
    mt_cfg = config.get('model_training', {})
    target_col = mt_cfg.get('target_metric', 'cpu_p95')
    horizon_days = mt_cfg.get('horizon_days', 90)
    split_date = pd.to_datetime(mt_cfg.get('test_split_date', '2024-01-01'))

    df_group = df_group.sort_values('timestamp').copy()
    train_df = df_group[df_group['timestamp'] < split_date].copy()
    test_df  = df_group[df_group['timestamp'] >= split_date].copy()

    if len(train_df) < mt_cfg.get('min_history_days', 365):
        logger.warning(f"Server {server_id}: Insufficient history ({len(train_df)} days). Skipping.")
        return None, None

    if len(test_df) == 0:
        logger.warning(f"Server {server_id}: No test data after {split_date}.")
        return None, None

    # Future dates & regressors
    last_date = df_group['timestamp'].max()
    future_dates_df = pd.DataFrame({
        'timestamp': pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    })
    future_reg = generate_calendar_features(future_dates_df['timestamp'])
    future_dates_df = pd.concat([future_dates_df, future_reg], axis=1)

    results = []
    forecasts_parts = []

    models_dir = Path(config.get('paths', {}).get('local_data_dir', 'data/scratch')) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────
    # 1. Baseline
    # ────────────────────────────────────────────────
    if 'baseline_naive_seasonal' in mt_cfg.get('enabled_models', []):
        try:
            test_preds, future_preds, _ = train_predict_naive_seasonal(
                train_df, test_df, horizon_days, target_col
            )
            metrics = calculate_metrics(test_df[target_col].values, test_preds)

            results.append({
                'server_id': server_id,
                'model': 'baseline',
                'metrics': metrics
            })

            test_rows = pd.DataFrame({
                'server_id': [server_id] * len(test_df),
                'model':     ['baseline'] * len(test_df),
                'timestamp': test_df['timestamp'].values,
                'actual':    test_df[target_col].values,
                'predicted': test_preds,
                'yhat_lower': [np.nan] * len(test_df),
                'yhat_upper': [np.nan] * len(test_df),
                'is_future':  [False] * len(test_df)
            })

            future_rows = pd.DataFrame({
                'server_id': [server_id] * len(future_dates_df),
                'model':     ['baseline'] * len(future_dates_df),
                'timestamp': future_dates_df['timestamp'].values,
                'actual':    [np.nan] * len(future_dates_df),
                'predicted': future_preds,
                'yhat_lower': [np.nan] * len(future_dates_df),
                'yhat_upper': [np.nan] * len(future_dates_df),
                'is_future':  [True] * len(future_dates_df)
            })

            forecasts_parts.append(pd.concat([test_rows, future_rows], ignore_index=True))

        except Exception as e:
            logger.error(f"Baseline failed for {server_id}: {e}")

    # ────────────────────────────────────────────────
    # 2. Prophet
    # ────────────────────────────────────────────────
    if 'prophet' in mt_cfg.get('enabled_models', []):
        try:
            test_preds, future_preds, model, _ = train_predict_prophet(
                train_df, test_df, future_dates_df, target_col, config
            )
            metrics = calculate_metrics(test_df[target_col].values, test_preds)

            results.append({
                'server_id': server_id,
                'model': 'prophet',
                'metrics': metrics
            })

            # Save model
            with open(models_dir / f"prophet_{server_id}.pkl", 'wb') as f:
                pickle.dump(model, f)

            test_rows = pd.DataFrame({
                'server_id': [server_id] * len(test_df),
                'model':     ['prophet'] * len(test_df),
                'timestamp': test_df['timestamp'].values,
                'actual':    test_df[target_col].values,
                'predicted': test_preds,
                'yhat_lower': [np.nan] * len(test_df),
                'yhat_upper': [np.nan] * len(test_df),
                'is_future':  [False] * len(test_df)
            })

            future_rows = pd.DataFrame({
                'server_id': [server_id] * len(future_dates_df),
                'model':     ['prophet'] * len(future_dates_df),
                'timestamp': future_dates_df['timestamp'].values,
                'actual':    [np.nan] * len(future_dates_df),
                'predicted': future_preds[:, 0],
                'yhat_lower': future_preds[:, 1],
                'yhat_upper': future_preds[:, 2],
                'is_future':  [True] * len(future_dates_df)
            })

            forecasts_parts.append(pd.concat([test_rows, future_rows], ignore_index=True))

        except Exception as e:
            logger.error(f"Prophet failed for {server_id}: {e}")

    # ────────────────────────────────────────────────
    # 3. Random Forest
    # ────────────────────────────────────────────────
    if 'random_forest' in mt_cfg.get('enabled_models', []):
        try:
            test_preds, future_preds, model = train_predict_rf(
                train_df, test_df, future_dates_df, target_col, config
            )
            metrics = calculate_metrics(test_df[target_col].values, test_preds)

            results.append({
                'server_id': server_id,
                'model': 'random_forest',
                'metrics': metrics
            })

            # Save model
            with open(models_dir / f"rf_{server_id}.pkl", 'wb') as f:
                pickle.dump(model, f)

            test_rows = pd.DataFrame({
                'server_id': [server_id] * len(test_df),
                'model':     ['random_forest'] * len(test_df),
                'timestamp': test_df['timestamp'].values,
                'actual':    test_df[target_col].values,
                'predicted': test_preds,
                'yhat_lower': [np.nan] * len(test_df),
                'yhat_upper': [np.nan] * len(test_df),
                'is_future':  [False] * len(test_df)
            })

            future_rows = pd.DataFrame({
                'server_id': [server_id] * len(future_dates_df),
                'model':     ['random_forest'] * len(future_dates_df),
                'timestamp': future_dates_df['timestamp'].values,
                'actual':    [np.nan] * len(future_dates_df),
                'predicted': future_preds,
                'yhat_lower': [np.nan] * len(future_dates_df),
                'yhat_upper': [np.nan] * len(future_dates_df),
                'is_future':  [True] * len(future_dates_df)
            })

            forecasts_parts.append(pd.concat([test_rows, future_rows], ignore_index=True))

        except Exception as e:
            logger.error(f"RF failed for {server_id}: {e}")

    if forecasts_parts:
        server_forecasts = pd.concat(forecasts_parts, ignore_index=True)
    else:
        server_forecasts = None

    return results, server_forecasts


# =============================================================================
# Main
# =============================================================================
def main_process(config):
    logger.info("=== MODULE 04 : Model Training & Forward Forecasting ===")

    # Load latest processed file
    processed_dir = Path(config.get('paths', {}).get('local_data_dir', 'data/scratch')) / "processed"
    parquet_files = list(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No processed parquet files found in data/scratch/processed/")

    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading latest feature data: {latest_file.name}")

    df = load_from_s3_or_local(config, prefix="processed/", filename=latest_file.name)
    if df is None:
        raise FileNotFoundError(f"Failed to load {latest_file.name}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['server_id', 'timestamp'])

    # Output directories
    local_base = Path(config.get('paths', {}).get('local_data_dir', 'data/scratch'))
    for sub in ["models", "forecasts", "metrics"]:
        (local_base / sub).mkdir(parents=True, exist_ok=True)

    # Optional debug limit
    servers = df['server_id'].unique()
    debug_max = config.get('model_training', {}).get('debug_max_servers')
    if debug_max:
        servers = servers[:int(debug_max)]
        logger.info(f"DEBUG MODE: processing only first {debug_max} servers")

    logger.info(f"Processing {len(servers)} servers...")

    all_metrics = []
    all_forecasts = []

    for sid in servers:
        results, fc_df = process_server_group(sid, df[df['server_id'] == sid].copy(), config)
        if results:
            for r in results:
                m = r['metrics'].copy()
                m['server_id'] = sid
                m['model'] = r['model']
                all_metrics.append(m)
        if fc_df is not None:
            all_forecasts.append(fc_df)

    if not all_metrics:
        logger.error("No successful model runs.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    summary = metrics_df.groupby('model')['MAPE'].describe()
    logger.info("Model Performance Summary (MAPE):\n" + summary.to_string())

    avg_mape = metrics_df.groupby('model')['MAPE'].mean().sort_values()
    best_model = avg_mape.idxmin()
    logger.info(f"Best model by average MAPE: {best_model} ({avg_mape[best_model]:.2f}%)")

    # Save forecasts
    if all_forecasts:
        forecasts_combined = pd.concat(all_forecasts, ignore_index=True)
        save_path = save_processed_data(
            forecasts_combined, config,
            prefix="forecasts/",
            filename="all_model_forecasts.parquet"
        )
        logger.info(f"Forecasts saved → {save_path}")

    # Save metrics
    metrics_summary = {
        "processed_at": datetime.now().isoformat(),
        "total_servers_processed": len(servers),
        "models_evaluated": avg_mape.to_dict(),
        "best_model": best_model,
        "details": metrics_df.to_dict(orient='records')
    }
    save_to_s3_or_local(
        json.dumps(metrics_summary, indent=2),
        config,
        prefix="metrics/",
        filename="model_comparison.json"
    )

    logger.info("✔ Module 04 completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 04 — Model Training & Forecasting")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.env:
        config.setdefault('execution', {})['mode'] = args.env
    validate_config(config)

    main_process(config)
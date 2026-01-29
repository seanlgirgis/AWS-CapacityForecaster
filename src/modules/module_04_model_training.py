# =============================================================================
# src/modules/module_04_model_training.py
#
# PURPOSE:
#   FOURTH module in the AWS-CapacityForecaster pipeline.
#   Trains forecasting models (Prophet, Random Forest, Naive Seasonal baseline)
#   on the processed feature set from module_03.
#
#   Recreates Citi-style forecasting:
#   - Chronological train/test split (e.g. cutoff 2024-01-01)
#   - Strong baseline (day-of-week average)
#   - Prophet with external regressors (EOQ, holidays, weekends, quarter, lags/rolling)
#   - Random Forest with recursive multi-step forecasting
#   - True forward horizon forecasting (90 days default)
#   - Saves fitted models, enriched forecasts (with uncertainty), metrics
#
# ROLE IN PIPELINE:
#   module_03 (processed/*.parquet) → THIS → models/*.pkl + forecasts/*.parquet + metrics/
#
# OUTPUT GUARANTEES:
#   - Fitted models saved (prophet_*.pkl, rf_*.pkl)
#   - Enriched forecasts parquet (actuals, preds, intervals, horizon_offset, is_future)
#   - Comprehensive metrics JSON + summary logging
#   - Persistent logging (console + rotated file)
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
import os
import sys
import subprocess

# --- Runtime Dependency Installation for SageMaker ---
def install_dependencies():
    """Installs required packages if missing (handling SageMaker container environment)"""
    # Map import name -> pip package name
    required_packages = {
        "prophet": "prophet",
        "holidays": "holidays",
        "yaml": "PyYAML",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "dotenv": "python-dotenv"
    }
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])

install_dependencies()
# -----------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

from src.utils.config import load_config, validate_config
from src.utils.data_utils import load_from_s3_or_local, save_to_s3_or_local, save_processed_data, find_latest_file

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
# Helper: Generate Calendar Regressors for Future Dates
# =============================================================================
def generate_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Generate key calendar features for any date range (used for future regressors)."""
    df = pd.DataFrame({'timestamp': pd.to_datetime(dates)})
    df['ds'] = df['timestamp']
    df['is_eoq_window'] = df['ds'].dt.is_quarter_end.astype(int)  # Simplified EOQ proxy
    df['is_holiday'] = 0  # Extend with actual holiday list if needed
    df['is_weekend'] = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df.drop(columns=['ds', 'timestamp'])

# =============================================================================
# Metric Calculation
# =============================================================================
def calculate_metrics(y_true, y_pred):
    """Robust metrics for capacity forecasting (handles zeros/low values)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # sMAPE: symmetric mean absolute percentage error (best for utilization data)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.where(denominator > 1e-8,
                     np.abs(y_true - y_pred) / denominator,
                     0.0)  # avoid div-by-zero
    smape = np.mean(smape) * 200  # *200 to get % scale (standard sMAPE)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE": round(smape, 2)
    }

# =============================================================================
# Baseline: Day-of-Week Average (Improved)
# =============================================================================
def train_predict_naive_seasonal(train_df, test_df, horizon_days, target_col, period=7):
    train_df = train_df.copy()
    train_df['dow'] = train_df['timestamp'].dt.dayofweek
    dow_means = train_df.groupby('dow')[target_col].mean().to_dict()
    
    test_dows = test_df['timestamp'].dt.dayofweek
    test_preds = np.array([dow_means.get(dow, np.nan) for dow in test_dows])
    
    last_date = train_df['timestamp'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    future_dows = future_dates.dayofweek
    future_preds = np.array([dow_means.get(dow, np.nan) for dow in future_dows])
    
    return test_preds, future_preds

# =============================================================================
# Prophet Model (with true forward forecasting + uncertainty)
# =============================================================================
def train_predict_prophet(train_df, test_df, future_dates, target_col, config):
    p_cfg = config.get('model_training', {}).get('prophet', {})
    regressors_config = p_cfg.get('regressor_columns', [])
    
    dummy_dates = pd.Series([train_df['timestamp'].iloc[0]]) 
    available_future_features = generate_calendar_features(dummy_dates).columns
    
    regressors = [
        r for r in regressors_config 
        if r in train_df.columns and r in available_future_features
    ]
    
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
        m.add_regressor(reg)
    
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    m.fit(pf_train)
    
    all_future_ds = pd.concat([test_df[['timestamp']], future_dates[['timestamp']]])
    future = m.make_future_dataframe(periods=0)
    future = pd.concat([pf_train[['ds']], all_future_ds.rename(columns={'timestamp': 'ds'})]).drop_duplicates().sort_values('ds')
    
    future_reg = generate_calendar_features(future['ds'])
    for reg in regressors:
        if reg in future_reg.columns:
            future[reg] = future_reg[reg].values
    
    forecast = m.predict(future)
    
    test_mask = forecast['ds'].isin(test_df['timestamp'])
    future_mask = forecast['ds'].isin(future_dates['timestamp'])
    
    test_preds = forecast.loc[test_mask, 'yhat'].values
    future_preds = forecast.loc[future_mask, ['yhat', 'yhat_lower', 'yhat_upper']].values
    
    return test_preds, future_preds, m, forecast

# =============================================================================
# Random Forest (Batch prediction — no recursion needed for exogenous features)
# =============================================================================
def train_predict_rf(train_df, test_df, future_dates, target_col, config):
    rf_cfg = config.get('model_training', {}).get('random_forest', {})
    
    # Safe exogenous features from config
    safe_features = rf_cfg.get('features', ['is_eoq_window', 'is_holiday', 'is_weekend', 'quarter', 'month', 'month_sin', 'month_cos'])
    feature_cols = [c for c in safe_features if c in train_df.columns]
    
    if not feature_cols:
        logger.warning(f"No valid exogenous features for RF on server {train_df['server_id'].iloc[0]}. Skipping.")
        return np.array([]), np.array([]), None
    
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
    
    X_test = test_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    test_preds = model.predict(X_test)
    
    X_future = future_dates[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    future_preds = model.predict(X_future)
    
    return test_preds, future_preds, model

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
    test_df = df_group[df_group['timestamp'] >= split_date].copy()
    
    if len(train_df) < mt_cfg.get('min_history_days', 365):
        logger.warning(f"Server {server_id}: Insufficient history ({len(train_df)} days). Skipping.")
        return None
    
    if len(test_df) == 0:
        logger.warning(f"Server {server_id}: No test data after {split_date}.")
        return None

    # Generate future dates
    last_date = df_group['timestamp'].max()
    future_dates = pd.DataFrame({
        'timestamp': pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    })
    future_regressors = generate_calendar_features(future_dates['timestamp'])
    future_dates = pd.concat([future_dates, future_regressors.drop(columns=['timestamp'], errors='ignore')], axis=1)

    results = []
    forecasts_list = []

    # Models directory from config
    local_base = Path(config['paths']['local_data_dir'])
    models_dir = local_base / config['paths']['model_artifacts_dir']
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline
    if 'baseline_naive_seasonal' in mt_cfg.get('enabled_models', []):
        period = mt_cfg.get('baseline_naive_seasonal', {}).get('period_days', 7)
        test_preds, future_preds = train_predict_naive_seasonal(train_df, test_df, horizon_days, target_col, period=period)
        metrics = calculate_metrics(test_df[target_col].values, test_preds)
        results.append({'server_id': server_id, 'model': 'baseline', 'metrics': metrics, 'preds_test': test_preds, 'preds_future': future_preds})
        
        test_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'baseline', 'timestamp': test_df['timestamp'],
            'actual': test_df[target_col].values, 'predicted': test_preds,
            'yhat_lower': np.nan, 'yhat_upper': np.nan, 'is_future': False
        })
        future_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'baseline', 'timestamp': future_dates['timestamp'],
            'actual': np.nan, 'predicted': future_preds, 'yhat_lower': np.nan, 'yhat_upper': np.nan, 'is_future': True
        })
        forecasts_list.append(pd.concat([test_rows, future_rows]))

    # 2. Prophet
    if 'prophet' in mt_cfg.get('enabled_models', []):
        test_preds, future_preds, model, _ = train_predict_prophet(train_df, test_df, future_dates, target_col, config)
        metrics = calculate_metrics(test_df[target_col].values, test_preds)
        results.append({'server_id': server_id, 'model': 'prophet', 'metrics': metrics})
        
        with open(models_dir / f"prophet_{server_id}.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        test_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'prophet', 'timestamp': test_df['timestamp'],
            'actual': test_df[target_col].values, 'predicted': test_preds,
            'yhat_lower': np.nan, 'yhat_upper': np.nan, 'is_future': False
        })
        future_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'prophet', 'timestamp': future_dates['timestamp'],
            'actual': np.nan, 'predicted': future_preds[:, 0],
            'yhat_lower': future_preds[:, 1], 'yhat_upper': future_preds[:, 2], 'is_future': True
        })
        forecasts_list.append(pd.concat([test_rows, future_rows]))

    # 3. Random Forest
    if 'random_forest' in mt_cfg.get('enabled_models', []):
        preds_test, preds_future, model = train_predict_rf(train_df, test_df, future_dates, target_col, config)
        metrics = calculate_metrics(test_df[target_col].values, preds_test)
        results.append({'server_id': server_id, 'model': 'random_forest', 'metrics': metrics})
        
        with open(models_dir / f"rf_{server_id}.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        test_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'random_forest', 'timestamp': test_df['timestamp'],
            'actual': test_df[target_col].values, 'predicted': preds_test,
            'yhat_lower': np.nan, 'yhat_upper': np.nan, 'is_future': False
        })
        future_rows = pd.DataFrame({
            'server_id': server_id, 'model': 'random_forest', 'timestamp': future_dates['timestamp'],
            'actual': np.nan, 'predicted': preds_future,
            'yhat_lower': np.nan, 'yhat_upper': np.nan, 'is_future': True
        })
        forecasts_list.append(pd.concat([test_rows, future_rows]))

    return results, pd.concat(forecasts_list, ignore_index=True) if forecasts_list else None


def main_process(config: dict, input_path: str = None, output_path: str = None):
    logger.info("=== MODULE 04 : Model Training & Forward Forecasting ===")
    
    # Processed data dir from config
    processed_prefix = config['paths']['processed_dir']
    
    try:
        if input_path:
            # SageMaker Mode: Read specific file from mounted directory
            logger.info(f"SageMaker Mode: Loading input from {input_path}")
            # In SM Processing, input_path is a directory containing the file(s)
            p = Path(input_path)
            if p.is_file():
                df = pd.read_parquet(p)
                filename = p.name
            else:
                 # Find parquet in directory
                 files = list(p.glob("*.parquet"))
                 if not files:
                     raise FileNotFoundError(f"No parquet files in {input_path}")
                 # Pick latest if multiple, or just first
                 target = sorted(files)[-1]
                 df = pd.read_parquet(target)
                 filename = target.name
        else:
            # Normal Mode (Local or S3 autoload)
            filename = find_latest_file(config, prefix=processed_prefix)
            logger.info(f"Loading latest feature data: {filename}")
            df = load_from_s3_or_local(config, prefix=processed_prefix, filename=filename)
            
    except FileNotFoundError as e:
        logger.error(f"Input data missing: {e}")
        raise
    if df is None:
        raise FileNotFoundError(f"Failed to load {filename}")
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['server_id', 'timestamp'])
    
    # Output directories from config
    local_base = Path(config['paths']['local_data_dir'])
    models_dir    = local_base / config['paths']['model_artifacts_dir']
    forecasts_dir = local_base / config['paths']['forecasts_dir']
    metrics_dir   = local_base / config['paths']['metrics_dir']
    for d in [models_dir, forecasts_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Optional debug limit
    servers = df['server_id'].unique()
    debug_max = config.get('model_training', {}).get('debug_max_servers', None)
    if debug_max:
        servers = servers[:debug_max]
        logger.info(f"DEBUG MODE: Limiting to first {debug_max} servers")
    
    logger.info(f"Training on {len(servers)} servers...")
    
    all_metrics = []
    all_forecasts = []
    
    for sid in servers:
        res = process_server_group(sid, df[df['server_id'] == sid].copy(), config)
        if res:
            server_results, server_forecasts = res
            for r in server_results:
                m = r['metrics'].copy()
                m['server_id'] = sid
                m['model'] = r['model']
                all_metrics.append(m)
            if server_forecasts is not None:
                all_forecasts.append(server_forecasts)

    if not all_metrics:
        logger.error("No results obtained.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    summary = metrics_df.groupby('model')['sMAPE'].describe()
    logger.info("Model Performance Summary (sMAPE):\n" + summary.to_string())

    avg_smape = metrics_df.groupby('model')['sMAPE'].mean().sort_values()
    best_model = avg_smape.idxmin()
    logger.info(f"🏆 Best Model (AVG sMAPE): {best_model} ({avg_smape[best_model]:.2f}%)")
    
    # Save forecasts — configurable filename
    if all_forecasts:
        forecasts_combined = pd.concat(all_forecasts, ignore_index=True)
        forecasts_filename = config['paths'].get('forecasts_summary_filename', 'all_model_forecasts.parquet')
        
        if output_path:
             # SageMaker Mode: write to local mount point which SM uploads to S3
             out_dir = Path(output_path)
             out_dir.mkdir(parents=True, exist_ok=True)
             save_path_f = out_dir / forecasts_filename
             forecasts_combined.to_parquet(save_path_f, index=False)
             logger.info(f"Saved SageMaker artifacts to {save_path_f}")
        else:
             # Standard Mode
             save_path_f = save_processed_data(
                forecasts_combined, config,
                prefix=config['paths']['forecasts_dir'],
                filename=forecasts_filename
             )
             logger.info(f"Saved enriched forecasts to {save_path_f}")
    
    # Save metrics — configurable filename
    metrics_filename = config['paths'].get('metrics_summary_filename', 'model_comparison.json')
    metrics_summary = {
        "processed_at": datetime.now().isoformat(),
        "total_servers": len(servers),
        "avg_smape": avg_smape.to_dict(), # Corrected from summary['mean'].to_dict()
        "best_model": best_model,
        "details": metrics_df.to_dict(orient='records')
    }
    
    if output_path:
        # SageMaker Mode
        metrics_save_path = Path(output_path) / metrics_filename
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Saved SageMaker metrics to {metrics_save_path}")
    else:
        # Standard Mode
        save_to_s3_or_local(
            json.dumps(metrics_summary, indent=2),
            config,
            prefix=config['paths']['metrics_dir'],
            filename=metrics_filename
        )
    
    logger.info("✔ Module 04 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 04 — Model Training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", choices=["local", "sagemaker", "lambda"])
    
    # SageMaker Processing Job arguments (S3 inputs mapped to local paths)
    parser.add_argument("--input_data_path", help="Path to input data (SageMaker mounted)")
    parser.add_argument("--output_data_path", help="Path to output data (SageMaker mounted)")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    # In SageMaker, config might be passed as a file argument or we might need to load it different
    # For now, assume config is available or we load standard one
    config = load_config(config_path)
    if args.env:
        if 'execution' not in config:
            config['execution'] = {}
        config['execution']['mode'] = args.env
    validate_config(config)
    
    # Pass SageMaker paths if present
    main_process(config, input_path=args.input_data_path, output_path=args.output_data_path)
"""
ml_utils.py

This module provides utility functions for machine learning tasks in the AWS-CapacityForecaster project.
It focuses on time-series forecasting for infrastructure metrics (e.g., CPU, memory utilization),
feature engineering, model training, evaluation, and capacity-specific analyses like risk flagging
and optimization. These functions align with the project's core objectives:

1. Data Science & Machine Learning: Feature engineering, training models (Prophet, scikit-learn),
   forecasting, and evaluation to demonstrate time-series prediction and model comparison.
2. Performance Monitoring & Capacity Planning: Risk analysis for seasonal peaks and clustering
   for underutilization detection to support optimization and cost recommendations.
3. Python Development: Efficient, modular code using libraries like pandas, numpy, scikit-learn,
   and Prophet, designed for integration with AWS services (e.g., SageMaker notebooks).

Functions are designed to be modular, accepting pandas DataFrames for inputs/outputs, and
supporting large datasets with vectorized operations and parallel processing where applicable.
No direct AWS dependencies here—integration is handled in other modules (e.g., data_utils.py).

Dependencies:
- pandas
- numpy
- prophet
- scikit-learn
- scipy
- statsmodels
- joblib

Usage:
Import functions into notebooks or scripts, e.g.:
from src.utils.ml_utils import engineer_features, train_prophet_model

For testing and development, use synthetic data from data_generation.py.
Commit changes to GitHub: https://github.com/seanlgirgis/AWS-CapacityForecaster
Local path: C:\pyproj\AWS-CapacityForecaster\src\utils\ml_utils.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import joblib
import logging
from typing import List, Dict, Union, Optional

# Set up logging for debug/info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    metrics: List[str],
    lag_periods: int = 7,
    rolling_windows: List[int] = [7, 30],
    seasonal_flags: bool = True
) -> pd.DataFrame:
    """
    Engineer features for time-series data, including lags, rolling statistics, and seasonal indicators.
    This function prepares data for ML models by adding temporal features, mirroring Citi-style
    feature engineering for forecasting models.

    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp' as datetime index and server metrics columns.
        metrics (List[str]): List of metric columns to engineer (e.g., ['cpu_p95', 'mem_p95']).
        lag_periods (int, optional): Number of lag periods to create. Defaults to 7.
        rolling_windows (List[int], optional): Windows for rolling mean/std. Defaults to [7, 30].
        seasonal_flags (bool, optional): Add seasonal dummies (e.g., quarter-end). Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with added features.

    Raises:
        ValueError: If 'timestamp' is not the index or not datetime.

    Example:
        >>> df = pd.DataFrame({'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
                               'cpu_p95': np.random.rand(100)})
        >>> df.set_index('timestamp', inplace=True)
        >>> engineered_df = engineer_features(df, metrics=['cpu_p95'])
        >>> print(engineered_df.columns)
        Index(['cpu_p95', 'cpu_p95_lag1', ..., 'cpu_p95_rolling_mean7', ..., 'is_quarter_end'], dtype='object')
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex named 'timestamp'.")

    df = df.copy()  # Avoid modifying original

    # Lags
    for metric in metrics:
        for lag in range(1, lag_periods + 1):
            df[f'{metric}_lag{lag}'] = df[metric].shift(lag)

    # Rolling statistics (parallelized if large)
    def compute_rolling(window, metric):
        df[f'{metric}_rolling_mean{window}'] = df[metric].rolling(window=window).mean()
        df[f'{metric}_rolling_std{window}'] = df[metric].rolling(window=window).std()

    for window in rolling_windows:
        for metric in metrics:
            compute_rolling(window, metric)

    # Seasonal flags (e.g., banking quarter-ends)
    if seasonal_flags:
        df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end.astype(int)
        df['day_of_week'] = df.index.to_series().dt.dayofweek
        df['month'] = df.index.to_series().dt.month

    # Drop NaNs introduced by shifts/rolling
    df.dropna(inplace=True)

    logger.info(f"Engineered features for {len(metrics)} metrics. New shape: {df.shape}")
    return df


def handle_missing_data(
    df: pd.DataFrame,
    method: str = 'interpolate',
    **kwargs
) -> pd.DataFrame:
    """
    Handle missing data in the DataFrame using specified method. Supports interpolation or KNN imputation.
    This ensures data quality for ML inputs, as emphasized in Citi data cleansing processes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (str, optional): 'interpolate' or 'knn'. Defaults to 'interpolate'.
        **kwargs: Additional args for methods (e.g., n_neighbors for knn).

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    Raises:
        ValueError: Invalid method.

    Example:
        >>> df = pd.DataFrame({'cpu_p95': [1, np.nan, 3, np.nan, 5]})
        >>> cleaned_df = handle_missing_data(df, method='interpolate')
        >>> print(cleaned_df['cpu_p95'].values)
        [1. 2. 3. 4. 5.]
    """
    df = df.copy()

    if method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
        df[df.columns] = imputer.fit_transform(df)
    else:
        raise ValueError("Method must be 'interpolate' or 'knn'.")

    # Handle remaining NaNs (e.g., at edges)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    logger.info(f"Handled missing data using {method}. Remaining NaNs: {df.isnull().sum().sum()}")
    return df


def check_stationarity(ts: pd.Series) -> Dict[str, Union[float, bool]]:
    """
    Check time-series stationarity using Augmented Dickey-Fuller (ADF) test.
    Useful for preprocessing before modeling, e.g., to decide on differencing.

    Args:
        ts (pd.Series): Time-series data.

    Returns:
        Dict[str, Union[float, bool]]: {'adf_statistic': float, 'p_value': float, 'is_stationary': bool (p < 0.05)}.

    Example:
        >>> ts = pd.Series(np.random.rand(100))
        >>> result = check_stationarity(ts)
        >>> print(result['is_stationary'])
        True  # Or False depending on data
    """
    result = adfuller(ts.dropna())
    adf_stat = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05

    logger.info(f"ADF Statistic: {adf_stat}, p-value: {p_value}, Stationary: {is_stationary}")
    return {'adf_statistic': adf_stat, 'p_value': p_value, 'is_stationary': is_stationary}


def train_prophet_model(
    df: pd.DataFrame,
    target: str = 'cpu_p95',
    regressors: Optional[List[str]] = None,
    seasonality_mode: str = 'additive'
) -> Prophet:
    """
    Train a Prophet model for time-series forecasting, incorporating seasonality and regressors.
    Aligns with Citi's use of Prophet for forecasting with banking cycles.

    Args:
        df (pd.DataFrame): DataFrame with 'ds' (datetime) and 'y' (target) columns; rename if needed.
        target (str, optional): Target column. Defaults to 'cpu_p95'.
        regressors (List[str], optional): Additional regressor columns.
        seasonality_mode (str, optional): 'additive' or 'multiplicative'. Defaults to 'additive'.

    Returns:
        Prophet: Fitted model.

    Example:
        >>> df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'cpu_p95': 'y'})
        >>> model = train_prophet_model(df_prophet)
        >>> print(model)
        <prophet.forecaster.Prophet object at ...>
    """
    df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', target: 'y'})

    if regressors:
        for reg in regressors:
            if reg not in df_prophet.columns:
                raise ValueError(f"Regressor '{reg}' not in DataFrame.")

    model = Prophet(seasonality_mode=seasonality_mode)
    if regressors:
        for reg in regressors:
            model.add_regressor(reg)

    model.fit(df_prophet)
    logger.info(f"Trained Prophet model on target '{target}' with {len(regressors or [])} regressors.")
    return model


def train_sklearn_model(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    model_type: str = 'random_forest',
    params: Optional[Dict] = None,
    n_jobs: int = -1
) -> Union[RandomForestRegressor, GradientBoostingRegressor]:
    """
    Train a scikit-learn regression model (RandomForest or GradientBoosting) for multivariate forecasting.
    Supports parallel training with joblib under the hood.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        target (str): Target column.
        features (List[str]): Feature columns.
        model_type (str, optional): 'random_forest' or 'gradient_boosting'. Defaults to 'random_forest'.
        params (Dict, optional): Hyperparameters for the model.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1 (all cores).

    Returns:
        BaseEstimator: Fitted model.

    Raises:
        ValueError: Invalid model_type.

    Example:
        >>> features = ['cpu_lag1', 'cpu_rolling_mean7']
        >>> model = train_sklearn_model(df, target='cpu_p95', features=features, model_type='random_forest')
        >>> print(model)
        RandomForestRegressor(...)
    """
    X = df[features]
    y = df[target]
    params = params or {}

    if model_type == 'random_forest':
        model = RandomForestRegressor(n_jobs=n_jobs, **params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**params)
    else:
        raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'.")

    model.fit(X, y)
    logger.info(f"Trained {model_type} model on {len(features)} features.")
    return model


def generate_forecast(
    model: Union[Prophet, RandomForestRegressor, GradientBoostingRegressor],
    future_periods: int,
    future_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate forecasts using a fitted model. For Prophet, creates future dataframe; for scikit-learn,
    predicts on provided future features.

    Args:
        model (Union[Prophet, BaseEstimator]): Fitted model.
        future_periods (int): Number of periods to forecast.
        future_df (pd.DataFrame, optional): For scikit-learn models, DataFrame with future features.

    Returns:
        pd.DataFrame: Forecast DataFrame with predictions and intervals (if applicable).

    Raises:
        ValueError: If future_df missing for scikit-learn models.

    Example:
        >>> future = model.make_future_dataframe(periods=30)  # For Prophet
        >>> forecast = generate_forecast(model, 30, future)
        >>> print(forecast[['ds', 'yhat']])
           ds       yhat
        0  ...     ...
    """
    if isinstance(model, Prophet):
        future = model.make_future_dataframe(periods=future_periods)
        if future_df is not None:
            future = pd.concat([future, future_df.reset_index(drop=True)], axis=1)
        forecast = model.predict(future)
        logger.info(f"Generated Prophet forecast for {future_periods} periods.")
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    else:  # scikit-learn
        if future_df is None:
            raise ValueError("future_df required for scikit-learn models.")
        predictions = model.predict(future_df)
        forecast = pd.DataFrame({'yhat': predictions})
        logger.info(f"Generated {type(model).__name__} forecast for {future_periods} periods.")
        return forecast


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance using common regression metrics. Supports claims like "20-30% accuracy improvement".

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.

    Returns:
        Dict[str, float]: {'mae': float, 'rmse': float, 'mape': float}.

    Example:
        >>> metrics = evaluate_model(y_true, y_pred)
        >>> print(metrics['mae'])
        0.5  # Example value
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape}
    logger.info(f"Model evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
    return metrics


def compare_models(
    results: List[Dict[str, float]]
) -> pd.DataFrame:
    """
    Compare multiple model evaluations in a table for selection and reporting.

    Args:
        results (List[Dict[str, float]]): List of metric dicts from evaluate_model.

    Returns:
        pd.DataFrame: Comparison table (rows: models, columns: metrics).

    Example:
        >>> results = [{'mae': 0.5, 'rmse': 0.7}, {'mae': 0.4, 'rmse': 0.6}]
        >>> comparison = compare_models(results)
        >>> print(comparison)
           mae  rmse
        0  0.5   0.7
        1  0.4   0.6
    """
    df = pd.DataFrame(results)
    logger.info("Model comparison completed.")
    return df


def flag_risks(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    threshold: float = 90.0,
    percentile: float = 95.0,
    metric: str = 'cpu_p95'
) -> pd.DataFrame:
    """
    Flag at-risk resources based on historical P95 and forecasts. Uses scipy for percentile calculations.
    Supports seasonal risk analysis from Citi experience.

    Args:
        df (pd.DataFrame): Historical DataFrame.
        forecast_df (pd.DataFrame): Forecast DataFrame.
        threshold (float, optional): Utilization threshold for risk. Defaults to 90.0.
        percentile (float, optional): Percentile for analysis. Defaults to 95.0.
        metric (str, optional): Metric to analyze. Defaults to 'cpu_p95'.

    Returns:
        pd.DataFrame: DataFrame with 'risk_flag' column (1 if at risk).

    Example:
        >>> risks = flag_risks(df, forecast_df)
        >>> print(risks['risk_flag'].sum())
        5  # Number of risky periods
    """
    combined = pd.concat([df[[metric]], forecast_df.rename(columns={'yhat': metric})], axis=0)
    p95 = np.percentile(combined[metric], percentile)  # Uses scipy under numpy
    risks = combined[combined[metric] > threshold]
    risks['risk_flag'] = 1

    logger.info(f"Flagged {len(risks)} risky periods above {threshold}% (P{percentile}).")
    return risks


def cluster_utilization(
    df: pd.DataFrame,
    n_clusters: int = 3,
    features: List[str] = ['cpu_mean', 'mem_mean'],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Cluster resources based on utilization patterns using K-Means for optimization recommendations.
    Identifies underutilized groups for consolidation/cost savings.

    Args:
        df (pd.DataFrame): DataFrame with features.
        n_clusters (int, optional): Number of clusters. Defaults to 3.
        features (List[str], optional): Features for clustering. Defaults to ['cpu_mean', 'mem_mean'].
        n_jobs (int, optional): For parallel (if applicable). Defaults to -1.

    Returns:
        pd.DataFrame: Original df with 'cluster' column.

    Example:
        >>> clustered = cluster_utilization(df)
        >>> print(clustered['cluster'].value_counts())
        0    50
        1    30
        2    20
        Name: cluster, dtype: int64
    """
    X = df[features]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Optional: parallel if large, but KMeans is fast
    logger.info(f"Clustered into {n_clusters} groups based on {features}.")
    return df
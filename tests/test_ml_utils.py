"""
# Run command: python -m pytest tests/test_ml_utils.py -v
Unit Tests for ML Utilities (src/utils/ml_utils.py)

This module contains a comprehensive test suite for the machine learning utility functions.
It validates:
1.  Feature Engineering correctness (lags, rolling stats).
2.  Data cleansing (missing value imputation).
3.  Model training and forecasting execution (Prophet & Scikit-Learn).
4.  Evaluation metrics calculation.
5.  Risk flagging and clustering logic.

Tests use synthetic data and mocked configurations to ensure isolation.
"""

import pytest
import os
import logging
import warnings

# Suppress joblib/loky warning on Windows (must be before sklearn import)
if os.name == 'nt':
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

# Suppress Prophet/cmdstanpy warnings before import
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*cmdstanpy.*")
# Suppress joblib/loky physical core warning
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from src.utils.ml_utils import (
    engineer_features,
    handle_missing_data,
    check_stationarity,
    train_prophet_model,
    train_sklearn_model,
    generate_forecast,
    evaluate_model,
    compare_models,
    flag_risks,
    cluster_utilization
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_df():
    """Create a sample daily dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "cpu_p95": np.random.rand(50),
        "mem_p95": np.random.rand(50)
    })
    df.set_index("timestamp", inplace=True)
    return df

@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration to avoid dependency on local config files."""
    # We patch the getter functions imported in ml_utils
    with patch("src.utils.ml_utils.get_feature_engineering_config") as mock_fe, \
         patch("src.utils.ml_utils.get_ml_config") as mock_ml, \
         patch("src.utils.ml_utils.get_risk_config") as mock_risk:
        
        mock_fe.return_value = {"lags": [1, 7], "rolling_windows": [7]}
        mock_ml.return_value = {"models": [{"name": "random_forest", "params": {"n_estimators": 10}}]}
        mock_risk.return_value = {
            "high_risk_threshold": 0.9, 
            "clustering": {"n_clusters": 2, "features": ["cpu_mean", "mem_mean"]}
        }
        yield

# -----------------------------
# Tests
# -----------------------------

def test_engineer_features(sample_df):
    """Test lag, rolling window, and seasonality feature creation."""
    df = engineer_features(
        sample_df, 
        metrics=["cpu_p95"], 
        lag_periods=[1], 
        rolling_windows=[7]
    )
    
    # Check columns exist
    expected_cols = ["cpu_p95_lag1", "cpu_p95_rolling_mean7", "cpu_p95_rolling_std7"]
    for col in expected_cols:
        assert col in df.columns
        
    # Check rows dropped due to NaN (rolling window 7 -> first 6 dropped usually)
    # Plus lag 1.
    assert len(df) < 50
    assert not df.isnull().values.any()

def test_engineer_features_defaults(sample_df):
    """Test that it uses config defaults when args not provided."""
    # Config mocked to lags=[1, 7], rolling=[7]
    df = engineer_features(sample_df, metrics=["cpu_p95"])
    assert "cpu_p95_lag7" in df.columns
    assert "cpu_p95_rolling_mean7" in df.columns

def test_handle_missing_data_interpolate():
    """Test interpolation logic."""
    df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
    clean = handle_missing_data(df, method="interpolate")
    assert clean["val"].iloc[1] == 2.0
    assert not clean.isnull().values.any()

def test_check_stationarity(sample_df):
    """Test ADF check."""
    # Mock adfuller since it's a stats call
    with patch("src.utils.ml_utils.adfuller") as mock_adf:
        mock_adf.return_value = (-3.5, 0.01, 0, 0, {}, 0) # Stationary result
        result = check_stationarity(sample_df["cpu_p95"])
        assert result["is_stationary"] is True
        assert result["p_value"] == 0.01

def test_train_prophet_model(sample_df):
    """Test basic Prophet model training."""
    # We need to rename index to fit expected input if not done by caller? 
    # train_prophet_model handles reset_index internally.
    model = train_prophet_model(sample_df, target="cpu_p95")
    assert isinstance(model, Prophet)

def test_train_sklearn_model(sample_df):
    """Test sklearn model training."""
    # Create features for valid training
    df = sample_df.copy()
    df["feature1"] = df["cpu_p95"].shift(1)
    df.dropna(inplace=True)
    
    model = train_sklearn_model(
        df, 
        target="cpu_p95", 
        features=["feature1"],
        model_type="random_forest"
    )
    assert isinstance(model, RandomForestRegressor)
    # Check n_estimators from mock config (10)
    assert model.n_estimators == 10

def test_generate_forecast_sklearn(sample_df):
    """Test forecasting with sklearn model."""
    df = sample_df.copy()
    df["feature1"] = np.random.rand(len(df))
    
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5] * 5)
    
    future_df = pd.DataFrame({"feature1": [0.1]*5})
    forecast = generate_forecast(mock_model, 5, future_df=future_df)
    
    assert len(forecast) == 5
    assert "yhat" in forecast.columns
    assert (forecast["yhat"] == 0.5).all()

def test_evaluate_model():
    """Test metric calculation."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 4.0]) # Last one off by 1
    
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["mae"] == 1.0 / 3.0 # (0+0+1)/3 = 0.333
    assert "rmse" in metrics
    assert "mape" in metrics

def test_flag_risks():
    """Test risk flagging logic."""
    hist_df = pd.DataFrame({"cpu_p95": [0.5, 0.6]})
    forecast_df = pd.DataFrame({"yhat": [0.8, 0.95]}) # 0.95 is > 0.9 threshold (mocked)
    
    risks = flag_risks(hist_df, forecast_df, metric="cpu_p95")
    # Should flag the row with 0.95
    assert len(risks) == 1
    assert risks["risk_flag"].iloc[0] == 1
    assert risks["cpu_p95"].iloc[0] == 0.95 # Renamed yhat -> metric

def test_cluster_utilization():
    """Test KMeans clustering."""
    df = pd.DataFrame({
        "cpu_mean": [0.1, 0.1, 0.9, 0.9],
        "mem_mean": [0.1, 0.1, 0.9, 0.9]
    })
    # Mocked config: n_clusters=2
    clustered = cluster_utilization(df)
    
    assert "cluster" in clustered.columns
    # Should ideally separate low (0.1) and high (0.9) into different clusters
    counts = clustered["cluster"].value_counts()
    assert len(counts) == 2 # 2 clusters found

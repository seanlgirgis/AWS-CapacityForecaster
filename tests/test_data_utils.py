
# Run command: python -m pytest tests/test_data_utils.py -v
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import boto3
from botocore.exceptions import ClientError
from src.utils.data_utils import (
    generate_synthetic_server_metrics,
    generate_server_metadata,
    merge_metrics_with_metadata,
    validate_capacity_df,
    add_calendar_features,
    clean_numerical_columns,
    resample_to_daily,
    detect_anomalies_simple,
    load_from_s3,
    get_date_range,
    METRIC_COLUMNS,
    SERVER_METADATA_COLUMNS
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_metrics_df():
    """Create a small valid metrics DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "server_id": ["server_001"] * 5,
        "date": dates,
        "cpu_p95": [0.1, 0.2, 0.3, 0.4, 0.5],
        "memory_p95": [0.5, 0.5, 0.5, 0.5, 0.5],
        "disk_p95": [0.8, 0.8, 0.8, 0.8, 0.8],
        "network_out_p95": [0.1, 0.1, 0.1, 0.1, 0.1],
    })

# -----------------------------
# Test: generate_synthetic_server_metrics
# -----------------------------

def test_generate_synthetic_server_metrics_structure():
    """Test output structure and basic properties."""
    df = generate_synthetic_server_metrics(n_servers=2, start_date="2024-01-01", end_date="2024-01-10", freq="D")
    
    assert isinstance(df, pd.DataFrame)
    assert "server_id" in df.columns
    assert "date" in df.columns
    for col in METRIC_COLUMNS:
        assert col in df.columns
        
    # Check row count: 2 servers * 10 days = 20 rows
    assert len(df) == 20
    
    # Check value ranges (should be [0, 1] due to clipping)
    assert df[METRIC_COLUMNS].min().min() >= 0.0
    assert df[METRIC_COLUMNS].max().max() <= 1.0

def test_generate_synthetic_metrics_defaults():
    """Test that it runs with defaults (pulling from config/default args)."""
    df = generate_synthetic_server_metrics(n_servers=1)
    assert not df.empty
    assert "cpu_p95" in df.columns

def test_generate_synthetic_metrics_seasonality_and_spikes():
    """Test that seasonality and spikes don't crash the generation."""
    df = generate_synthetic_server_metrics(
        n_servers=10, 
        add_seasonality=True, 
        add_eoq_spikes=True,
        noise_level=0.0
    )
    assert not df.empty

# -----------------------------
# Test: generate_server_metadata
# -----------------------------

def test_generate_server_metadata():
    """Test metadata generation structure."""
    n = 5
    df = generate_server_metadata(n_servers=n)
    
    assert len(df) == n
    assert list(df.columns) == SERVER_METADATA_COLUMNS
    assert df["server_id"].nunique() == n
    assert df["criticality"].isin(['High', 'Medium', 'Low']).all()

# -----------------------------
# Test: merge_metrics_with_metadata
# -----------------------------

def test_merge_metrics_with_metadata(sample_metrics_df):
    """Test merging metrics with metadata."""
    meta_df = pd.DataFrame({
        "server_id": ["server_001"],
        "region": ["us-east-1"],
        "criticality": ["High"]
    })
    
    merged = merge_metrics_with_metadata(sample_metrics_df, meta_df)
    
    assert "region" in merged.columns
    assert "criticality" in merged.columns
    assert merged["region"].iloc[0] == "us-east-1"
    # Ensure no rows dropped
    assert len(merged) == len(sample_metrics_df)

# -----------------------------
# Test: validate_capacity_df
# -----------------------------

def test_validate_capacity_df_valid(sample_metrics_df):
    """Test validation with clean data."""
    is_valid, issues = validate_capacity_df(sample_metrics_df)
    assert is_valid
    assert len(issues) == 0

def test_validate_capacity_df_missing_columns(sample_metrics_df):
    """Test missing mandatory column."""
    bad_df = sample_metrics_df.drop(columns=["cpu_p95"])
    is_valid, issues = validate_capacity_df(bad_df)
    assert not is_valid
    assert any("Missing columns" in i for i in issues)

def test_validate_capacity_df_invalid_date(sample_metrics_df):
    """Test non-datetime date column."""
    sample_metrics_df["date"] = "not-a-date"
    # It might run depending on pandas version, but check type check
    is_valid, issues = validate_capacity_df(sample_metrics_df)
    assert not is_valid
    assert "'date' column is not datetime" in issues

def test_validate_capacity_df_negative_values(sample_metrics_df):
    """Test negative value detection."""
    sample_metrics_df.loc[0, "cpu_p95"] = -0.01
    is_valid, issues = validate_capacity_df(sample_metrics_df, allow_negative=False)
    assert not is_valid
    assert "Negative values in cpu_p95" in issues

def test_validate_capacity_df_out_of_bounds(sample_metrics_df):
    """Test values > 1.0 (warning)."""
    sample_metrics_df.loc[0, "cpu_p95"] = 1.5
    is_valid, issues = validate_capacity_df(sample_metrics_df)
    assert not is_valid
    assert "Values >1 in cpu_p95 (clip recommended)" in issues

# -----------------------------
# Test: add_calendar_features
# -----------------------------

def test_add_calendar_features():
    """Test feature engineering."""
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-03-31", "2024-07-04"])})
    
    df_feat = add_calendar_features(df, include_holidays=True)
    
    assert "year" in df_feat.columns
    assert "is_eoq" in df_feat.columns
    assert "is_holiday" in df_feat.columns
    
    # Check specific logic
    # Jan 1 is holiday (New Year)
    assert df_feat.loc[0, "is_holiday"] == 1
    # Mar 31 is EOQ
    assert df_feat.loc[1, "is_eoq"] == 1
    # Jul 4 is holiday (Independence Day)
    assert df_feat.loc[2, "is_holiday"] == 1

# -----------------------------
# Test: clean_numerical_columns
# -----------------------------

def test_clean_numerical_columns_clipping():
    """Test value clipping."""
    df = pd.DataFrame({"server_id": [1, 1], "cpu": [-0.5, 1.5]})
    cleaned = clean_numerical_columns(df, columns=["cpu"], clip_min=0.0, clip_max=1.0)
    
    assert cleaned["cpu"].min() == 0.0
    assert cleaned["cpu"].max() == 1.0

def test_clean_numerical_columns_interpolation():
    """Test missing value interpolation."""
    df = pd.DataFrame({
        "server_id": ["s1", "s1", "s1"],
        "cpu": [0.1, np.nan, 0.3]
    })
    
    cleaned = clean_numerical_columns(df, columns=["cpu"], fill_method="interpolate")
    assert not cleaned["cpu"].isnull().any()
    assert context_approx(cleaned.iloc[1]["cpu"], 0.2)

def context_approx(val, target, tol=0.001):
    return abs(val - target) < tol

# -----------------------------
# Test: resample_to_daily
# -----------------------------

def test_resample_to_daily():
    """Test resampling from hourly to daily P95 (max)."""
    df = pd.DataFrame({
        "server_id": ["s1", "s1", "s1"],
        "date": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00", "2024-01-02 08:00"]),
        "cpu": [0.5, 0.9, 0.4]
    })
    
    resampled = resample_to_daily(df, agg_method='max')
    
    assert len(resampled) == 2  # 2 days
    # Day 1 max should be 0.9
    day1 = resampled[resampled["date"] == "2024-01-01"]
    assert day1["cpu"].iloc[0] == 0.9

# -----------------------------
# Test: detect_anomalies_simple
# -----------------------------

def test_detect_anomalies_simple():
    """Test Z-score anomaly detection."""
    # Create mostly normal data with one huge outlier
    data = np.random.normal(0.5, 0.01, 100)
    data[50] = 10.0  # Huge outlier
    
    df = pd.DataFrame({"val": data})
    outliers = detect_anomalies_simple(df, columns=["val"], threshold=3.0)
    
    assert outliers["is_outlier_val"].iloc[50] == True
    assert outliers["is_outlier_val"].sum() == 1

def test_detect_anomalies_zero_std():
    """Test proper handling of constant arrays (std=0)."""
    df = pd.DataFrame({"const": [1, 1, 1, 1]})
    outliers = detect_anomalies_simple(df, columns=["const"])
    assert not outliers["is_outlier_const"].any()

# -----------------------------
# Test: load_from_s3 (Mocked)
# -----------------------------

@patch("boto3.Session")
def test_load_from_s3_success(mock_session):
    """Test successful S3 load via mock."""
    # Setup Mock
    mock_s3 = MagicMock()
    mock_session.return_value.client.return_value = mock_s3
    
    # Mock S3 response body
    mock_body = MagicMock()
    # Simulate CSV content read
    mock_body.read.return_value = b"col1,col2\n1,2"
    # To support pd.read_csv(obj['Body']), we generally return a file-like object or bytes
    # Here, let's just patch pd.read_csv to simplify
    
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({"col1": [1], "col2": [2]})
        
        mock_s3.get_object.return_value = {"Body": mock_body}
        
        df = load_from_s3("bucket", "key.csv", region="us-west-2")
        
        assert not df.empty
        mock_s3.get_object.assert_called_with(Bucket="bucket", Key="key.csv")
        mock_session.assert_called_with(profile_name=None, region_name="us-west-2")

@patch("boto3.Session")
def test_load_from_s3_error(mock_session):
    """Test S3 client error handling."""
    mock_s3 = MagicMock()
    mock_session.return_value.client.return_value = mock_s3
    
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, 
        "GetObject"
    )
    
    with pytest.raises(ClientError):
        load_from_s3("bucket", "missing.csv")

# -----------------------------
# Test: get_date_range
# -----------------------------

def test_get_date_range():
    dr = get_date_range("2024-01-01", "2024-01-05")
    assert len(dr) == 5
    assert dr[0] == pd.Timestamp("2024-01-01")
    assert dr[-1] == pd.Timestamp("2024-01-05")

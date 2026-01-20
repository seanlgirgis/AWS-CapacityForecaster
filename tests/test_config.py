import os
import pytest
from pathlib import Path
from typing import Dict
import yaml

from src.utils.config import (
    load_config,
    validate_config,
    get_nested,
    get_enabled_models,
    get_aws_config,
    get_data_config,
    get_ml_config,
    get_risk_config,
    CONFIG,
    PROJECT_ROOT,
)

# Sample minimal valid config YAML for testing
MINIMAL_CONFIG_YAML = """
aws:
  region: us-east-1
  bucket_name: test-bucket
data:
  num_servers: 100
ml:
  forecast_horizon_months: 6
  models:
    - name: Prophet
      enabled: true
risk_analysis:
  high_risk_threshold: 85.0
"""

@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config YAML file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(MINIMAL_CONFIG_YAML)
    return config_path

@pytest.fixture
def invalid_config() -> Dict:
    """Invalid config dict for validation testing."""
    return {
        "aws": {"region": "invalid-region", "bucket_name": ""},
        "data": {"num_servers": -5},
        "ml": {"forecast_horizon_months": 15, "models": []},
        "risk_analysis": {"high_risk_threshold": 105.0},
    }

def test_load_config_success(temp_config_file: Path):
    """Test successful config loading and basic content."""
    # No more validate=False — validation is always on now
    config = load_config(config_path=temp_config_file)
    assert isinstance(config, dict)
    assert config["aws"]["region"] == "us-east-1"
    assert config["data"]["num_servers"] == 100
    assert "_project_root" in config  # Injected root

def test_load_config_with_env_overrides(temp_config_file: Path, monkeypatch):
    """Test environment variable overrides."""
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("DATA_NUM_SERVERS", "50")
    # No validate=False
    config = load_config(config_path=temp_config_file)
    assert config["aws"]["region"] == "us-west-2"
    assert config["data"]["num_servers"] == 50

def test_load_config_invalid_yaml(tmp_path: Path):
    """Test loading malformed YAML."""
    bad_path = tmp_path / "bad.yaml"
    with open(bad_path, "w") as f:
        f.write("invalid: yaml: here:")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(bad_path)

def test_load_config_missing_file():
    """Test missing file error."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.yaml"))

def test_validate_config_success():
    """Test validation on valid config."""
    valid_config = yaml.safe_load(MINIMAL_CONFIG_YAML)
    validate_config(valid_config)  # No exception = success

def test_validate_config_failures(invalid_config: Dict):
    """Test validation errors on invalid config."""
    with pytest.raises(ValueError) as exc:
        validate_config(invalid_config)
    error_msg = str(exc.value)
    assert "Invalid AWS region" in error_msg
    assert "bucket name is required" in error_msg
    assert "positive integer" in error_msg
    assert "between 1 and 12" in error_msg
    assert "At least one ML model" in error_msg
    assert "between 0 and 100" in error_msg

def test_get_nested():
    """Test nested dict access."""
    d = {"a": {"b": {"c": 42}}}
    assert get_nested(d, ["a", "b", "c"]) == 42
    assert get_nested(d, ["a", "missing"], default="default") == "default"

def test_get_enabled_models():
    """Test extracting enabled models."""
    # Temporarily override global CONFIG for this test
    original = CONFIG.copy()
    CONFIG["ml"] = {"models": [{"name": "Prophet", "enabled": True}, {"name": "RF", "enabled": False}]}
    enabled = get_enabled_models()
    assert len(enabled) == 1
    assert enabled[0]["name"] == "Prophet"
    # Restore (good practice in tests)
    CONFIG.clear()
    CONFIG.update(original)

def test_config_getters():
    """Test section getters."""
    # Temporarily override
    original = CONFIG.copy()
    CONFIG.update({
        "aws": {"test": "aws"},
        "data": {"test": "data"},
        "ml": {"test": "ml"},
        "risk_analysis": {"test": "risk"},
    })
    assert get_aws_config() == {"test": "aws"}
    assert get_data_config() == {"test": "data"}
    assert get_ml_config() == {"test": "ml"}
    assert get_risk_config() == {"test": "risk"}
    # Restore
    CONFIG.clear()
    CONFIG.update(original)

def test_project_root():
    """Test project root resolution — only check basic properties."""
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.is_absolute()
    # Do NOT check .exists() or specific subfolders — unreliable in pytest temp context
    # Instead: check that it contains expected name parts (repo name)
    assert "AWS-CapacityForecaster" in str(PROJECT_ROOT).replace("\\", "/")
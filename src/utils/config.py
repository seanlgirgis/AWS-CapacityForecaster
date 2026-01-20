"""
Configuration Management Module for AWS-CapacityForecaster.

Centralized, single source of truth for all configurable parameters.
Loads settings from config/config.yaml with environment variable overrides from .env.

Usage:
    from src.utils.config import CONFIG

    bucket = CONFIG['aws']['bucket_name']
    num_servers = CONFIG['data']['num_servers']
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Determine project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# Valid AWS regions for validation
VALID_AWS_REGIONS = {
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
    "ap-northeast-1", "ap-northeast-2", "ap-southeast-1", "ap-southeast-2",
    "ap-south-1", "sa-east-1", "ca-central-1",
}


def get_nested(d: Dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely access nested dictionary values using a list of keys.

    Args:
        d: Dictionary to access
        keys: List of keys for nested access, e.g., ['aws', 'bucket_name']
        default: Default value if key path doesn't exist

    Returns:
        Value at the nested key path, or default if not found

    Example:
        >>> config = {'aws': {'bucket_name': 'my-bucket'}}
        >>> get_nested(config, ['aws', 'bucket_name'])
        'my-bucket'
        >>> get_nested(config, ['aws', 'missing'], 'default')
        'default'
    """
    result = d
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config dictionary.

    Environment variables take precedence over YAML values.
    Mapping:
        AWS_REGION -> config['aws']['region']
        AWS_BUCKET_NAME -> config['aws']['bucket_name']
        AWS_SAGEMAKER_ROLE_ARN -> config['aws']['sagemaker_role_arn']
        AWS_PROFILE -> config['aws']['profile']
        DATA_NUM_SERVERS -> config['data']['num_servers']
        ML_FORECAST_HORIZON_MONTHS -> config['ml']['forecast_horizon_months']
        EXECUTION_MODE -> config['execution']['mode']
    """
    env_mappings = [
        ("AWS_REGION", ["aws", "region"], str),
        ("AWS_BUCKET_NAME", ["aws", "bucket_name"], str),
        ("AWS_SAGEMAKER_ROLE_ARN", ["aws", "sagemaker_role_arn"], str),
        ("AWS_PROFILE", ["aws", "profile"], str),
        ("DATA_NUM_SERVERS", ["data", "num_servers"], int),
        ("ML_FORECAST_HORIZON_MONTHS", ["ml", "forecast_horizon_months"], int),
        ("EXECUTION_MODE", ["execution", "mode"], str),
    ]

    for env_var, key_path, cast_type in env_mappings:
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate to parent and set the value
            target = config
            for key in key_path[:-1]:
                target = target.setdefault(key, {})
            try:
                target[key_path[-1]] = cast_type(value)
                logger.debug(f"Override from env: {env_var} -> {key_path}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid env override {env_var}={value}: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values. Raises ValueError if invalid.

    Checks:
        - Required keys exist
        - Numeric values are in valid ranges
        - AWS region is valid
        - At least one ML model is enabled
    """
    errors = []

    # Required string fields (must be non-empty)
    required_strings = [
        (["aws", "bucket_name"], "AWS bucket name"),
        (["aws", "region"], "AWS region"),
    ]
    for key_path, description in required_strings:
        value = get_nested(config, key_path, "")
        if not value or not isinstance(value, str):
            errors.append(f"{description} is required (config path: {'.'.join(key_path)})")

    # Validate AWS region
    region = get_nested(config, ["aws", "region"], "")
    if region and region not in VALID_AWS_REGIONS:
        errors.append(f"Invalid AWS region: {region}. Must be one of: {', '.join(sorted(VALID_AWS_REGIONS))}")

    # Numeric validations
    num_servers = get_nested(config, ["data", "num_servers"], 0)
    if not isinstance(num_servers, int) or num_servers <= 0:
        errors.append(f"data.num_servers must be a positive integer, got: {num_servers}")

    forecast_horizon = get_nested(config, ["ml", "forecast_horizon_months"], 0)
    if not isinstance(forecast_horizon, int) or not (1 <= forecast_horizon <= 12):
        errors.append(f"ml.forecast_horizon_months must be between 1 and 12, got: {forecast_horizon}")

    # At least one ML model must be enabled
    models = get_nested(config, ["ml", "models"], [])
    enabled_models = [m for m in models if m.get("enabled", False)]
    if not enabled_models:
        errors.append("At least one ML model must be enabled in ml.models")

    # Risk thresholds must be percentages
    high_risk = get_nested(config, ["risk_analysis", "high_risk_threshold"], 0)
    if not (0 < high_risk <= 100):
        errors.append(f"risk_analysis.high_risk_threshold must be between 0 and 100, got: {high_risk}")

    if errors:
        error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(error_msg)

    logger.info("Configuration validation passed")


def load_config(config_path: Optional[Path] = None, validate: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config YAML file (defaults to config/config.yaml)
        validate: Whether to run validation checks (default True)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
        yaml.YAMLError: If YAML is malformed
    """
    # Load .env file if it exists
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        logger.debug(f"Loaded environment from {ENV_PATH}")

    # Determine config path
    path = config_path or CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Please create {path} or copy from config/config.yaml.example"
        )

    # Load YAML
    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if config is None:
        raise ValueError(f"Configuration file {path} is empty")

    logger.info(f"Loaded configuration from {path}")

    # Apply environment overrides
    config = _apply_env_overrides(config)

    # Inject project root for convenience
    config["_project_root"] = str(PROJECT_ROOT)

    # Validate
    if validate:
        validate_config(config)

    return config


# Load configuration on module import
# This makes CONFIG available immediately: from src.utils.config import CONFIG
try:
    CONFIG: Dict[str, Any] = load_config()
except Exception as e:
    # Log error but don't crash on import - allows tests to mock CONFIG
    logger.error(f"Failed to load configuration: {e}")
    CONFIG = {}


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_enabled_models() -> List[Dict[str, Any]]:
    """Return list of enabled ML model configurations."""
    models = get_nested(CONFIG, ["ml", "models"], [])
    return [m for m in models if m.get("enabled", False)]


def get_aws_config() -> Dict[str, Any]:
    """Return AWS-specific configuration section."""
    return CONFIG.get("aws", {})


def get_data_config() -> Dict[str, Any]:
    """Return data generation configuration section."""
    return CONFIG.get("data", {})


def get_ml_config() -> Dict[str, Any]:
    """Return ML/forecasting configuration section."""
    return CONFIG.get("ml", {})


def get_risk_config() -> Dict[str, Any]:
    """Return risk analysis configuration section."""
    return CONFIG.get("risk_analysis", {})

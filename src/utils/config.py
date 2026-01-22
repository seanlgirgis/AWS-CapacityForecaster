"""
Centralized configuration loader for AWS-CapacityForecaster project.

Loads settings from config/config.yaml, applies optional .env overrides,
performs strict validation, and provides convenient getter functions.

Supports seamless switching between local and AWS execution modes.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# -----------------------------
# Setup logging
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Assumes structure: src/utils/config.py → 2 levels up
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

VALID_AWS_REGIONS = {
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1",
    # Add more as needed — static list avoids boto3 dependency here
}

# -----------------------------
# Global config holder
# -----------------------------
CONFIG: Dict[str, Any] = {}


def get_nested(d: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely retrieve nested dictionary value with dot-like access.

    Args:
        d: Dictionary to search
        keys: List of keys in order (e.g., ['aws', 'region'])
        default: Value to return if path not found

    Returns:
        Nested value or default
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load and merge configuration from YAML + .env overrides.

    1. Loads base YAML config
    2. Loads .env file if exists
    3. Applies environment variable overrides (prefixed patterns)
    4. Injects project root
    5. Validates the final config

    Args:
        config_path: Path to config.yaml (defaults to DEFAULT_CONFIG_PATH)
        env_path: Path to .env file (defaults to PROJECT_ROOT/.env)

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file missing
        ValueError: On invalid YAML or validation failure
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    env_path = env_path or ENV_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base YAML
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    # Load .env if exists
    if env_path.exists():
        logger.info(f"Loading environment overrides from {env_path}")
        load_dotenv(env_path, override=True)

    # Apply env overrides using prefix convention (e.g., AWS_REGION → config['aws']['region'])
    _apply_env_overrides(config)

    # Inject project root for convenience
    config["_project_root"] = str(PROJECT_ROOT)

    # Validate everything
    validate_config(config)

    global CONFIG
    CONFIG.clear()
    CONFIG.update(config)
    logger.info("Configuration loaded and validated successfully")
    return config

def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Apply environment variable overrides to config dict with type conversion."""
    overrides_map = {
        "AWS_REGION": (["aws", "region"], str),
        "AWS_PROFILE": (["aws", "profile"], str),
        "AWS_BUCKET_NAME": (["aws", "bucket_name"], str),
        "AWS_SAGEMAKER_ROLE_ARN": (["aws", "sagemaker_role_arn"], str),
        "DATA_NUM_SERVERS": (["data", "num_servers"], int),
        "ML_FORECAST_HORIZON_MONTHS": (["ml", "forecast_horizon_months"], int),
        "EXECUTION_MODE": (["execution", "mode"], str),
        # Add more as your config grows
    }

    for env_key, (path, expected_type) in overrides_map.items():
        if raw_value := os.getenv(env_key):
            try:
                value = expected_type(raw_value)
                current = config
                for key in path[:-1]:
                    current = current.setdefault(key, {})
                current[path[-1]] = value
                logger.debug(f"Applied override: {env_key} = {value} ({type(value).__name__})")
            except ValueError as e:
                raise ValueError(
                    f"Invalid type for env var {env_key}: expected {expected_type.__name__}, got '{raw_value}'"
                ) from e

def validate_config(config: Dict[str, Any]) -> None:
    """Strict validation of the configuration structure and values.

    Raises ValueError with detailed messages on any failure.
    """
    errors = []

    # ------------------- AWS Section -------------------
    aws = get_nested(config, ["aws"], {})
    if not aws:
        errors.append("Missing 'aws' section in config")

    region = aws.get("region")
    if region not in VALID_AWS_REGIONS:
        errors.append(f"Invalid AWS region: '{region}'. Must be one of: {VALID_AWS_REGIONS}")

    if not aws.get("bucket_name"):
        errors.append("AWS bucket name is required")

    # Optional but if present, basic ARN check
    role_arn = aws.get("sagemaker_role_arn")
    if role_arn and not role_arn.startswith("arn:aws:iam::"):
        errors.append("SageMaker role ARN appears invalid (must start with arn:aws:iam::)")

    # ------------------- Data Section -------------------
    data = get_nested(config, ["data"], {})
    num_servers = data.get("num_servers")
    if not isinstance(num_servers, int) or num_servers <= 0:
        errors.append("data.num_servers must be a positive integer")

    # ------------------- ML Section -------------------
    ml = get_nested(config, ["ml"], {})
    horizon = ml.get("forecast_horizon_months")
    if not isinstance(horizon, int) or not (1 <= horizon <= 12):
        errors.append("ml.forecast_horizon_months must be an integer between 1 and 12")

    models = ml.get("models", [])
    if not isinstance(models, list) or not models:
        errors.append("At least one ML model must be defined in ml.models")
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"ml.models[{i}] must be a dictionary")
            continue
        if "name" not in model or not isinstance(model["name"], str):
            errors.append(f"ml.models[{i}]['name'] must be a non-empty string")
        if "enabled" not in model or not isinstance(model["enabled"], bool):
            errors.append(f"ml.models[{i}]['enabled'] must be boolean (true/false)")

    # ------------------- Risk Analysis -------------------
    risk = get_nested(config, ["risk_analysis"], {})
    threshold = risk.get("high_risk_threshold")
    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 100):
        errors.append("risk_analysis.high_risk_threshold must be a number between 0 and 100")

    # ------------------- Local Mode Path Checks -------------------
    if get_nested(config, ["execution", "mode"], "local") == "local":
        for key_path, expected_type in [
            (["data", "raw_data_path"], (str, Path)),
            (["data", "processed_data_path"], (str, Path)),
        ]:
            path_val = get_nested(config, key_path)
            if path_val and not isinstance(path_val, (str, Path)):
                errors.append(f"{'.'.join(key_path)} must be a string or Path")
            # Optional: check existence (commented out for flexibility during dev)
            # if path_val and not Path(path_val).exists():
            #     errors.append(f"Path does not exist in local mode: {path_val}")

    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))


# -----------------------------
# Convenience getters (use these in other modules)
# -----------------------------
def get_aws_config() -> Dict[str, Any]:
    return get_nested(CONFIG, ["aws"], {})


def get_data_config() -> Dict[str, Any]:
    return get_nested(CONFIG, ["data"], {})


def get_ml_config() -> Dict[str, Any]:
    return get_nested(CONFIG, ["ml"], {})


def get_risk_config() -> Dict[str, Any]:
    return get_nested(CONFIG, ["risk_analysis"], {})


def get_execution_mode() -> str:
    return get_nested(CONFIG, ["execution", "mode"], "local")


def get_enabled_models() -> List[Dict[str, Any]]:
    """Return list of enabled ML models from config."""
    models = get_ml_config().get("models", [])
    return [m for m in models if m.get("enabled", False)]


# Auto-load on module import (optional — can be explicit in main scripts)
try:
    load_config()
except Exception as e:
    logger.warning(f"Auto-loading config failed (will load later explicitly): {e}")
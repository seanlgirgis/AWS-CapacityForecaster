"""
Demo app for config.py usage.

Loads config and prints key sections for verification.
Can be extended to kick off data generation or ML training based on config.
"""

import logging
from src.utils.config import (
    CONFIG,
    get_aws_config,
    get_data_config,
    get_ml_config,
    get_enabled_models,
    get_risk_config,
)

# Basic logging setup (from config if desired)
logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Loaded CONFIG overview:")
    print(f"Project Root: {CONFIG['_project_root']}")
    print(f"AWS Config: {get_aws_config()}")
    print(f"Data Config: {get_data_config()}")
    print(f"ML Config: {get_ml_config()}")
    print(f"Enabled Models: {[m['name'] for m in get_enabled_models()]}")
    print(f"Risk Config: {get_risk_config()}")

    # Example usage: Simulate data gen params from config
    num_servers = get_data_config()["num_servers"]
    forecast_horizon = get_ml_config()["forecast_horizon_months"]
    print(f"\nDemo: Generating data for {num_servers} servers, forecasting {forecast_horizon} months ahead.")

    # Extend here: e.g., call data_generation.py with these params

if __name__ == "__main__":
    main()
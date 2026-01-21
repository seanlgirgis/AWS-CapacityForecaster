"""
# Run command: python demos/demo_config.py
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
    print("\n" + "="*60)
    print(" 🛠️  PROJECT CONFIGURATION OVERVIEW")
    print("="*60)

    print(f"\n📂 [Project Root]: {CONFIG['_project_root']}")

    import json
    def pretty_print(title, data):
        print(f"\n🔹 [{title}]:")
        print(json.dumps(data, indent=2, default=str))

    pretty_print("AWS Config", get_aws_config())
    pretty_print("Data Config", get_data_config())
    pretty_print("ML Config", get_ml_config())
    
    print(f"\n✅ [Enabled Models]: {', '.join([m['name'] for m in get_enabled_models()])}")
    
    pretty_print("Risk Config", get_risk_config())
    print("\n" + "-"*60)

    # Example usage: Simulate data gen params from config
    num_servers = get_data_config()["num_servers"]
    forecast_horizon = get_ml_config()["forecast_horizon_months"]
    print(f"\nDemo: Generating data for {num_servers} servers, forecasting {forecast_horizon} months ahead.")

    # Extend here: e.g., call data_generation.py with these params

if __name__ == "__main__":
    main()
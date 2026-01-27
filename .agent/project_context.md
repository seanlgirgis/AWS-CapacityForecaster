# Project Context & Architecture

## Overview
**AWS-CapacityForecaster** is a cloud-native, enterprise-grade tool for simulating, analyzing, and forecasting server capacity metrics. It models "Citi-style" enterprise environments with seasonality, trends, and risk factors.

## Repository Info
- **URL**: https://github.com/seanlgirgis/AWS-CapacityForecaster
- **Owner**: seanlgirgis@gmail.com

## Architecture
The project is structured into **Modules** (process steps) and **Utils** (shared capabilities).

### Core Pipeline (`src/modules/`)
1.  **Module 01: Data Generation** (`module_01_data_generation.py`)
    -   Generates synthetic server metrics (CPU, Memory, IO) with realistic constraints (holidays, business hours, seasonality).
2.  **Module 02: Data Load** (`module_02_data_load.py`)
    -   Handles loading of raw data, potentially from local sources or AWS S3.
3.  **Module 03: ETL & Feature Engineering** (`module_03_etl_feature_eng.py`)
    -   Cleans data, handles missing values, and generates ML features (lags, rolling averages, fiscal dates).
4.  **Module 04: Model Training** (`module_04_model_training.py`)
    -   Trains forecasting models (e.g., XGBoost, Linear Regression) on the processed data.

### Utilities (`src/utils/`)
-   **`config.py`**: Centralized configuration via `config/config.yaml` and env vars.
-   **`aws_utils.py`**: Boto3 wrappers for S3 and other AWS services.
-   **`data_utils.py`**: Data manipulation, validation, and generation helpers.
-   **`ml_utils.py`**: Machine learning helpers (metrics, splitting, evaluation).
-   **`server_archetypes.py`**: Definitions of different server types for simulation.
-   **`logging_config.py`**: Standardized logging setup.

### Development Environment
-   **Setup**: Powershell script `env_setter.ps1`.
-   **Virtual Env**: `c:\py_venv\AWS-CapacityForecaster`
-   **IDE**: VS Code configured via `.vscode`.
-   **Documentation**: External docs in `$env:KB_INBOX_PATH` (`C:\KB\00_Inbox`).

### Validation
-   **Tests**: `tests/` cover utils and core logic.
-   **Demos**: `demos/` provide runnable examples of utility usage.

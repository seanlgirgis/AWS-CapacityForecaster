# AWS-CapacityForecaster

A cloud-native, enterprise-grade capacity forecasting and optimization tool inspired by large-scale banking systems.

- **GitHub**: [https://github.com/seanlgirgis/AWS-CapacityForecaster](https://github.com/seanlgirgis/AWS-CapacityForecaster)
- **Contact**: seanlgirgis@gmail.com

## Overview

This project simulates and analyzes server capacity metrics (CPU, Memory, Disk, Network) to forecast future resource needs and identify risks. It is designed to run on AWS (SageMaker, Lambda, Athena) but includes a robust local development environment.

## Key Components

### 1. The Pipeline (`src/modules/`)
The core logic is divided into sequential modules:
-   **Module 01 - Data Generation**: Creates realistic "Citi-style" enterprise monitoring data (seasonality, trends, holidays) for 120+ servers (2022–2025).
-   **Module 02 - Data Load**: Ingests raw metrics coverage for downstream processing.
-   **Module 03 - ETL & Feature Engineering**: Cleans data, imputes missing values, and generates ML-ready features (lags, rolling stats, fiscal calendar flags).
-   **Module 04 - Model Training**: Trains and evaluates predictive models to forecast capacity usage.

### 2. Utilities (`src/utils/`)
-   **Configuration**: Centralized management (`config.py`) using YAML and environment variables.
-   **Data Utils**: robust tools for data generation, validation, and manipulation (`data_utils.py`).
-   **AWS Integration**: S3 and cloud resource management wrappers (`aws_utils.py`).
-   **ML Tools**: Helpers for metrics, splitting strategies, and evaluation (`ml_utils.py`).

## Getting Started

### Prerequisites
-   Python 3.12+
-   Windows (Powershell)

### Setup
1.  **Initialize Environment**:
    Use the provided setup script to activate the virtual environment and set necessary path variables.
    ```powershell
    . .\env_setter.ps1
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

### Running the Code
-   **Run a specific module**:
    ```powershell
    python -m src.modules.module_01_data_generation --env local
    ```
-   **Run Demos**:
    Check the `demos/` directory for example usages of the utility libraries.
    ```powershell
    python demos/demo_config.py
    ```

### Testing
Run the test suite to verify system integrity:
```powershell
pytest tests/
```

## Project Structure
```text
C:\pyproj\AWS-CapacityForecaster\
├── .agent/             # AI Context & Rules
├── .vscode/            # Editor Configuration
├── config/             # YAML configurations
├── demos/              # Usage examples
├── src/
│   ├── modules/        # Core execution pipeline (01-04)
│   └── utils/          # Shared libraries (AWS, Data, XML, Config)
├── tests/              # Unit tests
└── env_setter.ps1      # Environment activation script
```

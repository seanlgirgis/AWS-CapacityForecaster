# AWS-CapacityForecaster

A cloud-native, enterprise-grade capacity forecasting and optimization tool inspired by large-scale banking systems.

## Overview

This project simulates and analyzes server capacity metrics (CPU, Memory, Disk, Network) to forecast future resource needs and identify risks. It is designed to run on AWS (SageMaker, Lambda, Athena) but includes a robust local development environment.

## Key Components

### 1. Configuration (`src/utils/config.py`)
- Centralized configuration management using `config/config.yaml` and `.env` files.
- Supports environment variable overrides for flexible deployment across Local, SageMaker, and Lambda environments.
- Strict validation ensures data integrity and fail-fast behavior.

### 2. Data Utilities (`src/utils/data_utils.py`)
- **Synthetic Data Generation**: Creates realistic "Citi-style" enterprise monitoring data, including seasonality, trends, and banking holidays.
- **S3 Integration**: Seamlessly loads data from AWS S3 using `boto3`.
- **Quality Checks**: Validates data against enterprise standards (missing values, outliers, negative utilization).
- **Feature Engineering**: Adds calendar-based features (EOQ, holidays) for ML models.

## Getting Started

1.  **Environment Setup**:
    ```powershell
    # Install dependencies
    pip install -r requirements.txt
    
    # Set up environment variables (windows)
    .\env_setter.ps1
    ```

2.  **Run Tests**:
    ```powershell
    pytest tests/
    ```

3.  **Run Demo**:
    ```powershell
    python demos/demo_config.py
    ```

## Project Structure
- `src/`: Source code (utils, models, pipelines).
- `config/`: Configuration files.
- `docs/`: Detailed design documentation.
- `tests/`: Unit tests.
- `demos/`: Example scripts.

# Amazon Forecast Integration Details

## Overview
We have enhanced **Module 04 (Model Training)** to include **Amazon Forecast** as a managed AutoML comparator. This allows the pipeline to benchmark custom machine learning models (Prophet, Random Forest) against AWS's fully managed, deep-learning-based forecasting service.

## Architecture Change
1.  **Hybrid Training**: The pipeline now runs local models (Prophet/RF) on the SageMaker instance CPU *while simultaneously* orchestrating a remote Amazon Forecast job via boto3.
2.  **State Management**: The new `AmazonForecastRunner` class handles the complex state transitions of the Forecast service (Dataset Group -> Import Job -> Predictor Training -> Forecast Generation).
3.  **Unified Metrics**: Predictions from Amazon Forecast are retrieved and merged into the standard `all_model_forecasts.parquet` and `model_comparison.json`, allowing for direct "apples-to-apples" sMAPE comparison.

## Implementation Details

### 1. New Utility: `src/utils/amazon_forecast.py`
Created a robust helper class `AmazonForecastRunner` that encapsulates the AWS Forecast API complexity:
- **`prepare_and_upload_data`**: Converts our internal DataFrame to the required CSV schema (item_id, timestamp, target_value) and uploads it to S3.
- **`run_full_cycle`**: managing the dependent resource creation chain:
    - **Dataset Group**: `capacity_forecaster_dsg`
    - **Dataset Import**: Ingests the uploaded CSV.
    - **AutoPredictor**: Launches an AutoML training job (tries ARIMA, CNN-QR, DeepAR+, etc. automatically).
    - **Forecast Generation**: Creates inference points.
- **`query_forecasts`**: Uses the `forecastquery` client to retrieve P10, P50 (median), and P90 predictions for every server.

### 2. Module 04 Refactoring (`src/modules/module_04_model_training.py`)
- **Integration Point**: Added a new block after the existing model training loop.
- **Logic**:
    - Checks `config['ml']['models']['AmazonForecast']['enabled']`.
    - If true, instantiates the runner.
    - Uploads the **Global Train Split** (data < test_split_date).
    - Waits for the remote Forecast job to complete (synchronous blocking).
    - Merges the returned P50 predictions as the "predicted" value.
    - Calculates sMAPE/MAE/RMSE against the hold-out test set just like local models.

### 3. Configuration Updates (`config/config.yaml`)
Added the `AmazonForecast` model definition:
```yaml
    - name: AmazonForecast
      enabled: true
      params:
        dataset_group_name: "capacity_forecaster_dsg"
        forecast_frequency: "D"
        horizon: 90
        timestamp_format: "yyyy-MM-dd"
```

## How to Verify
1.  **Usage**: Run `python -m src.modules.module_00_pipeline_runner --env sagemaker --only 04`.
2.  **Monitoring**: The orchestrator logs will show "Creating Predictor..." and pause while Amazon Forecast trains.
3.  **Output**: `metrics/model_comparison.json` will now include an entry for `AmazonForecast` with its accuracy metrics.

## Value Add
This integration demonstrates advanced cloud-native design:
- **Managed Service Orchestration**: Controlling one AWS service (Forecast) from another (SageMaker).
- **AutoML Benchmarking**: Validating if "black box" deep learning beats custom feature engineering.

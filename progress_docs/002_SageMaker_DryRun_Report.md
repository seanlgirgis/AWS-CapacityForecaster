# 002 - SageMaker Dry Run & Deployment Report

**Date:** 2026-01-31
**Status:** Completed
**Focus:** moving from local execution to SageMaker Processing Jobs

---

## 1. Executive Summary: The Last Step
This batch of development focused on the critical transition from local testing to cloud execution. We successfully implemented a "Dry Run" capability for the `AWS-CapacityForecaster` pipeline, allowing it to submit Processing Jobs to Amazon SageMaker. This marks the bridge between local development and scalable cloud operations.

## 2. The Dry Run
We executed a dry run of **Module 04 (Model Training)** targeting the SageMaker environment.
-   **Goal:** Verify that our local runner (`module_00_pipeline_runner.py`) could successfully package code, authenticate, and submit a job to AWS.
-   **Method:** We integrated a new execution mode (`--env sagemaker`) that intercepts standard module calls and redirects them to the SageMaker API.

## 3. Challenges & Solutions

### Problem 1: `ModuleNotFoundError` & Dependency Hell (The "Sage" Issue)
**The Issue:**
We initially attempted to use the high-level `sagemaker` Python SDK (e.g., `from sagemaker.sklearn.processing import SKLearnProcessor`). However, our local environment had conflicting versions of core libraries (`pandas`, `numpy`) compared to what the `sagemaker` SDK required, leading to persistent `ModuleNotFoundError` and version conflicts even when simply trying to import the library.

**The Solution:**
We bypassed the high-level SDK entirely.
-   **Adopted `boto3`:** We rewrote the submission logic to use the lower-level `boto3` client (`boto3.client('sagemaker')`).
-   **Why this worked:** `boto3` is lightweight and comes pre-installed in almost all AWS environments without the heavy dependency chain of the full SageMaker SDK. It allowed us to construct the JSON API payload manually (defining inputs/outputs/containers) without breaking our local Python environment.

### Problem 2: Path Mapping (S3 vs. Container)
**The Issue:**
Code running locally sees files in `C:\pyproj\...` or `./data`. Code running in SageMaker sees files in `/opt/ml/processing/input`.
**The Solution:**
-   Standardized `input_data_configs` to map S3 buckets to `/opt/ml/processing/input/data`.
-   Standardized `output_data_configs` to map `/opt/ml/processing/output` back to S3.
-   Updated `module_00_pipeline_runner.py` to pass these paths dynamically via command-line arguments to the container.

## 4. Code Changes
We made significant changes to the orchestration layer to support this hybrid (Local + Cloud) workflow.

### Key Modifications:
1.  **`src/modules/module_00_pipeline_runner.py`**:
    -   Added `run_sagemaker_processing_job()` function.
    -   Added logic to check for `config['execution']['mode'] == 'sagemaker'`.
    -   Implemented the `boto3` job submission logic directly in the runner (or bridged to the launcher).

2.  **`src/utils/sagemaker_launcher.py`** (New/Refined):
    -   Created a dedicated utility to handle the verbose `boto3` `create_processing_job` payload construction.
    -   Abstracted away the JSON structure for `ProcessingInputs` and `ProcessingOutputConfig`.

3.  **`src/modules/module_04_model_training.py`**:
    -   Refactored input path handling to respect CLI arguments for data directories (crucial for container file systems).

4.  **`src/utils/amazon_forecast.py`**:
    -   Added capability for low-code forecasting using Amazon Forecast (via boto3), broadening our ML toolset beyond custom Scikit-Learn models.

## 5. Command Log
The following commands were used during this development batch:

**To trigger the SageMaker Dry Run:**
```powershell
python src/module_00_pipeline_runner.py --env sagemaker --only 04
```

**To verify Git status and history:**
```powershell
git log -n 15 --stat
git status
```

## 6. Artifact Inventory
**Modified Files:**
-   `src/modules/module_00_pipeline_runner.py` (Orchestrator)
-   `src/modules/module_04_model_training.py` (Training Logic)
-   `src/utils/config.py` (Configuration Loader)
-   `src/modules/module_feature_eng.py` (Feature Engineering - *referenced in logs*)

**Created Files:**
-   `src/utils/sagemaker_launcher.py` (The Boto3 Bridge)
-   `src/utils/amazon_forecast.py` (New Utility)
-   `progress_docs/002_SageMaker_DryRun_Report.md` (This Report)

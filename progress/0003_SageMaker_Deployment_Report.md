# SageMaker Deployment & AutoGluon Integration Report

**Date:** January 31, 2026
**Status:** ✅ Success
**Job ID:** `proc-job-1769883852`

## 1. Executive Summary

We have successfully deployed the `AWS-CapacityForecaster` pipeline (Module 04) to AWS SageMaker. The deployment now runs the full forecasting suite, including **Prophet**, **Random Forest**, and **AutoGluon** (AutoML) on 120 simulated servers. 

The process involved resolving significant challenges related to the SageMaker Python SDK, resource constraints, and complex dependency conflicts between modern Pandas/PyArrow and the legacy SageMaker Scikit-Learn container.

---

## 2. Work Accomplished

### A. Local Development & Logic Repairs
Before deployment, we ensured the core logic was sound:
- **Logic Fix**: Fixed a `NameError` in `module_04_inner.py` where aggregate metrics (`avg_smape`, `best_model`) were referenced before calculation.
- **Verification**: Ran a full local training cycle on 120 servers to validate stability and memory usage.

### B. Cloud Architecture Updates
To maximize reliability and control:
- **Refactoring Runner**: We rewrote `module_00_pipeline_runner.py` to remove the dependency on the high-level `sagemaker` SDK (which was causing `ModuleNotFoundError` in the custom environment).
- **Direct Boto3 Integration**: The pipeline now uses `boto3` directly to interact with SageMaker APIs, offering granular control over the `CreateProcessingJob` payload.

---

## 3. Challenges & Solutions

We encountered three main hurdles during the deployment phase.

### Challenge 1: SageMaker SDK Import Errors
- **Symptom**: `ModuleNotFoundError: No module named 'sagemaker.processing'` despite the package being installed.
- **Root Cause**: Environment path issues or conflicts in the local execution context.
- **Solution**: **Bypassed the SDK**. refactored the pipeline runner to use the underlying `boto3` client. This proved more robust and removed the need for the heavy `sagemaker` library in the runtime environment.

### Challenge 2: Silent Failures & OOM
- **Symptom**: Job failed with generic `AlgorithmError` and no clear logs.
- **Root Cause**: The default `ml.t3.medium` instance (4GB RAM) ran out of memory during the installation of heavy packages like `autogluon` and `torch`.
- **Solution**: 
    1. Upgraded instance type to **`ml.t3.large`** (8GB RAM).
    2. Implemented **Fail-Fast Logic** in the wrapper script: any installation error now immediately crashes the container with an exit code, preventing "zombie" runs.
    3. Enabled **Verbose Logging** for `pip install` to see exactly where it hangs.

### Challenge 3: Dependency Hell (PyArrow vs. Container)
- **Symptom**: `ImportError: Missing optional dependency 'pyarrow'` and `fastparquet`.
- **Root Cause**: `pip` automatically installed the latest `pyarrow==21.0.0`, which is incompatible with the older `pandas`/`numpy` available in the container's base image. `pandas.read_parquet()` failed to load the engine.
- **Solution**:
    1. **Strict Pinning**: Pinned `pyarrow==14.0.1` and `fastparquet==2023.10.1` in `module_04_model_training.py`.
    2. **Explicit Engine Usage**: Updated `module_04_inner.py` to call `pd.read_parquet(..., engine='pyarrow')` explicitely.
    3. **Pre-flight Check**: Added a verification block that attempts to import all critical libraries *before* starting the main workload.

---

## 4. Successful Deployment Logs

**Job**: `proc-job-1769883852`

```text
Include logs here...
Wrapper: Installing runtime dependencies...
Wrapper: Installing packages: ['prophet', ..., 'pyarrow==14.0.1', 'fastparquet==2023.10.1', 'autogluon>=1.1.0']
...
Wrapper: Dependencies installed successfully.
Wrapper: Verifying imports...
Imports successful
Wrapper: Launching inner script: /opt/ml/processing/input/project_root/src/modules/module_04_inner.py
...
INFO:__main__:=== Scalability Mode: Running Amazon Forecast Comparator (AutoML) ===
INFO:__main__:Training on 120 servers...
```

---

## 5. Code Changes

| File | Change Summary |
| :--- | :--- |
| **`src/modules/module_04_inner.py`** | • Fixed `metrics_summary` logic error.<br>• Added `engine='pyarrow'` to all parquet read/write calls.<br>• Removed unused variables. |
| **`src/modules/module_04_model_training.py`** | • Added Strict pinning for `pyarrow`, `fastparquet`, `pandas`.<br>• Added `verify_cmd` to check imports before running.<br>• Replaced quiet install with verbose logging. |
| **`src/modules/module_00_pipeline_runner.py`** | • Refactored `run_sagemaker_processing_job` to use `boto3`.<br>• Fixed logic to respect module-specific `instance_type` overrides. |
| **`config/config.yaml`** | • Upgraded module 04 instance type from `ml.t3.medium` to `ml.t3.large`. |
| **`fetch_logs.py`** | • **[NEW]** Utility script to safely fetch CloudWatch logs, bypassing Windows terminal encoding issues. |

---

## 6. Project Files Manifest

**Modified/Created Files:**
- `c:\pyproj\AWS-CapacityForecaster\src\modules\module_04_inner.py`
- `c:\pyproj\AWS-CapacityForecaster\src\modules\module_04_model_training.py`
- `c:\pyproj\AWS-CapacityForecaster\src\modules\module_00_pipeline_runner.py`
- `c:\pyproj\AWS-CapacityForecaster\config\config.yaml`
- `c:\pyproj\AWS-CapacityForecaster\fetch_logs.py`
- `c:\pyproj\AWS-CapacityForecaster\progress\0003_SageMaker_Deployment_Report.md` (This file)

**Referenced Files (Read-Only):**
- `c:\pyproj\AWS-CapacityForecaster\src\utils\sagemaker_launcher.py`
- `c:\pyproj\AWS-CapacityForecaster\task.md`

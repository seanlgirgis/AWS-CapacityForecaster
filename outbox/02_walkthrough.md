# Walkthrough - SageMaker Integration

## Summary
Successfully migrated Module 04 (Forecasting) and Module 05 (Risk Analysis) to run as scalable SageMaker Processing Jobs.

## Key Challenges & Solutions

### 1. "Silent Success" (Missing Files in S3)
**Problem**: SageMaker jobs reported "Success", but S3 locations were empty.
**Root Cause**: 
- Using `boto3` inside the container failed silently due to permission/path issues.
- The orchestrator script exited immediately without waiting for the job to finish (and thus didn't confirm uploads).
**Fix**: 
- **Code**: Updated modules to write to `/opt/ml/processing/output` (local mount). SageMaker automatically uploads this folder to S3 at job completion.
- **Orchestrator**: Added a `while True` polling loop to `module_00` to wait for job completion.

### 2. Dependency Hell (Binary Incompatibility)
**Problem**: `numpy`, `pandas`, and `pyarrow` versions clashed between the SageMaker container and our installed libraries.
**Fix**:
- **Downgrade**: Switched SageMaker image to `sagemaker-scikit-learn:0.23-1-cpu-py3` (native Pandas 1.x support).
- **Runtime Pining**: 
  - Module 05: Pins `pandas==1.1.3`, `numpy<1.24`, `matplotlib<3.6`.
  - Module 04: Installs `pyarrow>=3.0.0` at runtime to fix `ArrowTypeError`.
  - Config: Removed Python 3.8+ walrus operators (`:=`) for compatibility.

## Final Results
- **Module 04**: Trained Prophet models on SageMaker (`proc-job-1769654698`). Artifacts verified in `s3://.../forecasts/`.
- **Module 05**: Performed Risk Analysis. Artifacts verified in `s3://.../risk_analysis/` (`proc-job-1769655520`).

## Usage
Run on SageMaker:
```bash
python -m src.modules.module_00_pipeline_runner --env sagemaker
```
(This will now launch jobs, print ARNs, **wait** for completion, and verify success).

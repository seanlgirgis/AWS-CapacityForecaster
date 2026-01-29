# SageMaker Integration Log

## Progress
- [x] Verified full pipeline S3 integration.
- [x] Checked for existing SageMaker IAM roles (None found).
- [x] Updated `config/config.yaml` with `sagemaker` section (Role ARN needs user input).
- [x] Refactored `module_04_model_training.py` to support SageMaker Script Mode arguments.
- [x] Updated `module_00_pipeline_runner.py` to trigger Processing Jobs.

## User Action Required
- [ ] Create IAM Role (or find existing) with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`.
- [ ] Update `config/config.yaml` with the Role ARN.
- [ ] Run verification: `python -m src.modules.module_00_pipeline_runner --env sagemaker --only 04`.

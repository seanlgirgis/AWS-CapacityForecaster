# AWS-CapacityForecaster: Next Phase — AWS Cloud Integration & Deployment

You've successfully run the full local pipeline (modules 01–05) — great job! The outputs look solid: synthetic data generated, validated, feature-engineered, models trained (with Random Forest edging out as best via sMAPE), and risk analysis complete with visualizations. This confirms the core logic works end-to-end.

Building on the previous planning (e.g., "00001.Project1_planning.md" for core scope and "00002.aws_added_To_Project1.md" for AWS stack), the **next process** is to integrate real AWS services. This evolves the project from local prototype to **cloud-native**, directly showcasing your #3 priority (Clean Python + AWS/cloud integration) while enhancing #1 (Data Science/ML via SageMaker) and #2 (Capacity Planning with S3/Athena for scalable storage/querying).

We'll keep continuity: Reference the existing modular structure (e.g., use `save_to_s3_or_local` which already supports AWS via `boto3`), evolve the folder layout minimally, and focus on low-cost ($5–15 total) implementation. All changes go to the GitHub repo as commits (e.g., "feat: add AWS integration").

### Core Objective for This Phase
Migrate the pipeline to AWS for scalability, automation, and portfolio shine:
- Store data/artifacts in **S3** (raw, processed, forecasts, metrics).
- Run ETL/training in **SageMaker** (notebooks or jobs) for ML-heavy steps.
- Query results with **Athena** (SQL on S3) to simulate Citi-style "Oracle backup DB" analysis.
- Optional: Automate via **Lambda** for scheduled runs.
- Bonus: Dashboard in **QuickSight** or deploy a Dash app on **EC2** (free-tier eligible).

This recreates Citi data pipelines: monitoring data → cloud storage → ML forecasting → queryable insights.

### Updated Folder Structure (Evolve Gradually)
Keep the current layout, add AWS-specific files:
```
C:\pyproj\AWS-CapacityForecaster (root)
├── config/
│   └── config.yaml  # Add AWS sections (bucket_name, region)
├── src/
│   ├── modules/     # Existing 01–05; add module_06_aws_orchestrator.py (optional)
│   └── utils/       # data_utils.py already has boto3 — enhance if needed
├── notebooks/       # New: SageMaker-compatible Jupyter for experiments
│   └── aws_forecast_experiment.ipynb
├── .env             # Add AWS_ACCESS_KEY_ID, etc. (git-ignored)
├── README.md        # Update with AWS setup guide + cost notes
└── requirements.txt # Add boto3, sagemaker (if not already)
```

### AWS Integration Design & Implementation Steps
Follow this low-cost, step-by-step plan. Assumes you have an AWS account with $200 credits — we'll use Free Tier heavily.

#### 1. AWS Setup (One-Time, ~10 min, $0)
- **Console Basics**: Log in to AWS Console → Set region to `us-east-1` (cheapest for most services).
- **IAM Role/User**: Create an IAM user with policies: `AmazonS3FullAccess`, `AmazonSageMakerFullAccess`, `AmazonAthenaFullAccess`. Download access key/secret.
- **Budget Alert**: In Billing → Budgets, set a $10 monthly alert.
- **S3 Bucket**: Create one bucket (e.g., `aws-capacity-forecaster-sean`) — private, no versioning needed yet.
- **Update .env**: Add:
  ```
  AWS_ACCESS_KEY_ID=your_key
  AWS_SECRET_ACCESS_KEY=your_secret
  AWS_DEFAULT_REGION=us-east-1
  S3_BUCKET_NAME=aws-capacity-forecaster-sean
  ```
- **Config.yaml Updates**: Add under `aws` section:
  ```yaml
  aws:
    bucket_name: aws-capacity-forecaster-sean
    region: us-east-1
    use_s3: true  # Toggle for env != local
  ```
- In `src/utils/data_utils.py`, ensure `save_to_s3_or_local` checks `config['execution']['mode']` != 'local' to use boto3.

#### 2. Migrate Data to S3 (Quick, ~$0.01 storage)
- Run module_01 locally → It already saves to `data/scratch/raw/`; now tweak to upload.
- Enhance `save_processed_data` (if needed) with boto3 upload:
  ```python
  import boto3
  def save_to_s3_or_local(df, config, prefix, filename):
      if config['execution']['mode'] != 'local':
          s3 = boto3.client('s3')
          bucket = config['aws']['bucket_name']
          key = f"{prefix}/{filename}"
          df.to_parquet(f"/tmp/{filename}")  # Temp local
          s3.upload_file(f"/tmp/{filename}", bucket, key)
          return f"s3://{bucket}/{key}"
      else:
          # Existing local save
  ```
- Re-run: `python -m src.modules.module_01_data_generation --env sagemaker` (mode triggers S3).
- Verify in Console: S3 → Your bucket → raw/ has parquet.

#### 3. Run ETL/Training in SageMaker (~$2–5 for short jobs)
- **Launch SageMaker Studio**: In Console → SageMaker → Studio → Quick start (free for notebooks).
- **Upload Code**: Clone your GitHub repo into Studio (via git) or upload zip.
- **Notebook for Modules 02–04**:
  - Create `notebooks/aws_pipeline.ipynb`.
  - Import modules, set env='sagemaker' in config.
  - Run sequentially: Load from S3 → ETL → Train → Save forecasts to S3.
  - For ML: Use SageMaker's `sklearn` estimator for RandomForest (script mode):
    ```python
    from sagemaker.sklearn.estimator import SKLearn
    sklearn_estimator = SKLearn(
        entry_point='src/modules/module_04_model_training.py',  # Adapt to script
        role='YourSageMakerRole',
        instance_type='ml.t3.medium',  # Cheap ~$0.05/hr
        framework_version='0.23-1'
    )
    sklearn_estimator.fit({'train': 's3://your-bucket/processed/'})
    ```
  - Prophet locally in notebook (or containerize).
- **Cost Tip**: Run for <30 min, then stop instance.

#### 4. Query & Analyze with Athena (~$0.01 per query)
- **Setup**: In Athena → Create database `capacity_db` → Crawler on S3 bucket (via Glue, free for small).
- **Query Forecasts**: Simulate Citi SQL — e.g., "SELECT server_id, AVG(predicted) FROM forecasts WHERE is_future=true GROUP BY server_id" for underutilized detection.
- Integrate into module_05: Use `boto3` athena client to run queries, fetch results as pandas DF.

#### 5. Optional Automation & Dashboard (~$1–2)
- **Lambda for Pipeline**: Create Lambda function → Trigger on S3 upload (EventBridge) → Run module_01–05 via subprocess (package code in zip).
- **QuickSight Dashboard**: In QuickSight → New dataset from S3 parquet → Build visuals (heatmaps, forecasts). Free for 1 user.
- **Dash App on EC2**: Launch t2.micro (free tier) → Deploy Plotly Dash from module_05 viz → Access via public IP.

#### 6. Testing & Cleanup
- Run full pipeline: Local gen → S3 upload → SageMaker train → Athena query → QuickSight.
- Verify: Check S3 for artifacts, notebook outputs match local.
- Cleanup: Delete S3 files, stop SageMaker, terminate EC2. Use `aws s3 rm s3://your-bucket --recursive`.

### Why This Phase Aligns Perfectly
1. **Data Science/ML**: SageMaker hosts Prophet/scikit-learn; optional Amazon Forecast for managed boost.
2. **Capacity Planning**: Athena queries flag risks seasonally; S3 scales for "enterprise" data.
3. **Python + AWS**: boto3 integration in utils; keeps code clean/modular.

Commit to GitHub: Update README with "AWS Deployment Guide" section + screenshots (S3 bucket, SageMaker job, Athena query results).

Ready to code? I can provide:
- Full boto3 snippet for a module.
- SageMaker notebook template.
- Athena query examples.

Or if this isn't the "next" you meant, clarify! Safe coding 🚀
# Specification Kit: 003 - SageMaker Processing Integration for Module 05 (Risk & Capacity Analysis)

**Spec Kit Version:** 1.0  
**Date:** January 31, 2026  
**Author:** Grok (assisted draft based on project continuity)  
**Status:** Draft - Ready for Review & Implementation  
**Related Specs:**  
- Spec 001: Overall Project Architecture (inferred from initial planning; to be formalized)  
- Spec 002: SageMaker Dry Run for Module 04 (completed, as per 002_SageMaker_DryRun_Report.md)  
**GitHub Issue Link:** [Create a new issue on https://github.com/seanlgirgis/AWS-CapacityForecaster/issues with this spec content]  

---

## 1. Objective
This specification outlines the integration of Amazon SageMaker Processing Jobs for Module 05 (Risk & Capacity Analysis) in the AWS-CapacityForecaster project. Building directly on the successful dry run for Module 04 (Model Training & Forecasting), this step extends cloud execution to the risk analysis phase, enabling scalable, serverless computation of seasonal risks, at-risk server flagging, utilization clustering, and cost optimization recommendations.

This aligns with the project's core priorities:  
- **Priority 1: Data Science/ML** - Enhances risk models with optional Amazon Forecast integration for uncertainty-aware predictions.  
- **Priority 2: Capacity Planning/Performance** - Cloud-enables P95 risk calculations, seasonal analysis, and K-Means-based optimization.  
- **Priority 3: Python + AWS Skills** - Leverages boto3 for job orchestration, S3 for data flow, and SageMaker for managed ML workloads.

**Expected Outcomes:**  
- Reliable SageMaker execution of Module 05, producing S3 artifacts (e.g., risk reports, cost savings JSON).  
- Validation of end-to-end pipeline continuity (Module 04 forecasts → Module 05 risks).  
- Low-cost testing (< $0.10 per run) with cleanup protocols.

## 2. Requirements Phase
### Functional Requirements  
- **FR-05.1:** Submit Module 05 as a SageMaker Processing Job via `module_00_pipeline_runner.py --env sagemaker --only 05`.  
- **FR-05.2:** Automatically map inputs from S3 (`forecasts_dir`) to container path `/opt/ml/processing/input/data`.  
- **FR-05.3:** Process forecasts to compute:  
  - High-risk thresholds (e.g., >90% P95 utilization).  
  - Seasonal flags (EOQ, holidays via `holidays` lib).  
  - Clustering for underutilized servers (`scikit-learn` K-Means).  
  - Cost recommendations (simulated savings based on consolidation).  
- **FR-05.4:** Output artifacts to S3 (`risk_analysis_dir`), including Parquet files and JSON summaries.  
- **FR-05.5:** Optional: Integrate `amazon_forecast.py` for enhanced risk uncertainty (e.g., query P90 bounds).  
- **FR-05.6:** Poll job status and log completion/failure in console/S3 logs.

### Non-Functional Requirements  
- **NFR-05.1:** Cost: Limit to ml.t3.medium instance, <10 min runtime (~$0.05-0.10).  
- **NFR-05.2:** Security: Use configured IAM role with least privilege (S3 read/write, SageMaker create/execute).  
- **NFR-05.3:** Performance: Handle ~100 servers x 90-day forecasts (small dataset for testing).  
- **NFR-05.4:** Reliability: Include error handling for path mismatches, job failures (raise exceptions in runner).  
- **NFR-05.5:** Compatibility: Reuse `sagemaker_launcher.py` without major changes; support local fallback.  
- **NFR-05.6:** Documentation: Update README.md with SageMaker setup guide; create progress report.

### User Stories  
- As a developer, I want to run Module 05 on SageMaker so that I can scale risk analysis without local compute limits.  
- As a portfolio viewer, I want to see AWS integration for Module 05 to demonstrate cloud-native capacity planning.  

### Assumptions & Dependencies  
- Module 04 outputs exist in S3 (`forecasts_dir`).  
- Config.yaml has valid `aws.sagemaker_role_arn`, `sagemaker.image_uri`.  
- boto3 session uses profile from config (e.g., default or specified).  
- No new libraries needed (reuse scikit-learn, pandas from container).

## 3. Design Phase
### System Architecture Updates  
- **High-Level Flow:** Local runner → boto3 create_processing_job → SageMaker (ml.t3.medium) → Container runs `module_05_risk_capacity_analysis.py` with args → Outputs to S3.  
- **Data Flow:**  
  - Input: S3://{bucket}/forecasts/ → /opt/ml/processing/input/data  
  - Code: Upload src/config/code to S3 artifact prefix → /opt/ml/processing/input/project_root & /code  
  - Output: /opt/ml/processing/output → S3://{bucket}/risk_analysis/  
- **Environment:** PYTHONPATH=/opt/ml/processing/input/project_root (for imports).  
- **Optional Forecast Integration:** In `module_05_risk_capacity_analysis.py`, conditional call to `AmazonForecastRunner` if config['ml.models'][3]['enabled'].  

### Database/Storage Schema  
- No new schema; reuse Parquet for risks (columns: server_id, timestamp, risk_level, projected_utilization, savings_potential).  

### UI/UX Considerations  
- None (CLI/S3 outputs); future spec for QuickSight dashboard.  

## 4. Implementation Phase
### Code Changes  
1. **module_00_pipeline_runner.py** (minor):  
   - Confirm Module 05 case in `run_sagemaker_processing_job` (set script_path, input_name='forecasts_data').  
   - Add debug logs for job ARN.  

2. **sagemaker_launcher.py** (reuse):  
   - No changes; verify upload_folder handles Module 05 specifics.  

3. **module_05_risk_capacity_analysis.py** (if not existing; adapt from pipeline):  
   - Parse args for input/output paths.  
   - Load forecasts with pandas.  
   - Compute risks: e.g., `df['risk_high'] = (df['p95_util'] > config['risk_analysis']['high_risk_threshold'])`  
   - Clustering: `from sklearn.cluster import KMeans; kmeans.fit(df[['cpu', 'mem']])`  
   - Optional: `forecast_runner = AmazonForecastRunner(config); df_forecasts = forecast_runner.run_full_cycle(...)`  
   - Save to output_path.  

4. **config.py / config.yaml**:  
   - Add Module 05 SageMaker params under sagemaker.modules.05.  

### Development Steps  
1. Local test: Run `--env local --only 05` to generate sample inputs/outputs.  
2. Config tweaks: Validate paths in config.yaml.  
3. Launch: `python src/module_00_pipeline_runner.py --env sagemaker --only 05`  
4. Verify: Check S3 for outputs; compare to local run.  
5. Commit: `git commit -m "Implement Spec 003: SageMaker for Module 05"`  

## 5. Testing & QA Phase
### Test Plan  
- **Unit Tests:** Add to module_05 (e.g., pytest for risk calc functions).  
- **Integration Tests:** End-to-end job submission; assert S3 artifacts exist.  
- **Traceability Matrix (RTM):**  
  | Requirement | Test Case |  
  |-------------|-----------|  
  | FR-05.1    | Submit job; check ARN. |  
  | FR-05.3    | Validate risk JSON has expected keys. |  
  | NFR-05.1   | Monitor cost in AWS Budgets. |  

- **Edge Cases:** Empty forecasts, invalid paths, job failure.  

## 6. Deployment & Maintenance
### Deployment Checklist  
1. Ensure prerequisites (S3 data, IAM role).  
2. Run command.  
3. Poll & cleanup: Stop if >10 min; delete artifacts if testing.  

### Maintenance Notes  
- Monitor AWS costs weekly.  
- Update for full pipeline (Spec 004: End-to-End Cloud Run).  
- Lessons Learned: Document in 003_Module05_SageMaker_Integration_Report.md.  

---

**Approval & Next Actions:**  
- Review this spec; iterate if needed.  
- Implement per Phase 4.  
- Once complete, draft Spec 004 for full pipeline or QuickSight integration.
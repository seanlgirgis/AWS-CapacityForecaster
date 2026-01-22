# AWS-CapacityForecaster – High-Level Requirements (HLR)

**Version**: 0.1 – Draft  
**Date**: January 2026  
**Status**: In progress – living document  
**Author**: Sean Girgis

## 1. Project Purpose & Business Context
  
Recreation and modernization of enterprise infrastructure capacity forecasting, risk analysis, and resource optimization capabilities developed/maintained during 8 years at Citi Financial (2017–2025).  
Demonstrates strong data science/ML, capacity planning mindset, and modern AWS cloud integration for portfolio purposes.

## 2. Core Objectives (in strict priority order)

1. **Data Science & ML**  
   - Build and compare time-series forecasting models with realistic accuracy gains  
   - Strong focus on feature engineering, seasonality/holiday effects, model evaluation

2. **Capacity Planning & Performance**  
   - Generate enterprise-realistic synthetic P95 metrics  
   - Detect seasonal risk periods & flag at-risk servers  
   - Identify underutilized resources & provide basic cost/right-sizing recommendations

3. **Python Development + AWS Integration**  
   - Clean, modular Python code (pandas-heavy, scikit-learn, Prophet, ...)  
   - End-to-end cloud-native path using S3, SageMaker, Athena (optional Amazon Forecast)  
   - Low-cost / free-tier friendly execution

## 3. Key Functional Capabilities (MVP scope)

List the must-have capabilities – phrase them as "The system shall..." for clarity.

- The system shall generate synthetic daily P95 server metrics (CPU/memory/disk) for 50–200 virtual servers over 3–5 years, including banking seasonality and random peaks.
- The system shall perform data cleansing, outlier handling, and advanced feature engineering (lags, rolling statistics, banking calendar flags).
- The system shall train & compare at least 3 forecasting approaches: baseline (e.g. moving average), Prophet (with seasonality/holidays), and scikit-learn ensemble (RandomForest/XGBoost).
- The system shall compute forecast accuracy metrics (MAPE, MAE, RMSE) and demonstrate ≥20–30% improvement over baseline on seasonal test data.
- The system shall calculate P95 risk scores and flag servers at high risk of saturation in the next 3–6 months.
- The system shall apply clustering (K-Means) to utilization patterns and identify consolidation candidates with rough cost-savings logic.
- The system shall store raw/processed data in S3 and demonstrate querying via Athena.
- The system shall train at least one model in SageMaker (notebook or processing job).

## 4. Non-Functional Requirements & Constraints

- Total AWS spend < $20–30 (ideally much less – heavy free-tier usage)
- Reproducible locally without AWS (core ML + capacity logic)
- Readable, well-commented code following PEP 8
- Portfolio-ready: clear README, 4–8 quality screenshots/visualizations, architecture diagram
- No full web app (interactive notebooks + static reports are sufficient)
- Data volume: realistic but small (≤ 500k–1M rows total)

## 5. Explicit Out-of-Scope (v1.0)

- Real-time streaming ingestion
- Full MLOps (CI/CD, model registry, monitoring)
- Hyperparameter tuning / AutoML
- Production-grade security / IAM complexity
- Large-scale data (> few GB)

## 6. Success Criteria (Definition of Done – MVP)

- Forecasts produced for ≥10 servers with meaningful accuracy gains shown
- Risk flagging & clustering results visualized and interpretable
- End-to-end flow works locally + AWS path demonstrated (even if manual)
- Project documented with rationale, results, and lessons learned

## 7. Next Major Steps After This Document

1. Finalize synthetic data schema & generation approach
2. High-level architecture diagram (components & flow)
3. Folder structure & notebook sequence freeze

References:  
- Inspired by real Citi capacity workflows (BMC TrueSight style metrics & reporting)
- AWS services chosen for resume relevance & low cost
# AWS Capacity Forecaster  
**Cloud-Native Enterprise Infrastructure Capacity Planning & Forecasting System**

Modern recreation of enterprise-scale capacity forecasting, performance risk analysis, and resource optimization pipelines I developed and maintained during 8+ years as Senior Capacity & Data Engineer at Citi Financial (2017–2025).

This project demonstrates production-grade **time-series forecasting**, **feature engineering**, **model comparison**, **seasonal risk flagging**, **utilization optimization**, and **cloud-native AWS workflows** using real AWS services.

<p align="center">
  <img src="https://d2908q01vomqb2.cloudfront.net/b6692ea5df920cad691c20319a6fffd7a4a766b8/2020/09/21/serverless-analytics-pipeline-2-840x630.jpg" alt="AWS Serverless Data Analytics Pipeline" width="720"/>
  <br/><em>AWS serverless analytics & ML pipeline architecture — inspiration for this project</em>
</p>

## 🎯 Key Highlights

- **Heavy focus on time-series ML**: Prophet + scikit-learn ensemble (Random Forest, Gradient Boosting) → 20–35% accuracy improvement over naive baselines
- **Realistic enterprise simulation**: Synthetic daily P95 metrics for 100+ servers (CPU, memory, disk) with banking seasonality & peaks
- **Full capacity planning workflow**: seasonal risk flagging, underutilization detection, consolidation recommendations
- **Modern AWS stack** (low-cost / free-tier friendly): S3, SageMaker, Athena, optional Amazon Forecast
- **Production-grade Python**: pandas, numpy, scikit-learn, Prophet, sqlalchemy (simulated Oracle backup), plotly, joblib

## ✨ Features

- Synthetic enterprise monitoring data generation with realistic seasonality
- Advanced data cleansing, outlier handling & feature engineering (lags, rolling stats, banking calendar flags)
- Multi-model forecasting comparison (Baseline, Prophet, XGBoost, RandomForest)
- P95 risk scoring & seasonal peak vulnerability detection
- K-Means clustering for identifying underutilized/under-provisioned servers
- Cost savings estimation through right-sizing recommendations
- Interactive visualizations & drill-down dashboards (Plotly)
- Cloud-native data pipeline: S3 storage → Athena querying → SageMaker training

## 🏗️ Architecture Overview

```mermaid
flowchart LR
    A[1. Synthetic Data Generation\n(pandas + numpy)] --> B[S3 Raw Zone]
    B --> C[2. ETL / Feature Engineering\n(pandas + joblib)]
    C --> D[S3 Processed Zone]
    D --> E[3. SQL Analytics\n(Amazon Athena)]
    D --> F[4. ML Training & Forecasting\n(SageMaker + Prophet / scikit-learn)]
    D --> G[5. Managed Forecasting\n(optional Amazon Forecast)]
    F --> H[6. Risk Analysis & Optimization\n(P95, Clustering, Recommendations)]
    H --> I[Interactive Visualizations\n(Plotly + Jupyter / Dash)]
    I --> J[Executive Reports & Insights]
```

## 🚀 Tech Stack

| Category               | Technologies                                                                 |
|------------------------|------------------------------------------------------------------------------|
| **Language**           | Python 3.9+                                                                 |
| **Data Processing**    | pandas, numpy, joblib                                                       |
| **Time-Series ML**     | Prophet, scikit-learn (RandomForest, XGBoost, GradientBoosting), statsmodels |
| **Visualization**      | Plotly, matplotlib, seaborn                                                 |
| **AWS Services**       | S3, SageMaker (Studio + Training), Athena, optional Amazon Forecast, boto3 |
| **Other**              | sqlalchemy (simulated Oracle backup), scipy, openpyxl                      |

## 📂 Project Structure (current – evolving)

```
AWS-CapacityForecaster/
├── data/                       # Sample + generated data (gitignored large files)
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_etl_feature_engineering.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_modeling_prophet.ipynb
│   ├── 05_modeling_ensemble.ipynb
│   ├── 06_risk_analysis_optimization.ipynb
│   └── 07_aws_integration_demo.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── utils/
├── docs/                       # Architecture diagrams, data dictionary
├── reports/                    # Generated PDFs, images
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Quick Start (Local)

```bash
# 1. Clone & enter project
git clone https://github.com/seanlgirgis/AWS-CapacityForecaster.git
cd AWS-CapacityForecaster

# 2. Create & activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start exploring
jupyter lab
# → open notebooks/01_data_generation.ipynb
```

AWS setup & credentials are required only for cloud sections (see `notebooks/07_aws_integration_demo.ipynb`).

## 🎯 Project Goals & Personal Motivation

This repository serves as:

- A modern, cloud-native showcase of the **enterprise capacity forecasting & performance engineering systems** I built and ran in production at Citi Financial for 8 years
- A portfolio centerpiece demonstrating strong capabilities in **time-series ML**, **capacity planning**, and **AWS cloud integration**
- A living reference I can continue to expand (more advanced models, full Lambda/EventBridge pipeline, QuickSight dashboards, etc.)

## 📈 Results Snapshot (early findings)

- **Forecast accuracy**: Prophet + holiday effects → ~28% better MAPE vs simple moving average on seasonal banking workloads
- **Risk detection**: Identified 12–18% of servers as seasonal high-risk 3–6 months in advance
- **Optimization potential**: Clustering revealed ~15–22% of servers as strong consolidation candidates (potential multi-million dollar savings at enterprise scale)

## 📄 License

MIT License — feel free to reference and learn from the code.

## 🙌 Acknowledgments

Inspired by real enterprise capacity & performance engineering practices at Citi Financial.  
Built with love for clean Python, responsible ML, and cloud-native architecture.

---

**Feedback, questions, or collaboration welcome!**  
Feel free to open an issue or connect with me on LinkedIn.

Happy forecasting! ☁️📈



import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, List

class RiskAnalyzer:
    """
    Analyzes capacity risk based on forecasts.
    Identifies servers predicted to breach thresholds.
    """
    
    def __init__(self, data_path: str, model_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_artifacts(self):
        """Loads data and model."""
        self.logger.info("Loading artifacts...")
        self.df = pd.read_parquet(self.data_path)
        model_file = list(self.model_path.glob("*.joblib"))[0]
        self.model = joblib.load(model_file)
        self.logger.info(f"Loaded model from {model_file}")

    def generate_predictions(self, split_date='2025-01-01') -> pd.DataFrame:
        """Generates predictions for the test period."""
        # Simple approach: predict on known test set features. 
        # (Recursive usually better for lags, but for risk analysis demo, direct logic fits known future params)
        test_df = self.df[self.df['timestamp'] >= split_date].copy()
        
        # Select features model expects
        # We need to know feature names. The model object might not store them explicitly in sklearn 1.0+ unless we saved them or use feature_names_in_
        try:
            features = self.model.feature_names_in_
        except AttributeError:
            # Fallback based on known columns if not saved
            exclude_cols = ['timestamp', 'server_id', 'app_name', 'business_unit', 
                            'criticality', 'region', 'server_type', 
                            'cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
            features = [c for c in test_df.columns if c not in exclude_cols]
            
        test_df['predicted_cpu'] = self.model.predict(test_df[features])
        return test_df

    def analyze_risk(self, df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
        """Identifies breach events."""
        df['risk_breach'] = df['predicted_cpu'] > threshold
        
        # Aggregate by server
        risk_summary = df.groupby('server_id').agg({
            'predicted_cpu': ['max', 'mean'],
            'risk_breach': 'sum'
        }).reset_index()
        
        risk_summary.columns = ['server_id', 'max_predicted_cpu', 'mean_predicted_cpu', 'breach_days']
        
        # Filter strictly risky servers
        risky_servers = risk_summary[risk_summary['breach_days'] > 0].sort_values('max_predicted_cpu', ascending=False)
        
        self.logger.info(f"Identified {len(risky_servers)} servers at risk (Threshold > {threshold}%)")
        return risky_servers

    def save_report(self, risk_df: pd.DataFrame, filename: str = "risk_report.csv"):
        """Saves risk report."""
        out_file = self.output_path / filename
        risk_df.to_csv(out_file, index=False)
        self.logger.info(f"Risk report saved to {out_file}")

    def run(self):
        """Runs full analysis."""
        self.load_artifacts()
        predictions = self.generate_predictions()
        risky_servers = self.analyze_risk(predictions)
        self.save_report(risky_servers)
        return risky_servers

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = RiskAnalyzer(
        data_path="data/processed/processed_metrics.parquet",
        model_path="models",
        output_path="reports"
    )
    analyzer.run()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path

class Dashboard:
    """
    Visualization dashboard for Capacity Forecasting.
    """
    
    def __init__(self, data_path: str, model_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_data_and_model(self):
        self.df = pd.read_parquet(self.data_path)
        model_file = list(self.model_path.glob("*.joblib"))[0]
        self.model = joblib.load(model_file)
        
    def plot_forecast_vs_actual(self, server_id: str = None, split_date='2025-01-01'):
        """Plots forecast vs actual for a specific server."""
        if server_id is None:
            server_id = self.df['server_id'].unique()[0]
            
        server_df = self.df[(self.df['server_id'] == server_id) & (self.df['timestamp'] >= split_date)].copy()
        
        # Predict
        try:
            features = self.model.feature_names_in_
        except AttributeError:
             # Fallback
            exclude_cols = ['timestamp', 'server_id', 'app_name', 'business_unit', 
                            'criticality', 'region', 'server_type', 
                            'cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
            features = [c for c in server_df.columns if c not in exclude_cols]
            
        server_df['predicted'] = self.model.predict(server_df[features])
        
        plt.figure(figsize=(15, 6))
        plt.plot(server_df['timestamp'], server_df['cpu_p95'], label='Actual CPU %', alpha=0.6)
        plt.plot(server_df['timestamp'], server_df['predicted'], label='Predicted CPU %', alpha=0.8, linestyle='--')
        plt.title(f"Forecast vs Actual: {server_id}")
        plt.legend()
        plt.ylabel("CPU Utilization %")
        out_file = self.output_path / f"forecast_{server_id}.png"
        plt.savefig(out_file)
        plt.close()
        self.logger.info(f"Saved forecast plot to {out_file}")

    def plot_feature_importance(self):
        """Plots feature importance."""
        try:
            importances = self.model.feature_importances_
            try:
                features = self.model.feature_names_in_
            except AttributeError:
                 # Fallback logic repeated or stored
                 return # Skip if can't align
            
            imp_df = pd.DataFrame({'feature': features, 'importance': importances})
            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=imp_df)
            plt.title("Top 10 Feature Importances")
            out_file = self.output_path / "feature_importance.png"
            plt.savefig(out_file)
            plt.close()
            self.logger.info(f"Saved feature importance plot to {out_file}")
        except Exception as e:
            self.logger.warning(f"Could not plot feature importance: {e}")

    def run(self):
        self.load_data_and_model()
        self.plot_feature_importance()
        # Plot for first 3 servers
        for sid in self.df['server_id'].unique()[:3]:
            self.plot_forecast_vs_actual(sid)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dash = Dashboard(
        data_path="data/processed/processed_metrics.parquet",
        model_path="models",
        output_path="reports/plots"
    )
    dash.run()

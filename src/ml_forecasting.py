
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, Tuple, List

class CapacityForecaster:
    """
    Forecasting model for Server Capacity (CPU/Memory).
    Uses Random Forest Regressor.
    """
    
    def __init__(self, data_path: str, model_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.features = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads processed parquet data."""
        self.logger.info(f"Loading data from {self.data_path}")
        return pd.read_parquet(self.data_path)

    def prepare_train_test(self, df: pd.DataFrame, target: str = 'cpu_p95', split_date: str = '2025-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Splits data into train and test based on date."""
        # Identifying features (exclude non-numeric metadata and target)
        exclude_cols = ['timestamp', 'server_id', 'app_name', 'business_unit', 
                        'criticality', 'region', 'server_type', 
                        'cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
        
        self.features = [c for c in df.columns if c not in exclude_cols]
        self.logger.info(f"Training features ({len(self.features)}): {self.features}")
        
        train = df[df['timestamp'] < split_date].copy()
        test = df[df['timestamp'] >= split_date].copy()
        
        return train, test, self.features

    def train(self, train_df: pd.DataFrame, target: str = 'cpu_p95'):
        """Trains the Random Forest model."""
        self.logger.info(f"Training model on {len(train_df)} samples for target: {target}")
        
        X_train = train_df[self.features]
        y_train = train_df[target]
        
        # Determine some reasonable hyperparameters
        # n_estimators=100 is standard default, n_jobs=-1 uses all cores
        self.model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=10, 
            n_jobs=-1, 
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.logger.info("Model training complete.")

    def evaluate(self, test_df: pd.DataFrame, target: str = 'cpu_p95') -> Dict[str, float]:
        """Evaluates model on test set."""
        X_test = test_df[self.features]
        y_test = test_df[target]
        
        predictions = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        self.logger.info(f"Evaluation Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {"mae": mae, "rmse": rmse}

    def save_model(self, filename: str = "rf_capacity_model.joblib"):
        """Saves model to disk."""
        out_file = self.model_path / filename
        joblib.dump(self.model, out_file)
        self.logger.info(f"Model saved to {out_file}")

    def run(self, target='cpu_p95'):
        """Runs full training pipeline."""
        df = self.load_data()
        
        # Simple NaN drop just in case
        df = df.dropna()
        
        train, test, _ = self.prepare_train_test(df, target=target)
        
        self.train(train, target=target)
        metrics = self.evaluate(test, target=target)
        self.save_model()
        return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forecaster = CapacityForecaster(
        data_path="data/processed/processed_metrics.parquet",
        model_path="models"
    )
    forecaster.run()

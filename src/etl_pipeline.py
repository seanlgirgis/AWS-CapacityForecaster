
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import timedelta

class ETLPipeline:
    """
    ETL Pipeline for AWS Capacity Forecasting.
    Handles data loading, cleaning, feature engineering, and splitting.
    """
    
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Loads raw data from CSV/Parquet."""
        try:
            if self.raw_data_path.suffix == '.gz':
                df = pd.read_csv(self.raw_data_path, compression='gzip')
            else:
                df = pd.read_csv(self.raw_data_path)
            
            # Ensure timestamp
            print("Sample raw timestamps:", df['timestamp'].head().tolist())
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("Sample parsed timestamps:", df['timestamp'].head().tolist())
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans data: handles missing values, duplicates."""
        initial_shape = df.shape
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['timestamp', 'server_id'])
        
        # Fill missing values (forward fill then backward fill for time series)
        # Group by server_id to avoid leaking across servers
        df = df.sort_values(['server_id', 'timestamp'])
        
        # Fill missing values column by column to be safe and efficient
        # Identify columns to fill (exclude server_id and timestamp which shouldn't be NaN essentially)
        cols_to_fill = [c for c in df.columns if c not in ['server_id', 'timestamp']]
        
        # Modern pandas way to fill within groups without apply overhead
        for col in cols_to_fill:
            df[col] = df.groupby('server_id')[col].ffill().bfill()
            
        # Fill remaining NaNs (e.g. if a whole server has NaN for a column)
        # For object cols, 'Unknown', for numeric, 0 or mean (using 0 for safety/metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[object_cols] = df[object_cols].fillna('Unknown')
            
        # df = df.reset_index(drop=True)  # Sort likely preserved index, but reset is fine
        df = df.reset_index(drop=True)
        print("Timestamp after cleaning:", df['timestamp'].head().tolist())
        
        self.logger.info(f"Cleaned data. Shape change: {initial_shape} -> {df.shape}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates lag and rolling features."""
        self.logger.info("Starting feature engineering...")
        
        # Sort for proper rolling
        df = df.sort_values(['server_id', 'timestamp'])
        
        # 1. Time Features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # 2. Lag & Rolling Features (on CPU mostly, can expand)
        metrics = ['cpu_p95', 'mem_p95']
        
        for metric in metrics:
            # Lags
            df[f'{metric}_lag_1'] = df.groupby('server_id')[metric].shift(1)
            df[f'{metric}_lag_7'] = df.groupby('server_id')[metric].shift(7)
            df[f'{metric}_lag_30'] = df.groupby('server_id')[metric].shift(30)
            
            # Rolling (closed='left' to avoid leakage if shifted, or just shift before rolling)
            # Rolling 7d mean (shift 1 to use past data for today's forecast)
            grouped = df.groupby('server_id')[metric].shift(1)
            df[f'{metric}_roll_mean_7'] = grouped.rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{metric}_roll_std_7'] = grouped.rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
            
            # Rolling 30d
            df[f'{metric}_roll_mean_30'] = grouped.rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        
        # Drop rows with NaN from lags (optional, or fill)
        # keeping them might be useful if using models that handle NaNs (XGBoost), but for clean training:
        print("Shape before dropna:", df.shape)
        df = df.dropna().reset_index(drop=True)
        print("Shape after dropna:", df.shape)
        
        self.logger.info(f"Feature engineering complete. Columns: {df.columns.tolist()}")
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_metrics.parquet"):
        """Saves processed dataframe."""
        out_file = self.output_path / filename
        df.to_parquet(out_file, index=False)
        self.logger.info(f"Saved processed data to {out_file}")

    def run(self):
        """Executes full pipeline."""
        df = self.load_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        self.save_processed_data(df)
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Default execution for testing
    pipeline = ETLPipeline(
        raw_data_path="data/scratch/server_metrics.csv.gz",
        output_path="data/processed"
    )
    pipeline.run()

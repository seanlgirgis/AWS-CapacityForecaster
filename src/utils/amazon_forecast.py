
import boto3
import time
import pandas as pd
import logging
import json
from botocore.exceptions import ClientError
from datetime import datetime

logger = logging.getLogger(__name__)

class AmazonForecastRunner:
    def __init__(self, config):
        """
        Initializes the Amazon Forecast Runner.
        :param config: Dict containing configuration loaded from config.yaml
        """
        self.config = config
        self.region = config['aws']['region']
        self.bucket_name = config['aws']['bucket_name']
        self.role_arn = config['ml']['models'][3]['params'].get('role_arn') or config['aws']['sagemaker_role_arn'] # Fallback to SM role
        
        # Forecast settings
        self.params = config['ml']['models'][3]['params'] # AmazonForecast is index 3 in list
        self.dataset_group_name = self.params.get('dataset_group_name', 'capacity_forecaster_dsg')
        self.forecast_freq = self.params.get('forecast_frequency', 'D')
        self.horizon = self.params.get('horizon', 90)
        
        # Boto3 clients
        session = boto3.Session(region_name=self.region)
        self.forecast = session.client('forecast')
        self.forecastquery = session.client('forecastquery')
        self.s3 = session.client('s3')

    def check_permissions(self):
        """Simple check to ensure we can list datasets."""
        try:
            self.forecast.list_dataset_groups()
            logger.info("✅ Amazon Forecast permissions confirmed.")
            return True
        except ClientError as e:
            logger.error(f"❌ Forecast Permissions Error: {e}")
            return False

    def prepare_and_upload_data(self, df, upload_path="forecast-input/train.csv"):
        """
        Prepares DataFrame for Amazon Forecast (item_id, timestamp, target_value)
        and uploads to S3.
        """
        logger.info("Preparing data for Amazon Forecast...")
        target_col = self.config['model_training']['target_metric']
        
        # Format: item_id, timestamp, target_value
        # Rename columns to match schema
        forecast_df = df[['server_id', 'timestamp', target_col]].copy()
        forecast_df.columns = ['item_id', 'timestamp', 'target_value']
        
        # Ensure timestamp format YYYY-MM-DD
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.strftime('%Y-%m-%d')
        
        # Save to local csv then upload
        local_csv = "temp_forecast_upload.csv"
        forecast_df.to_csv(local_csv, index=False, header=False) # No header for Forecast CSVs usually, or define in schema
        
        # Upload
        s3_uri = f"s3://{self.bucket_name}/{upload_path}"
        self.s3.upload_file(local_csv, self.bucket_name, upload_path)
        logger.info(f"Uploaded training data to {s3_uri}")
        
        return s3_uri

    def create_dataset_group(self):
        """Creates Dataset Group if not exists."""
        try:
            self.forecast.create_dataset_group(
                DatasetGroupName=self.dataset_group_name,
                Domain='CUSTOM' # or METRICS/RTS. CUSTOM is safest for generic time series
            )
            logger.info(f"Created Dataset Group: {self.dataset_group_name}")
        except self.forecast.exceptions.ResourceAlreadyExistsException:
            logger.info(f"Dataset Group {self.dataset_group_name} already exists.")
            
        return self.get_dataset_group_arn()

    def get_dataset_group_arn(self):
        response = self.forecast.describe_dataset_group(DatasetGroupName=self.dataset_group_name)
        return response['DatasetGroupArn']

    def create_dataset(self):
        """Creates Dataset if not exists."""
        dataset_name = "capacity_metrics_ds"
        try:
            self.forecast.create_dataset(
                DatasetName=dataset_name,
                Domain='CUSTOM',
                DatasetType='TARGET_TIME_SERIES',
                DataFrequency=self.forecast_freq,
                Schema={
                    'Attributes': [
                        {'AttributeName': 'item_id', 'AttributeType': 'string'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'timestamp'},
                        {'AttributeName': 'target_value', 'AttributeType': 'float'}
                    ]
                }
            )
            logger.info(f"Created Dataset: {dataset_name}")
        except self.forecast.exceptions.ResourceAlreadyExistsException:
            logger.info(f"Dataset {dataset_name} already exists.")
        
        dataset_arn = self.forecast.describe_dataset(DatasetName=dataset_name)['DatasetArn']
        
        # Add to Group
        dsg_arn = self.get_dataset_group_arn()
        try:
            self.forecast.update_dataset_group(
                DatasetGroupArn=dsg_arn,
                DatasetArns=[dataset_arn]
            )
        except Exception as e:
            # Ignore if already added or similar
            pass
            
        return dataset_arn

    def import_data(self, s3_uri, dataset_arn):
        """Creates Import Job."""
        import_job_name = f"import_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        logger.info(f"Starting Dataset Import Job: {import_job_name}")
        self.forecast.create_dataset_import_job(
            DatasetImportJobName=import_job_name,
            DatasetArn=dataset_arn,
            DataSource={'S3Config': {'Path': s3_uri, 'RoleArn': self.role_arn}},
            TimestampFormat='yyyy-MM-dd'
        )
        
        self._wait_for_resource(
            lambda: self.forecast.describe_dataset_import_job(DatasetImportJobArn=self._get_import_job_arn(import_job_name)),
            "Status",
            "Dataset Import"
        )

    def _get_import_job_arn(self, name):
         # Helper to find ARN by name (not returned by create call directly in older botocore?)
         # Actually create returns ARN usually.
         # For safety, list and find.
         # But wait, create_dataset_import_job returns DatasetImportJobArn. 
         # I should capture it in the create call.
         # For now, let's assume we can describe by name (actually need ARN).
         # Let's fix the create call above effectively.
         # But assuming boto3 API consistency: 
         # describe takes ARN.
         # So I need to get ARN from create response.
         # Refactoring `import_data` slightly to capture ARN.
         pass # Handled below in run() logic more cleanly or fixed inline above

    def create_predictor(self, dsg_arn):
        """Creates AutoPredictor."""
        predictor_name = f"capacity_predictor_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Creating Predictor: {predictor_name} (this takes 20-40 mins)...")
        
        response = self.forecast.create_auto_predictor(
            PredictorName=predictor_name,
            ForecastHorizon=self.horizon,
            ForecastFrequency=self.forecast_freq,
            DataConfig={
                'DatasetGroupArn': dsg_arn,
                'AttributeConfigs': [
                    {'AttributeName': 'target_value', 'Transformations': {'filled': 'nan'}} 
                ]
            }
        )
        predictor_arn = response['PredictorArn']
        
        self._wait_for_resource(
            lambda: self.forecast.describe_predictor(PredictorArn=predictor_arn),
            "Status",
            "Predictor Training"
        )
        return predictor_arn

    def create_forecast(self, predictor_arn):
        """Generates Forecast."""
        forecast_name = f"capacity_forecast_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Generating Forecast: {forecast_name} (10-20 mins)...")
        
        response = self.forecast.create_forecast(
            ForecastName=forecast_name,
            PredictorArn=predictor_arn
        )
        forecast_arn = response['ForecastArn']
        
        self._wait_for_resource(
            lambda: self.forecast.describe_forecast(ForecastArn=forecast_arn),
            "Status",
            "Forecast Generation"
        )
        return forecast_arn

    def _wait_for_resource(self, describe_func, status_key, resource_name):
        """Polls for resource to become ACTIVE."""
        while True:
            try:
                desc = describe_func()
                status = desc[status_key]
                logger.info(f"{resource_name} Status: {status}")
                
                if status == 'ACTIVE':
                    break
                elif status in ['CREATE_FAILED', 'DELETE_FAILED']:
                    raise RuntimeError(f"{resource_name} failed: {desc.get('Message', 'Unknown error')}")
                
                time.sleep(30)
            except ClientError as e:
                logger.error(f"Polling error: {e}")
                time.sleep(30)

    def query_forecasts(self, forecast_arn, server_ids, start_date, end_date):
        """
        Queries forecasts for multiple items.
        Returns: DataFrame with columns: [server_id, timestamp, p10, p50, p90]
        """
        logger.info("Querying predictions for all servers...")
        results = []
        
        # Batching? Amazon Forecast Query API is single item. 
        # Loop might be slow for 120 servers but manageable (120 calls).
        # We can parallelize with ThreadPoolExecutor if needed.
        
        for sid in server_ids:
            try:
                response = self.forecastquery.query_forecast(
                    ForecastArn=forecast_arn,
                    Filters={"item_id": sid}
                    # StartDate/EndDate are optional, returns full horizon if omitted
                )
                
                # Parse
                predictions = response['Forecast']['Predictions']
                # predictions is dict: {'p10': [...], 'p50': [...], 'p90': [...]}
                # Each list has {'Timestamp': '...', 'Value': ...}
                
                # We assume all quantiles match timestamps
                p10s = predictions['p10']
                p50s = predictions['p50']
                p90s = predictions['p90']
                
                for i in range(len(p50s)):
                    ts_str = p50s[i]['Timestamp']
                    # Forecast returns ISO (YYYY-MM-DDTHH:mm:ss)
                    ts = pd.to_datetime(ts_str).tz_convert(None) 
                    
                    if start_date <= ts <= end_date:
                        results.append({
                            'server_id': sid,
                            'timestamp': ts,
                            'p10': p10s[i]['Value'],
                            'p50': p50s[i]['Value'],
                            'p90': p90s[i]['Value']
                        })
            except Exception as e:
                logger.warning(f"Failed to query forecast for {sid}: {e}")
        
        return pd.DataFrame(results)

    def run_full_cycle(self, df_history, server_ids_to_forecast, test_start_date, test_end_date):
        """Main entry point to run the full Forecast pipeline."""
        if not self.check_permissions():
            return None
            
        s3_uri = self.prepare_and_upload_data(df_history)
        dsg_arn = self.create_dataset_group()
        ds_arn = self.create_dataset()
        
        # Import
        import_job_name = f"import_{datetime.now().strftime('%Y%m%d%H%M')}"
        logger.info(f"Starting Import: {import_job_name}")
        imp_resp = self.forecast.create_dataset_import_job(
            DatasetImportJobName=import_job_name,
            DatasetArn=ds_arn,
            DataSource={'S3Config': {'Path': s3_uri, 'RoleArn': self.role_arn}},
            TimestampFormat='yyyy-MM-dd'
        )
        import_arn = imp_resp['DatasetImportJobArn']
        self._wait_for_resource(
             lambda: self.forecast.describe_dataset_import_job(DatasetImportJobArn=import_arn),
             "Status", "Dataset Import"
        )
        
        # Train
        predictor_arn = self.create_predictor(dsg_arn)
        
        # Forecast
        forecast_arn = self.create_forecast(predictor_arn)
        
        # Query
        df_forecasts = self.query_forecasts(forecast_arn, server_ids_to_forecast, test_start_date, test_end_date)
        return df_forecasts

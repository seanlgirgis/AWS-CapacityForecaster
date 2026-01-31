# src/module_00_pipeline_runner.py
import argparse
import logging
import boto3
from pathlib import Path
import time
import sys
from typing import List, Optional

from src.utils.config import load_config, validate_config

# Import your module mains (adjust paths if needed)
from src.modules.module_01_data_generation import main_process as run_data_gen
from src.modules.module_02_data_load import main_process as run_data_load
from src.modules.module_03_etl_feature_eng import main_process as run_etl
from src.modules.module_04_inner import main_process as run_training
from src.modules.module_05_risk_capacity_analysis import main_process as run_risk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
logging.getLogger('botocore').setLevel(logging.INFO) # Keep botocore quiet unless needed

MODULES = [
    ("01 - Data Generation", run_data_gen),
    ("02 - Data Load & Validation", run_data_load),
    ("03 - ETL & Feature Engineering", run_etl),
    ("04 - Model Training & Forecasting", run_training),
    ("05 - Risk & Capacity Analysis", run_risk),
]

# New MODULES_MAP for easier lookup by module number
MODULES_MAP = {
    "01": ("Data Generation", run_data_gen),
    "02": ("Data Load & Validation", run_data_load),
    "03": ("ETL & Feature Engineering", run_etl),
    "04": ("Model Training & Forecasting", run_training),
    "05": ("Risk & Capacity Analysis", run_risk),
}

def run_pipeline(config, selected: Optional[List[str]] = None):
    start_total = time.time()
    
    # Determine which modules to run
    all_module_nums = sorted(list(MODULES_MAP.keys()))
    selected_modules = []

    if selected:
        # If --only is used, filter by selected
        selected_modules = [m for m in all_module_nums if m in selected]
    else:
        # If no --only, run all
        selected_modules = all_module_nums
    
    # Sort by module number
    selected_modules.sort()
    
    for module_num in selected_modules:
        module_name, func = MODULES_MAP[module_num]
        logger.info(f"Starting {module_num} - {module_name}...")
        module_start = time.time()
        
        try:
            # Special Handling for SageMaker Processing Job (Module 04 & 05)
            if config['execution']['mode'] == 'sagemaker' and module_num in ['04', '05']:
                run_sagemaker_processing_job(config, module_num)
            else:
                # Standard Local/S3 Execution
                func(config)
            
            duration = time.time() - module_start
            logger.info(f"✔ {module_num} - {module_name} completed in {duration:.1f}s")
        except Exception as e:
            logger.error(f"✘ {module_num} - {module_name} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    total_duration = time.time() - start_total
    logger.info(f"Pipeline finished. Total: {total_duration:.1f}s")

def run_sagemaker_processing_job(config, module_num):
    """
    Launches a SageMaker Processing Job for the specified module.
    """
    logger.info(f"🚀 Launching SageMaker Processing Job for Module {module_num}...")
    
    aws_cfg = config.get('aws', {})
    sm_cfg = config.get('sagemaker', {})
    
    role = aws_cfg.get('sagemaker_role_arn')
    if not role or "REPLACE_WITH_YOUR_ARN" in role:
        raise ValueError("SageMaker Role ARN not configured in config.yaml")

    bucket = aws_cfg.get('bucket_name')
    region = aws_cfg.get('region', 'us-east-1')
    profile = aws_cfg.get('profile') # Get profile from config
    
    # Initialize boto3 session with credentials
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    sm_client = boto_session.client('sagemaker')

    # Initialize Processor
    # --- Launch Job using Shared Utility ---
    from src.utils.sagemaker_launcher import launch_project_processing_job
    
    # Define Inputs/Outputs based on Module
    if module_num == '04':
        input_prefix = config['paths']['processed_dir'].strip('/')
        output_prefix = config['paths']['forecasts_dir'].strip('/')
        script_path = "src/modules/module_04_model_training.py"
        input_name = 'input_data' # Matches arg parser in mod 04
    elif module_num == '05':
        input_prefix = config['paths']['forecasts_dir'].strip('/')
        output_prefix = config['paths']['risk_analysis_dir'].strip('/')
        script_path = "src/modules/module_05_risk_capacity_analysis.py"
        input_name = 'forecasts_data' 
    else:
        raise ValueError(f"Module {module_num} not supported on SageMaker yet.")

    input_s3_uri = f"s3://{bucket}/{input_prefix}/"
    output_s3_uri = f"s3://{bucket}/{output_prefix}/"

    input_configs = [{
        'source': input_s3_uri,
        'dest': '/opt/ml/processing/input/data',
        'name': 'input_data'
    }]
    
    output_configs = [{
        'source': '/opt/ml/processing/output',
        'dest': output_s3_uri,
        'name': 'output_data'
    }]
    
    # Standardize container args: input mapped to /data, output mapped to /output
    container_args = [
        '--env', 'sagemaker',
        '--config', '/opt/ml/processing/input/project_root/config/config.yaml',
        '--input_data_path', '/opt/ml/processing/input/data',
        '--output_data_path', '/opt/ml/processing/output'
    ]
    
    try:
        arn, job_name = launch_project_processing_job(
            boto_session=boto_session,
            role_arn=role,
            image_uri=sm_cfg.get('modules', {}).get(module_num, {}).get('image_uri', sm_cfg.get('image_uri')),
            instance_type=sm_cfg.get('modules', {}).get(module_num, {}).get('instance_type', sm_cfg.get('instance_type', 'ml.t3.medium')),
            instance_count=sm_cfg.get('instance_count', 1),
            bucket_name=bucket,
            project_root_local=".",
            main_script_path=script_path,
            container_entrypoint_args=container_args,
            input_data_configs=input_configs,
            output_data_configs=output_configs
        )
        
        logger.info(f"✅ Job {job_name} submitted successfully! ARN: {arn}")
        logger.info(f"Check status in AWS Console: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/processing-jobs/{job_name}")

        # Polling for completion
        logger.info("Waiting for job to complete...")
        while True:
            desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
            status = desc['ProcessingJobStatus']
            if status in ['Completed', 'Failed', 'Stopped']:
                logger.info(f"Job finished with status: {status}")
                if status == 'Failed' and 'FailureReason' in desc:
                    logger.error(f"Failure reason: {desc['FailureReason']}")
                    raise RuntimeError(f"Job failed: {desc['FailureReason']}")
                break
            logger.info(f"Current status: {status} ... (waiting 30s)")
            time.sleep(30)
        
    except Exception as e:
        logger.error(f"Failed to launch processing job: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS-CapacityForecaster Pipeline Runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", nargs="?", const="local", choices=["local", "sagemaker"])
    parser.add_argument("--only", nargs="+", help="Run only these modules e.g. --only 03 05")
    parser.add_argument("--skip", nargs="+", help="Skip these modules")
    args = parser.parse_args()
    
    config = load_config(Path(args.config))
    if args.env:
        config.setdefault('execution', {})['mode'] = args.env
    validate_config(config)
    
    selected = None
    if args.only:
        selected = args.only
    elif args.skip:
        selected = [m[0].split(" - ")[0] for m in MODULES if m[0].split(" - ")[0] not in args.skip]
    
    run_pipeline(config, selected)
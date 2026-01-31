import sys
import boto3
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.utils.config import load_config
from src.utils.sagemaker_launcher import launch_project_processing_job

def launch_mod_05():
    print("Loading config...")
    config = load_config(Path("config/config.yaml"))
    
    aws_cfg = config.get('aws', {})
    sm_cfg = config.get('sagemaker', {})
    
    role = aws_cfg.get('sagemaker_role_arn')
    bucket = aws_cfg.get('bucket_name')
    region = aws_cfg.get('region', 'us-east-1')
    profile = aws_cfg.get('profile')
    
    print(f"Using profile: {profile}, Region: {region}, Bucket: {bucket}")
    
    session = boto3.Session(profile_name=profile, region_name=region)
    
    # Module 05 Settings
    module_num = '05'
    input_prefix = config['paths']['forecasts_dir'].strip('/')
    output_prefix = config['paths']['risk_analysis_dir'].strip('/')
    script_path = "src/modules/module_05_risk_capacity_analysis.py"
    
    input_s3_uri = f"s3://{bucket}/{input_prefix}/"
    output_s3_uri = f"s3://{bucket}/{output_prefix}/"
    
    print(f"Input: {input_s3_uri}")
    print(f"Output: {output_s3_uri}")
    
    input_configs = [{
        'source': input_s3_uri,
        'dest': '/opt/ml/processing/input/data',
        'name': 'forecasts_data'
    }]
    
    output_configs = [{
        'source': '/opt/ml/processing/output',
        'dest': output_s3_uri,
        'name': 'output_data'
    }]
    
    container_args = [
        '--env', 'sagemaker',
        '--config', '/opt/ml/processing/input/project_root/config/config.yaml',
        '--input_data_path', '/opt/ml/processing/input/data',
        '--output_data_path', '/opt/ml/processing/output'
    ]
    
    # Get image/instance from config
    mod_cfg = sm_cfg.get('modules', {}).get(module_num, {})
    image_uri = mod_cfg.get('image_uri', sm_cfg.get('image_uri'))
    instance_type = mod_cfg.get('instance_type', sm_cfg.get('instance_type'))
    
    print(f"Launching with Image: {image_uri}")
    print(f"Instance: {instance_type}")
    
    try:
        arn, job_name = launch_project_processing_job(
            boto_session=session,
            role_arn=role,
            image_uri=image_uri,
            instance_type=instance_type,
            instance_count=1,
            bucket_name=bucket,
            project_root_local=".",
            main_script_path=script_path,
            container_entrypoint_args=container_args,
            input_data_configs=input_configs,
            output_data_configs=output_configs
        )
        print(f"SUCCESS: Job Submitted! ARN: {arn}")
        print(f"Job Name: {job_name}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_mod_05()

import os
import time
import logging
import boto3

logger = logging.getLogger(__name__)

def upload_folder(boto_session, bucket_name, local_path, s3_prefix):
    """
    Recursively uploads a directory to S3.
    """
    s3_client = boto_session.client('s3')
    logger.info(f"Uploading folder {local_path} to s3://{bucket_name}/{s3_prefix}...")
    
    count = 0
    for root, dirs, files in os.walk(local_path):
        if "__pycache__" in root or ".git" in root or ".ipynb_checkpoints" in root: 
            continue
            
        for file in files:
            if file.endswith((".pyc", ".DS_Store")): 
                continue
                
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, local_path)
            # Ensure S3 keys use forward slashes
            s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/")
            
            s3_client.upload_file(local_file, bucket_name, s3_key)
            count += 1
            
    logger.info(f"Uploaded {count} files from {local_path}")
    return f"s3://{bucket_name}/{s3_prefix}"

def launch_project_processing_job(
    boto_session,
    role_arn,
    image_uri,
    instance_type,
    instance_count,
    bucket_name,
    project_root_local,
    main_script_path,
    container_entrypoint_args,
    input_data_configs, # List of dicts {'source': s3_uri, 'dest': container_path, 'name': str}
    output_data_configs, # List of dicts {'source': container_path, 'dest': s3_uri, 'name': str}
    job_name_prefix="proc-job"
):
    """
    Launches a SageMaker Processing Job using the 'Manual Folder Upload + Direct API' strategy.
    
    Automatically uploads:
      - project_root_local/src   -> /opt/ml/processing/input/project_root/src
      - project_root_local/config -> /opt/ml/processing/input/project_root/config
      - main_script_path         -> /opt/ml/processing/input/code/script.py
      
    Sets PYTHONPATH to /opt/ml/processing/input/project_root
    """
    sm_client = boto_session.client('sagemaker')
    s3_client = boto_session.client('s3')
    
    # 1. Generate Job ID and Artifact Prefix
    timestamp = int(time.time())
    job_name = f"{job_name_prefix}-{timestamp}"
    # Ensure job name doesn't exceed 63 chars (SageMaker limit), though timestamp is safe
    artifact_prefix = f"sagemaker/artifacts/{job_name}"
    
    logger.info(f"Preparing to launch Job: {job_name}")
    
    # 2. Upload Project Artifacts (src and config)
    # We assume project_root_local contains 'src' and 'config'
    src_local = os.path.join(project_root_local, "src")
    config_local = os.path.join(project_root_local, "config")
    
    src_s3 = upload_folder(boto_session, bucket_name, src_local, f"{artifact_prefix}/src")
    config_s3 = upload_folder(boto_session, bucket_name, config_local, f"{artifact_prefix}/config")
    
    # 3. Upload Main Script
    script_filename = os.path.basename(main_script_path)
    code_s3_key = f"{artifact_prefix}/code/{script_filename}"
    logger.info(f"Uploading main script {main_script_path}...")
    s3_client.upload_file(main_script_path, bucket_name, code_s3_key)
    code_s3_uri = f"s3://{bucket_name}/{code_s3_key}"
    
    # 4. Construct ProcessingInputs
    processing_inputs = []
    
    # Data Inputs
    for inp in input_data_configs:
        processing_inputs.append({
            'InputName': inp['name'],
            'S3Input': {
                'S3Uri': inp['source'],
                'LocalPath': inp['dest'],
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        })
        
    # Code/Project Inputs
    processing_inputs.extend([
        {
            'InputName': 'src_code',
            'S3Input': {
                'S3Uri': src_s3,
                'LocalPath': '/opt/ml/processing/input/project_root/src',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        {
            'InputName': 'config_files',
            'S3Input': {
                'S3Uri': config_s3,
                'LocalPath': '/opt/ml/processing/input/project_root/config',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        {
            'InputName': 'code',
            'S3Input': {
                'S3Uri': code_s3_uri,
                'LocalPath': '/opt/ml/processing/input/code',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        }
    ])
    
    # 5. Construct ProcessingOutputs
    processing_outputs = []
    for out in output_data_configs:
        processing_outputs.append({
            'OutputName': out['name'],
            'S3Output': {
                'S3Uri': out['dest'],
                'LocalPath': out['source'],
                'S3UploadMode': 'EndOfJob'
            }
        })
        
    # 6. Call CreateProcessingJob
    logger.info(f"Sending CreateProcessingJob request for {job_name}...")
    
    # The entrypoint in the container will be the uploaded script in /code
    container_entrypoint = ['python3', f"/opt/ml/processing/input/code/{script_filename}"]
    
    response = sm_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
                'VolumeSizeInGB': 30
            }
        },
        AppSpecification={
            'ImageUri': image_uri,
            'ContainerEntrypoint': container_entrypoint,
            'ContainerArguments': container_entrypoint_args
        },
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig={
            'Outputs': processing_outputs
        },
        Environment={
            'PYTHONPATH': '/opt/ml/processing/input/project_root' # Critical for imports
        }
    )
    
    return response['ProcessingJobArn'], job_name

# src/utils/aws_utils.py
"""
AWS Utilities Module

This module provides utility functions for interacting with AWS services, primarily focusing on S3 operations.
It abstracts away common AWS tasks using boto3, promoting reusable and maintainable code.
Designed to support the AWS-CapacityForecaster project by handling data storage and retrieval for ETL pipelines,
ML model training, and capacity forecasting workflows.

Key Features:
- Session management for consistent AWS interactions.
- S3 bucket creation, file upload/download, and object listing.
- Error handling with logging for reliability in cloud environments.
- Cost-aware design: Functions are lightweight to minimize AWS usage and costs.

Dependencies:
- boto3: AWS SDK for Python.
- botocore: Low-level interface for boto3, used for exception handling.
- logging: For traceability and debugging.
- os: For local file path handling.

Usage:
Import functions as needed, e.g., from utils.aws_utils import upload_to_s3
Ensure AWS credentials are configured via environment variables or AWS CLI.

Author: [Your Name]
Date: January 2026
Version: 1.0
"""

import boto3
import botocore.exceptions
import logging
import botocore.exceptions
import logging
import os
from src.utils.config import get_aws_config

# Set up logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Adjust level as needed (e.g., DEBUG for development)

def get_aws_session(region: str = None) -> boto3.Session:
    """
    Create and return a boto3 session for AWS operations.

    This function initializes a session with the specified region, using credentials from
    environment variables, AWS CLI configuration, or IAM roles. It ensures consistent
    session handling across the project, reducing overhead in repeated calls.

    Args:
        region (str): AWS region to use (defaults to config value or 'us-east-1').

    Returns:
        boto3.Session: An active boto3 session object.

    Raises:
        botocore.exceptions.NoCredentialsError: If no AWS credentials are found.
    """
    try:
        aws_cfg = get_aws_config()
        if region is None:
            region = aws_cfg.get('region', 'us-east-1')
        
        # Optional: Support named profiles from config if set
        profile = aws_cfg.get('profile')
        
        session = boto3.Session(region_name=region, profile_name=profile)
        logger.info(f"AWS session created for region: {region}" + (f" with profile: {profile}" if profile else ""))
        return session
    except botocore.exceptions.NoCredentialsError as e:
        logger.error("AWS credentials not found. Please configure via AWS CLI or environment variables.")
        raise e
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        raise e

def create_s3_bucket(bucket_name: str, region: str = None) -> bool:
    """
    Create an S3 bucket if it does not already exist.

    This function checks for the existence of the specified bucket and creates it if necessary.
    It sets the bucket location to the provided region. Useful for initial project setup to store
    synthetic data, processed metrics, or model artifacts.

    Args:
        bucket_name (str): The unique name of the S3 bucket to create.
        region (str): AWS region for the bucket (defaults to session region).

    Returns:
        bool: True if the bucket was created or already exists.

    Raises:
        botocore.exceptions.ClientError: If there's an issue with bucket creation (e.g., naming conflict).
    """
    session = get_aws_session(region)
    # Ensure we use the resolved region from the session if argument was None
    region = session.region_name
    s3_client = session.client('s3')
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"S3 bucket '{bucket_name}' already exists.")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            # Bucket does not exist, create it
            create_params = {'Bucket': bucket_name}
            if region != 'us-east-1':  # us-east-1 does not require location constraint
                create_params['CreateBucketConfiguration'] = {'LocationConstraint': region}
            s3_client.create_bucket(**create_params)
            logger.info(f"S3 bucket '{bucket_name}' created in region '{region}'.")
            return True
        else:
            logger.error(f"Error checking/creating S3 bucket '{bucket_name}': {str(e)}")
            raise e

def upload_to_s3(local_path: str, bucket_name: str, s3_key: str, session: boto3.Session = None) -> str:
    """
    Upload a local file to an S3 bucket.

    This function uploads the specified local file to S3 under the given key. It supports
    automatic multipart uploads for large files. Ideal for storing synthetic server metrics
    or processed data in the project's ETL pipeline.

    Args:
        local_path (str): Path to the local file to upload.
        bucket_name (str): Name of the target S3 bucket.
        s3_key (str): S3 object key (path within the bucket).
        session (boto3.Session, optional): Existing session; creates one if None.

    Returns:
        str: The S3 URI of the uploaded object (e.g., 's3://bucket/key').

    Raises:
        FileNotFoundError: If the local file does not exist.
        botocore.exceptions.ClientError: If upload fails (e.g., permissions issue).
    """
    if not os.path.exists(local_path):
        logger.error(f"Local file not found: {local_path}")
        raise FileNotFoundError(f"File not found: {local_path}")
    
    if session is None:
        session = get_aws_session()
    
    s3_client = session.client('s3')
    try:
        s3_client.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"File uploaded to {s3_uri}")
        return s3_uri
    except botocore.exceptions.ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise e

def download_from_s3(bucket_name: str, s3_key: str, local_path: str, session: boto3.Session = None) -> bool:
    """
    Download an object from S3 to a local path.

    This function retrieves the specified S3 object and saves it locally. Useful for pulling
    back processed data or model outputs from S3 for local analysis or visualization.

    Args:
        bucket_name (str): Name of the source S3 bucket.
        s3_key (str): S3 object key to download.
        local_path (str): Local path to save the file.
        session (boto3.Session, optional): Existing session; creates one if None.

    Returns:
        bool: True if download succeeds.

    Raises:
        botocore.exceptions.ClientError: If download fails (e.g., object not found).
    """
    if session is None:
        session = get_aws_session()
    
    s3_client = session.client('s3')
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(Bucket=bucket_name, Key=s3_key, Filename=local_path)
        logger.info(f"File downloaded from s3://{bucket_name}/{s3_key} to {local_path}")
        return True
    except botocore.exceptions.ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        raise e

def list_s3_objects(bucket_name: str, prefix: str = '', session: boto3.Session = None) -> list:
    """
    List objects in an S3 bucket with an optional prefix.

    This function retrieves a list of object keys matching the prefix. It handles pagination
    for large buckets. Helpful for inventorying data files or debugging uploads in the project.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Optional prefix to filter objects (default: '').
        session (boto3.Session, optional): Existing session; creates one if None.

    Returns:
        list: List of S3 object keys (strings).

    Raises:
        botocore.exceptions.ClientError: If listing fails (e.g., bucket not found).
    """
    if session is None:
        session = get_aws_session()
    
    s3_client = session.client('s3')
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        objects = []
        for page in pages:
            if 'Contents' in page:
                objects.extend([obj['Key'] for obj in page['Contents']])
        
        logger.info(f"Listed {len(objects)} objects in s3://{bucket_name}/{prefix}")
        return objects
    except botocore.exceptions.ClientError as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        raise e

# Example usage (for testing; remove or comment out in production)
if __name__ == "__main__":
    # Configure logging to console for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test: Create bucket (use a unique name)
    # create_s3_bucket('test-aws-capacity-forecast-bucket')
    
    # Test: Upload a sample file
    # sample_path = 'path/to/sample.csv'  # Replace with actual path
    # upload_to_s3(sample_path, 'test-aws-capacity-forecast-bucket', 'test/sample.csv')
    
    # Test: List objects
    # print(list_s3_objects('test-aws-capacity-forecast-bucket'))
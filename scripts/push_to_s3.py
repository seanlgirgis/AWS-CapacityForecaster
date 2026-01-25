
"""
push_to_s3.py

This script uploads the generated synthetic data to the AWS S3 bucket configured in config.yaml.
It serves as the first stage of the ETL pipeline, moving local data to the cloud for processing.

Usage:
    python scripts/push_to_s3.py
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.config import get_data_config, get_aws_config
from utils.aws_utils import upload_to_s3, create_s3_bucket, get_aws_session

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('push_to_s3')

def main():
    logger.info("Initializing S3 Upload Process...")

    # 1. Load Configuration
    data_config = get_data_config()
    aws_config = get_aws_config()

    file_path = data_config.get('generated_data_path')
    bucket_name = aws_config.get('bucket_name')
    raw_prefix = aws_config.get('raw_prefix', 'raw/')
    profile = aws_config.get('profile')

    if not file_path:
        logger.error("No 'generated_data_path' found in config.yaml.")
        sys.exit(1)
    
    # Resolve absolute path relative to project root if it's not absolute
    local_path = Path(file_path)
    if not local_path.is_absolute():
        local_path = project_root / local_path

    if not local_path.exists():
        logger.error(f"Data file not found at: {local_path}")
        logger.info("Run 'python src/data_generation.py' first.")
        sys.exit(1)

    logger.info("Configuration Loaded:")
    logger.info(f"  - Local File: {local_path}")
    logger.info(f"  - Bucket: {bucket_name}")
    logger.info(f"  - Profile: {profile}")

    # 2. Check/Create Bucket
    logger.info("Verifying S3 Bucket...")
    try:
        # get_aws_session will automatically use the profile from config if set there, 
        # but we can explicitly ensure it matches our expectation.
        session = get_aws_session() 
        create_s3_bucket(bucket_name)
    except Exception as e:
        logger.error(f"Failed to access/create bucket: {e}")
        sys.exit(1)

    # 3. Upload File
    file_name = local_path.name
    s3_key = f"{raw_prefix}{file_name}"
    
    logger.info(f"Uploading to s3://{bucket_name}/{s3_key}")
    try:
        s3_uri = upload_to_s3(str(local_path), bucket_name, s3_key, session=session)
        logger.info("=" * 50)
        logger.info(f"SUCCESS: Data successfully uploaded to AWS S3.")
        logger.info(f"Location: {s3_uri}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

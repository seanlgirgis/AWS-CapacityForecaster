import sys
import os
from pathlib import Path
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Add project root to path so we can import src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

def setup_aws_resources():
    logger.info("loading configuration...")
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    aws_cfg = config.get('aws', {})
    region = aws_cfg.get('region', 'us-east-1')
    bucket_name = aws_cfg.get('bucket_name')

    if not bucket_name:
        logger.error("No 'bucket_name' found in config.yaml under 'aws' section.")
        return

    logger.info(f"Target Region: {region}")
    logger.info(f"Target Bucket: {bucket_name}")

    # Initialize S3 client
    # Debug: Check if credentials exist (Env Vars OR ~/.aws/credentials)
    key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_dir = Path.home() / ".aws" / "credentials"
    
    if not key_id and not aws_dir.exists():
        logger.error("❌ CRITICAL: No AWS credentials found!")
        logger.error(f"Checked environment variables and {aws_dir}")
        return

    if key_id:
        logger.info(f"✔ Found credentials in Environment Variables")
    elif aws_dir.exists():
        logger.info(f"✔ Found credentials in {aws_dir}")

    try:
        profile = aws_cfg.get('profile')
        logger.info(f"Using AWS Profile: {profile}")
        
        session = boto3.Session(profile_name=profile, region_name=region)
        s3 = session.client('s3')
        
        # Verify credentials by making a harmless call
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"✔ Authenticated as: {identity['Arn']}")
    except NoCredentialsError:
        logger.error("AWS Credentials not found (boto3 could not locate them).")
        return
    except ClientError as e:
        logger.error(f"AWS Authentication failed: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to create boto3 client: {e}")
        return

    # Create Bucket
    logger.info("Attempting to create S3 bucket...")
    try:
        if region == "us-east-1":
            # us-east-1 is special, it doesn't take LocationConstraint
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        logger.info(f"✔ Bucket '{bucket_name}' created successfully (or already exists).")
        
        # Enable versioning (optional but recommended)
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        logger.info("✔ Versioning enabled.")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'BucketAlreadyOwnedByYou':
            logger.info(f"✔ Bucket '{bucket_name}' already exists and is owned by you.")
        elif error_code == 'BucketAlreadyExists':
            logger.error(f"✘ Bucket '{bucket_name}' already exists but is owned by SOMEONE ELSE. Choose a different name in config.yaml.")
        else:
            logger.error(f"✘ Failed to create bucket: {e}")

if __name__ == "__main__":
    setup_aws_resources()

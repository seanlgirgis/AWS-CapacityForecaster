import boto3
import json
import logging
import time
import sys
from pathlib import Path
from botocore.exceptions import ClientError
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

ROLE_NAME = "CapacityForecaster-SageMakerRole"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def create_sagemaker_role(iam_client):
    """
    Creates an IAM role for SageMaker with S3 and SageMaker Full Access.
    """
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        # Check if role exists
        logger.info(f"Checking for existing role '{ROLE_NAME}'...")
        role = iam_client.get_role(RoleName=ROLE_NAME)
        role_arn = role['Role']['Arn']
        logger.info(f"✔ Role already exists: {role_arn}")
        return role_arn

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            logger.info("Role not found. Creating new role...")
            try:
                role = iam_client.create_role(
                    RoleName=ROLE_NAME,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description="Role for AWS-CapacityForecaster SageMaker Processing Jobs"
                )
                role_arn = role['Role']['Arn']
                logger.info(f"✔ Created role: {role_arn}")
            except Exception as create_err:
                logger.error(f"Failed to create role: {create_err}")
                raise
        else:
            logger.error(f"Failed to check role: {e}")
            raise

    # Attach Policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ]

    for policy_arn in policies:
        try:
            iam_client.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
            logger.info(f"✔ Attached policy: {policy_arn.split('/')[-1]}")
        except ClientError as e:
            logger.error(f"Failed to attach policy {policy_arn}: {e}")

    # Wait for propagation
    logger.info("Waiting 10s for IAM propagation...")
    time.sleep(10)
    return role_arn

def update_config(role_arn):
    """
    Updates config.yaml with the new Role ARN.
    """
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found at {CONFIG_PATH}")
        return False

    try:
        # Use a round-trip loader to preserve comments if possible, 
        # but standard yaml often strips them. 
        # For safety and distinct modification, we'll read as string first to check.
        
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Simple string replacement to preserve comments structure best we can
        new_lines = []
        updated = False
        for line in lines:
            if "sagemaker_role_arn:" in line:
                # Keep indentation
                indent = line.split("sagemaker_role_arn:")[0]
                new_lines.append(f'{indent}sagemaker_role_arn: "{role_arn}" # Auto-updated\n')
                updated = True
            else:
                new_lines.append(line)
        
        if not updated:
            # Fallback if key missing
            logger.warning("sagemaker_role_arn key not found in text. Appending...")
            new_lines.append(f'\naws:\n  sagemaker_role_arn: "{role_arn}"\n')

        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        logger.info(f"✔ Updated {CONFIG_PATH} with new Role ARN")
        return True

    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False

def main():
    logger.info("Loading AWS credentials...")
    try:
        # Load config to get profile
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        profile = cfg.get('aws', {}).get('profile', 'default')
        logger.info(f"Using profile: {profile}")
        
        session = boto3.Session(profile_name=profile)
        iam = session.client('iam')
        
        # 1. Create Role
        role_arn = create_sagemaker_role(iam)
        
        # 2. Update Config
        if update_config(role_arn):
            logger.info("\n✅ Setup Complete!")
            logger.info("You can now run the pipeline verification:")
            print("\n    python -m src.modules.module_00_pipeline_runner --env sagemaker --only 04\n")
            
    except Exception as e:
        logger.error(f"Setup Failed: {e}")
        logger.error("Ensure you have 'iam:CreateRole' and 'iam:AttachRolePolicy' permissions.")

if __name__ == "__main__":
    main()

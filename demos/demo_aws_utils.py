"""
# Run command: python demos/demo_aws_utils.py
Demo: AWS Utilities (src/utils/aws_utils.py)

This script demonstrates the functionality of the aws_utils module by performing a complete
workflow of S3 operations using the configured AWS profile (e.g., 'study').

Functionality Covered:
1.  Session Management: Creating an AWS session.
2.  Buckets: Creating a new S3 bucket (or verifying existence).
3.  Upload: Uploading a local file to S3.
4.  Listing: Listing objects in the bucket.
5.  Download: Downloading the file back from S3 to verify integrity.

Prerequisites:
- AWS credentials must be configured (via 'aws configure' or .env).
- The 'study' profile (or default) should have S3 permissions.
"""

import logging
import os
import sys
import uuid
from datetime import datetime

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.aws_utils import (
    get_aws_session,
    create_s3_bucket,
    upload_to_s3,
    download_from_s3,
    list_s3_objects
)
from src.utils.config import get_aws_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_demo():
    print("\n" + "="*80)
    print(" ☁️  AWS-CapacityForecaster: AWS Utilities Demo")
    print("="*80)

    # 1. Initialize Session
    print("\n[1] Initializing AWS Session...")
    try:
        session = get_aws_session()
        region = session.region_name
        # Mask account ID for privacy in logs
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"    ✅ Session active in region: {region}")
        print(f"    ✅ Authenticated as: {identity['Arn']}")
    except Exception as e:
        print(f"    ❌ Failed to initialize session: {e}")
        return

    # 2. Bucket Creation
    # Use a unique bucket name to avoid collisions
    # Note: In a real app, this would be a config value
    bucket_config = get_aws_config().get("bucket_name", f"demo-bucket-{uuid.uuid4().hex[:8]}")
    # Force a temporary demo bucket just for this script if config bucket is generic?
    # Let's use the config one if set, or generate one.
    
    print(f"\n[2] Creating/Verifying Bucket: '{bucket_config}'")
    try:
        create_s3_bucket(bucket_config, region=region)
        print(f"    ✅ Bucket ready: {bucket_config}")
    except Exception as e:
        print(f"    ❌ Failed to create bucket: {e}")
        return

    # 3. Create a Dummy Local File
    print("\n[3] Preparing Local Data...")
    demo_dir = os.path.join(os.getcwd(), "demos", "temp_data")
    os.makedirs(demo_dir, exist_ok=True)
    
    local_filename = f"demo_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    local_path = os.path.join(demo_dir, local_filename)
    
    with open(local_path, "w") as f:
        f.write("server_id,cpu_usage,timestamp\n")
        f.write("server_01,0.85,2024-01-01T12:00:00\n")
        f.write("server_02,0.42,2024-01-01T12:00:00\n")
    
    print(f"    ✅ Created local file: {local_path}")

    # 4. Upload to S3
    s3_key = f"demos/{local_filename}"
    print(f"\n[4] Uploading to S3 (Key: {s3_key})...")
    try:
        s3_uri = upload_to_s3(local_path, bucket_config, s3_key)
        print(f"    ✅ Upload successful: {s3_uri}")
    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        return

    # 5. List Objects
    print(f"\n[5] Listing Objects in s3://{bucket_config}/demos/ ...")
    try:
        objects = list_s3_objects(bucket_config, prefix="demos/")
        print(f"    ✅ Found {len(objects)} objects:")
        for obj in objects:
            print(f"       - {obj}")
    except Exception as e:
        print(f"    ❌ Listing failed: {e}")

    # 6. Download from S3
    download_dest = os.path.join(demo_dir, f"downloaded_{local_filename}")
    print(f"\n[6] Downloading back to: {download_dest} ...")
    try:
        download_from_s3(bucket_config, s3_key, download_dest)
        
        # Verify content match
        with open(local_path, "r") as f1, open(download_dest, "r") as f2:
            if f1.read() == f2.read():
                print("    ✅ Download successful & content Verified!")
            else:
                print("    ⚠️  Download successful but content mismatch!")
    except Exception as e:
        print(f"    ❌ Download failed: {e}")

    # Cleanup (Optional - commented out for safety)
    # print("\n[7] Cleanup...")
    # os.remove(local_path)
    # os.remove(download_dest)
    # print("    ✅ Local temp files removed.")

    print("\n" + "="*80)
    print(" ✅ AWS Utils Demo Complete")
    print("="*80)

if __name__ == "__main__":
    run_demo()


# Run command: python -m pytest tests/test_aws_utils.py -v
import pytest
import boto3
import os
from moto import mock_aws
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError
from src.utils.aws_utils import (
    get_aws_session,
    create_s3_bucket,
    upload_to_s3,
    download_from_s3,
    list_s3_objects
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration to avoid ProfileNotFound errors."""
    with patch("src.utils.aws_utils.get_aws_config") as mock_get:
        # Return a simple config without a profile
        mock_get.return_value = {"region": "us-east-1", "profile": None}
        yield mock_get

@pytest.fixture(scope="function")
def s3_client(aws_credentials):
    """Return a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")

@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary dummy file for upload tests."""
    p = tmp_path / "test_data.csv"
    p.write_text("col1,col2\n1,2")
    return str(p)

# -----------------------------
# Tests
# -----------------------------

def test_get_aws_session(aws_credentials):
    """Test session creation."""
    session = get_aws_session(region="us-west-2")
    assert session.region_name == "us-west-2"
    
    # Test default from config/env (mocked env above sets us-east-1 as default)
    # Note: get_aws_session logic reads config, which might default to 'us-east-1' 
    # if CONFIG not mocked. But basic instantiation should work.
    session_default = get_aws_session()
    assert isinstance(session_default, boto3.Session)

def test_create_s3_bucket_success(s3_client):
    """Test simple bucket creation."""
    bucket_name = "my-test-bucket"
    created = create_s3_bucket(bucket_name)
    
    assert created is True
    # Verify existence
    response = s3_client.list_buckets()
    buckets = [b["Name"] for b in response["Buckets"]]
    assert bucket_name in buckets

def test_create_s3_bucket_existing(s3_client):
    """Test benign handling of existing bucket."""
    bucket_name = "existing-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Should return True without error
    result = create_s3_bucket(bucket_name)
    assert result is True

def test_upload_to_s3(s3_client, temp_file):
    """Test file upload."""
    bucket = "upload-bucket"
    s3_client.create_bucket(Bucket=bucket)
    
    key = "data/test.csv"
    uri = upload_to_s3(temp_file, bucket, key)
    
    assert uri == f"s3://{bucket}/{key}"
    
    # Verify content
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8")
    assert content.replace("\r\n", "\n") == "col1,col2\n1,2"

def test_upload_to_s3_missing_file():
    """Test error on missing local file."""
    with pytest.raises(FileNotFoundError):
        upload_to_s3("non_existent.csv", "bucket", "key")

def test_download_from_s3(s3_client, tmp_path):
    """Test file download."""
    bucket = "download-bucket"
    s3_client.create_bucket(Bucket=bucket)
    key = "remote.txt"
    s3_client.put_object(Bucket=bucket, Key=key, Body="hello world")
    
    local_path = str(tmp_path / "downloaded.txt")
    result = download_from_s3(bucket, key, local_path)
    
    assert result is True
    assert os.path.exists(local_path)
    with open(local_path, "r") as f:
        assert f.read() == "hello world"

def test_list_s3_objects(s3_client):
    """Test listing objects."""
    bucket = "list-bucket"
    s3_client.create_bucket(Bucket=bucket)
    
    keys = ["folder/a.csv", "folder/b.csv", "other/c.csv"]
    for k in keys:
        s3_client.put_object(Bucket=bucket, Key=k, Body="data")
        
    # List all
    all_objs = list_s3_objects(bucket)
    assert len(all_objs) == 3
    assert set(all_objs) == set(keys)
    
    # List with prefix
    folder_objs = list_s3_objects(bucket, prefix="folder/")
    assert len(folder_objs) == 2
    assert "folder/a.csv" in folder_objs
    assert "folder/b.csv" in folder_objs

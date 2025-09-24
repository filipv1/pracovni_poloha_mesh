"""
S3/R2 utility functions for file operations
"""
import boto3
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import logging

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

logger = logging.getLogger(__name__)


class StorageClient:
    """Unified client for R2 or S3 operations"""

    def __init__(self, provider=None):
        self.provider = provider or STORAGE_PROVIDER
        self.client = self._create_client()
        self.bucket_name = self._get_bucket_name()

    def _create_client(self):
        """Create boto3 client for R2 or S3"""
        if self.provider == 'r2':
            return boto3.client(
                's3',
                endpoint_url=R2_ENDPOINT_URL,
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
        else:  # s3
            return boto3.client(
                's3',
                aws_access_key_id=S3_ACCESS_KEY_ID,
                aws_secret_access_key=S3_SECRET_ACCESS_KEY,
                region_name=S3_REGION
            )

    def _get_bucket_name(self):
        """Get bucket name based on provider"""
        return R2_BUCKET_NAME if self.provider == 'r2' else S3_BUCKET_NAME

    def generate_presigned_url(self, key, operation='get', expiry=3600):
        """Generate presigned URL for upload or download

        Args:
            key: S3/R2 object key
            operation: 'get' for download, 'put' for upload
            expiry: URL expiry time in seconds

        Returns:
            Presigned URL string
        """
        try:
            url = self.client.generate_presigned_url(
                ClientMethod=f'{operation}_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiry
            )
            logger.info(f"Generated presigned URL for {operation} {key}")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    def upload_file(self, file_path, key):
        """Upload file to S3/R2

        Args:
            file_path: Local file path
            key: S3/R2 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upload_file(str(file_path), self.bucket_name, key)
            logger.info(f"Uploaded {file_path} to {key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            return False

    def download_file(self, key, file_path):
        """Download file from S3/R2

        Args:
            key: S3/R2 object key
            file_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.client.download_file(self.bucket_name, key, str(file_path))
            logger.info(f"Downloaded {key} to {file_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def upload_json(self, data, key):
        """Upload JSON data to S3/R2

        Args:
            data: Dictionary to upload
            key: S3/R2 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            json_str = json.dumps(data, indent=2)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_str,
                ContentType='application/json'
            )
            logger.info(f"Uploaded JSON to {key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload JSON: {e}")
            return False

    def download_json(self, key):
        """Download and parse JSON from S3/R2

        Args:
            key: S3/R2 object key

        Returns:
            Parsed JSON data or None if failed
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response['Body'].read())
            logger.info(f"Downloaded JSON from {key}")
            return data
        except ClientError as e:
            logger.error(f"Failed to download JSON: {e}")
            return None

    def delete_object(self, key):
        """Delete object from S3/R2

        Args:
            key: S3/R2 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted {key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete object: {e}")
            return False

    def object_exists(self, key):
        """Check if object exists in S3/R2

        Args:
            key: S3/R2 object key

        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False


class JobManager:
    """Manage job status and results in S3/R2"""

    def __init__(self, storage_client=None):
        self.storage = storage_client or StorageClient()

    def create_job_status(self, job_id, initial_status='accepted'):
        """Create initial job status

        Args:
            job_id: Unique job identifier
            initial_status: Initial status string

        Returns:
            Status dictionary
        """
        status = {
            'job_id': job_id,
            'status': initial_status,
            'progress': 0,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'result_key': None,
            'error': None
        }

        status_key = f"{STATUS_PREFIX}{job_id}.json"
        self.storage.upload_json(status, status_key)

        return status

    def update_job_status(self, job_id, **kwargs):
        """Update job status

        Args:
            job_id: Unique job identifier
            **kwargs: Fields to update (status, progress, result_key, error)

        Returns:
            Updated status dictionary
        """
        status_key = f"{STATUS_PREFIX}{job_id}.json"

        # Get current status
        status = self.storage.download_json(status_key)
        if not status:
            status = self.create_job_status(job_id)

        # Update fields
        for key, value in kwargs.items():
            if key in ['status', 'progress', 'result_key', 'error', 'results', 'download_url']:
                status[key] = value

        status['updated_at'] = datetime.utcnow().isoformat()

        # Save updated status
        self.storage.upload_json(status, status_key)

        return status

    def get_job_status(self, job_id):
        """Get current job status

        Args:
            job_id: Unique job identifier

        Returns:
            Status dictionary or None if not found
        """
        status_key = f"{STATUS_PREFIX}{job_id}.json"
        return self.storage.download_json(status_key)

    def cleanup_old_jobs(self, days=7):
        """Clean up old job files

        Args:
            days: Delete jobs older than this many days

        Returns:
            Number of deleted objects
        """
        # This would need listing objects and checking timestamps
        # Simplified for now
        logger.info(f"Cleanup not implemented yet")
        return 0


# Convenience functions
def get_storage_client():
    """Get default storage client"""
    return StorageClient()

def get_job_manager():
    """Get default job manager"""
    return JobManager()
"""
Cloudflare R2 Storage Client
Manages file storage and retrieval using Cloudflare R2 (S3-compatible)
"""
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class R2StorageClient:
    """Client for Cloudflare R2 storage operations"""
    
    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.account_id = account_id
        self.fallback_mode = False
        
        # R2 endpoint URL
        if account_id:
            self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        else:
            self.endpoint_url = None
            
        # Check if credentials have correct format
        if access_key and len(access_key) != 32:
            logger.warning(f"R2 access key has incorrect length ({len(access_key)} instead of 32) - using fallback mode")
            self.fallback_mode = True
            self.s3_client = None
            self.local_storage_path = os.path.join(os.path.dirname(__file__), '..', 'local_storage')
            os.makedirs(self.local_storage_path, exist_ok=True)
            return
            
        # Initialize S3 client for R2
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    signature_version='s3v4',
                    retries={'max_attempts': 3}
                ),
                region_name='auto'
            )
            
            # Verify bucket exists
            self._ensure_bucket_exists()
            
        except Exception as e:
            logger.error(f"Failed to initialize R2 client: {e} - using fallback mode")
            self.s3_client = None
            self.fallback_mode = True
            self.local_storage_path = os.path.join(os.path.dirname(__file__), '..', 'local_storage')
            os.makedirs(self.local_storage_path, exist_ok=True)
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if not"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                # Create bucket
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created bucket {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
            else:
                logger.error(f"Error checking bucket: {e}")
    
    def upload_file(self, file_path: str, key: str, metadata: Dict = None, expires_days: int = 7) -> Tuple[bool, str]:
        """Upload file to R2
        Returns (success, url_or_error)"""
        if self.fallback_mode:
            # Use local storage in fallback mode
            try:
                import shutil
                local_key_path = os.path.join(self.local_storage_path, key.replace('/', os.sep))
                os.makedirs(os.path.dirname(local_key_path), exist_ok=True)
                shutil.copy2(file_path, local_key_path)
                logger.info(f"[FALLBACK] File stored locally: {local_key_path}")
                return True, f"local://{local_key_path}"
            except Exception as e:
                logger.error(f"[FALLBACK] Failed to store file locally: {e}")
                return False, str(e)
        
        if not self.s3_client:
            return False, "Storage client not initialized"
            
        try:
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata['upload_date'] = datetime.utcnow().isoformat()
            file_metadata['expires_date'] = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Upload file with progress callback
            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=f,
                    Metadata=file_metadata,
                    ContentType=self._get_content_type(file_path)
                )
            
            # Generate public URL (R2 URLs are public by default if bucket is public)
            url = self.get_public_url(key)
            
            logger.info(f"Uploaded {file_path} to R2 as {key} ({file_size} bytes)")
            return True, url
            
        except FileNotFoundError:
            error = f"File not found: {file_path}"
            logger.error(error)
            return False, error
        except Exception as e:
            error = f"Upload failed: {str(e)}"
            logger.error(error)
            return False, error
    
    def download_file(self, key: str, destination: str) -> Tuple[bool, str]:
        """Download file from R2
        Returns (success, message)"""
        if self.fallback_mode:
            # Use local storage in fallback mode
            try:
                import shutil
                local_key_path = os.path.join(self.local_storage_path, key.replace('/', os.sep))
                if os.path.exists(local_key_path):
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copy2(local_key_path, destination)
                    logger.info(f"[FALLBACK] File retrieved from local storage: {local_key_path}")
                    return True, "Download successful"
                else:
                    return False, f"File not found in local storage: {key}"
            except Exception as e:
                logger.error(f"[FALLBACK] Failed to retrieve file: {e}")
                return False, str(e)
        
        if not self.s3_client:
            return False, "Storage client not initialized"
            
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(self.bucket_name, key, destination)
            
            logger.info(f"Downloaded {key} from R2 to {destination}")
            return True, "Download successful"
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                error = f"File not found: {key}"
            else:
                error = f"Download failed: {str(e)}"
            logger.error(error)
            return False, error
        except Exception as e:
            error = f"Download failed: {str(e)}"
            logger.error(error)
            return False, error
    
    def delete_file(self, key: str) -> Tuple[bool, str]:
        """Delete file from R2
        Returns (success, message)"""
        if not self.s3_client:
            return False, "Storage client not initialized"
            
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted {key} from R2")
            return True, "File deleted"
        except Exception as e:
            error = f"Deletion failed: {str(e)}"
            logger.error(error)
            return False, error
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists in R2"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def get_file_info(self, key: str) -> Optional[Dict]:
        """Get file metadata from R2"""
        if not self.s3_client:
            return None
            
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
        except ClientError:
            return None
    
    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned URL for temporary access
        expires_in: URL expiration time in seconds (default 1 hour)"""
        if self.fallback_mode:
            # Return local file URL in fallback mode
            local_key_path = os.path.join(self.local_storage_path, key.replace('/', os.sep))
            if os.path.exists(local_key_path):
                return f"file:///{os.path.abspath(local_key_path)}"
            return None
        
        if not self.s3_client:
            return None
            
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def get_public_url(self, key: str) -> str:
        """Get public URL for a file (if bucket is public)"""
        if self.endpoint_url:
            return f"{self.endpoint_url}/{self.bucket_name}/{key}"
        else:
            # Fallback to standard R2 public URL format
            return f"https://pub-{self.account_id}.r2.dev/{key}"
    
    def list_files(self, prefix: str = "", max_keys: int = 1000) -> list:
        """List files in bucket with optional prefix filter"""
        if not self.s3_client:
            return []
            
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
            
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def setup_lifecycle_rules(self, retention_days: int = 7):
        """Setup automatic deletion after specified days"""
        if not self.s3_client:
            return False
            
        try:
            lifecycle_config = {
                'Rules': [{
                    'ID': f'delete-after-{retention_days}-days',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'Expiration': {
                        'Days': retention_days
                    }
                }]
            }
            
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            
            logger.info(f"Set up lifecycle rule: delete files after {retention_days} days")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup lifecycle rules: {e}")
            return False
    
    def cleanup_expired_files(self):
        """Manually cleanup expired files based on metadata"""
        if not self.s3_client:
            return
            
        try:
            files = self.list_files()
            current_time = datetime.utcnow()
            
            for file_info in files:
                key = file_info['key']
                metadata = self.get_file_info(key)
                
                if metadata and 'metadata' in metadata:
                    expires_date_str = metadata['metadata'].get('expires_date')
                    if expires_date_str:
                        expires_date = datetime.fromisoformat(expires_date_str)
                        if current_time > expires_date:
                            self.delete_file(key)
                            logger.info(f"Deleted expired file: {key}")
                            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.pkl': 'application/octet-stream',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.mp4': 'video/mp4',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.csv': 'text/csv'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def get_storage_usage(self) -> Dict:
        """Get storage usage statistics"""
        if not self.s3_client:
            return {'total_files': 0, 'total_size': 0}
            
        try:
            files = self.list_files()
            total_size = sum(f['size'] for f in files)
            
            return {
                'total_files': len(files),
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {'total_files': 0, 'total_size': 0}
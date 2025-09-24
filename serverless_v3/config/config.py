"""
Configuration for CloudFlare R2 and RunPod V3 Architecture
"""
import os

# CloudFlare R2 Configuration
R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID', '')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID', '')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY', '')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME', 'ergonomic-analysis')
R2_ENDPOINT_URL = f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'

# Alternative: AWS S3 Configuration (fallback)
S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID', '')
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY', '')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'ergonomic-analysis')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')

# Storage provider selection
STORAGE_PROVIDER = os.getenv('STORAGE_PROVIDER', 'r2')  # 'r2' or 's3'

# Job configuration
JOB_STATUS_TTL = 86400  # 24 hours in seconds
UPLOAD_URL_EXPIRY = 3600  # 1 hour in seconds
DOWNLOAD_URL_EXPIRY = 86400  # 24 hours in seconds

# File paths in bucket
UPLOADS_PREFIX = 'uploads/'
RESULTS_PREFIX = 'results/'
STATUS_PREFIX = 'status/'

# Processing configuration
MAX_VIDEO_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
DEFAULT_QUALITY = 'medium'

# RunPod configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY', '')
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID', '')
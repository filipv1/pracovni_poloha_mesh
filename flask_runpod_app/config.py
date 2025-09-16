"""
Flask Application Configuration
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Upload
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE_MB', 5120)) * 1024 * 1024  # 5GB default
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'.mp4'}
    MAX_VIDEO_DURATION_SECONDS = int(os.environ.get('MAX_VIDEO_DURATION_SECONDS', 1800))  # 30 minutes
    
    # RunPod Configuration
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
    RUNPOD_POD_ID = os.environ.get('RUNPOD_POD_ID')
    RUNPOD_POD_TEMPLATE = os.environ.get('RUNPOD_POD_TEMPLATE')
    POD_IDLE_TIMEOUT_SECONDS = int(os.environ.get('POD_IDLE_TIMEOUT_SECONDS', 300))  # 5 minutes
    
    # Cloudflare R2 Configuration
    R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
    R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
    R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
    R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'pose-analysis-files')
    R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
    
    # Email Configuration
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD')
    EMAIL_FROM = os.environ.get('EMAIL_FROM', os.environ.get('SMTP_USERNAME'))
    
    # Application Settings
    JOB_RETRY_LIMIT = int(os.environ.get('JOB_RETRY_LIMIT', 10))
    FILE_RETENTION_DAYS = int(os.environ.get('FILE_RETENTION_DAYS', 7))
    
    @staticmethod
    def init_app(app):
        # Create upload folder if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
"""
Database Models for Flask Application
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    jobs = db.relationship('Job', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Job(db.Model):
    """Job model for processing queue"""
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='queued')  # queued|processing|completed|failed
    video_filename = db.Column(db.String(255), nullable=False)
    video_size_mb = db.Column(db.Float)
    total_frames = db.Column(db.Integer)
    processed_frames = db.Column(db.Integer, default=0)
    progress_percent = db.Column(db.Float, default=0)
    processing_stage = db.Column(db.String(50))  # uploading|mediapipe|smplx|angles|downloading
    time_elapsed_seconds = db.Column(db.Float)
    time_remaining_seconds = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    retry_count = db.Column(db.Integer, default=0)
    
    # Relationships
    files = db.relationship('File', backref='job', lazy='dynamic', cascade='all, delete-orphan')
    logs = db.relationship('Log', backref='job', lazy='dynamic', cascade='all, delete-orphan')
    usage_stats = db.relationship('UsageStats', backref='job', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'status': self.status,
            'video_filename': self.video_filename,
            'video_size_mb': self.video_size_mb,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'progress_percent': self.progress_percent,
            'processing_stage': self.processing_stage,
            'time_elapsed_seconds': self.time_elapsed_seconds,
            'time_remaining_seconds': self.time_remaining_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


class File(db.Model):
    """File model for output tracking"""
    __tablename__ = 'files'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # pkl|xlsx
    filename = db.Column(db.String(255), nullable=False)
    r2_key = db.Column(db.String(255), nullable=False)
    r2_url = db.Column(db.Text, nullable=False)
    size_mb = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'file_type': self.file_type,
            'filename': self.filename,
            'r2_key': self.r2_key,
            'r2_url': self.r2_url,
            'size_mb': self.size_mb,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


class Log(db.Model):
    """Log model for debugging"""
    __tablename__ = 'logs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=True)
    level = db.Column(db.String(20), nullable=False)  # info|warning|error
    message = db.Column(db.Text, nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    @staticmethod
    def create(job_id, level, message, details=None):
        """Helper method to create a log entry"""
        log = Log(job_id=job_id, level=level, message=message, details=details)
        db.session.add(log)
        db.session.commit()
        return log
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'level': self.level,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class UsageStats(db.Model):
    """RunPod usage tracking"""
    __tablename__ = 'usage_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    pod_id = db.Column(db.String(100))
    gpu_type = db.Column(db.String(50), default='A5000')
    processing_time_seconds = db.Column(db.Float)
    cost_usd = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'pod_id': self.pod_id,
            'gpu_type': self.gpu_type,
            'processing_time_seconds': self.processing_time_seconds,
            'cost_usd': self.cost_usd,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
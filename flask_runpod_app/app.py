"""
Flask RunPod Application
Main application file that orchestrates the pose analysis pipeline
"""
import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# Import configuration and models
from config import Config
from models import db, User, Job, File, Log, UsageStats
from auth import init_auth, authenticate_user, load_user, AuthUser

# Import core components
from core.runpod_client import RunPodClient
from core.storage_client import R2StorageClient
from core.job_processor import JobProcessor
from core.email_service import EmailService
from core.progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user_callback(user_id):
    """Callback for Flask-Login to load user"""
    return load_user(user_id)

# Initialize services (will be done after app context is available)
runpod_client = None
storage_client = None
job_processor = None
email_service = None
progress_tracker = None

def init_services():
    """Initialize all services"""
    global runpod_client, storage_client, job_processor, email_service, progress_tracker
    
    # Initialize RunPod client if configured
    if app.config.get('RUNPOD_API_KEY') and app.config.get('RUNPOD_POD_ID'):
        runpod_client = RunPodClient(
            app.config['RUNPOD_API_KEY'],
            app.config['RUNPOD_POD_ID']
        )
        logger.info("RunPod client initialized")
    else:
        logger.warning("RunPod not configured - using local processing mode")
    
    # Initialize R2 storage client if configured
    if all([app.config.get('R2_ACCOUNT_ID'), 
            app.config.get('R2_ACCESS_KEY_ID'),
            app.config.get('R2_SECRET_ACCESS_KEY')]):
        storage_client = R2StorageClient(
            app.config['R2_ACCOUNT_ID'],
            app.config['R2_ACCESS_KEY_ID'],
            app.config['R2_SECRET_ACCESS_KEY'],
            app.config['R2_BUCKET_NAME']
        )
        logger.info("R2 storage client initialized")
    else:
        logger.warning("R2 storage not configured - using local storage")
    
    # Initialize email service if configured
    if app.config.get('SMTP_USERNAME') and app.config.get('SMTP_PASSWORD'):
        email_service = EmailService(
            app.config['SMTP_SERVER'],
            app.config['SMTP_PORT'],
            app.config['SMTP_USERNAME'],
            app.config['SMTP_PASSWORD'],
            app.config['EMAIL_FROM']
        )
        logger.info("Email service initialized")
    else:
        logger.warning("Email service not configured")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    logger.info("Progress tracker initialized")
    
    # Initialize job processor
    job_processor = JobProcessor(app, runpod_client, storage_client, email_service)
    job_processor.start()
    logger.info("Job processor started")

# Routes

@app.route('/')
def index():
    """Home page - redirect to upload or login"""
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = authenticate_user(username, password)
        if user:
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('upload'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/upload')
@login_required
def upload():
    """Video upload page"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    """Handle video upload and create job"""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        filename = secure_filename(video.filename)
        if not filename.lower().endswith('.mp4'):
            return jsonify({'error': 'Only MP4 files are allowed'}), 400
        
        # Check file size
        video.seek(0, os.SEEK_END)
        size_bytes = video.tell()
        size_mb = size_bytes / (1024 * 1024)
        video.seek(0)
        
        max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        if size_mb > max_size_mb:
            return jsonify({'error': f'File too large. Maximum size is {max_size_mb:.0f} MB'}), 400
        
        # Generate unique filename and save
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"job_pending_{unique_filename}")
        video.save(upload_path)
        
        # Create job record
        job = Job(
            user_id=current_user.id,
            video_filename=filename,
            video_size_mb=round(size_mb, 2),
            status='queued'
        )
        db.session.add(job)
        db.session.commit()
        
        # Rename file with job ID
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job.id}_input.mp4")
        os.rename(upload_path, final_path)
        
        # Add job to processing queue
        job_processor.add_job(job.id)
        
        # Log the upload
        Log.create(job.id, 'info', f'Video uploaded: {filename} ({size_mb:.1f} MB)')
        
        return jsonify({
            'success': True,
            'job_id': job.id,
            'redirect': url_for('progress', job_id=job.id)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<int:job_id>')
@login_required
def progress(job_id):
    """Progress tracking page"""
    job = Job.query.get_or_404(job_id)
    
    # Check if user owns this job
    if job.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    return render_template('progress.html', job=job)

@app.route('/api/progress/<int:job_id>')
@login_required
def api_progress_stream(job_id):
    """SSE endpoint for real-time progress updates"""
    job = Job.query.get_or_404(job_id)
    
    # Check if user owns this job
    if job.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    return progress_tracker.create_sse_stream(job_id, app, db)

@app.route('/api/job/<int:job_id>')
@login_required
def api_job_status(job_id):
    """Get job status and details"""
    job = Job.query.get_or_404(job_id)
    
    # Check if user owns this job
    if job.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    return jsonify(job.to_dict())

@app.route('/api/job/<int:job_id>/files')
@login_required
def api_job_files(job_id):
    """Get files for a completed job"""
    job = Job.query.get_or_404(job_id)
    
    # Check if user owns this job
    if job.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    files = File.query.filter_by(job_id=job_id).all()
    return jsonify([f.to_dict() for f in files])

@app.route('/history')
@login_required
def history():
    """Job history page"""
    # Get user's jobs
    jobs = Job.query.filter_by(user_id=current_user.id)\
                    .order_by(Job.created_at.desc())\
                    .limit(50)\
                    .all()
    
    return render_template('history.html', jobs=jobs)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard with statistics"""
    if not current_user.is_admin:
        abort(403)
    
    # Calculate statistics
    stats = {
        'total_users': User.query.count(),
        'total_jobs': Job.query.count(),
        'queued_jobs': Job.query.filter_by(status='queued').count(),
        'processing_jobs': Job.query.filter_by(status='processing').count(),
        'completed_jobs': Job.query.filter_by(status='completed').count(),
        'failed_jobs': Job.query.filter_by(status='failed').count(),
        'total_processing_time': db.session.query(db.func.sum(Job.time_elapsed_seconds)).scalar() or 0,
        'total_cost': db.session.query(db.func.sum(UsageStats.cost_usd)).scalar() or 0,
        'active_connections': progress_tracker.get_active_connections_count() if progress_tracker else 0
    }
    
    # Get recent jobs
    recent_jobs = Job.query.order_by(Job.created_at.desc()).limit(20).all()
    
    # Get storage usage
    storage_stats = storage_client.get_storage_usage() if storage_client else {'total_files': 0, 'total_size_gb': 0}
    
    return render_template('admin_dashboard.html', 
                         stats=stats, 
                         recent_jobs=recent_jobs,
                         storage_stats=storage_stats)

@app.route('/admin/logs')
@login_required
def admin_logs():
    """View system logs"""
    if not current_user.is_admin:
        abort(403)
    
    # Get logs with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    logs = Log.query.order_by(Log.timestamp.desc())\
                    .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin_logs.html', logs=logs)

@app.route('/api/test/email')
@login_required
def test_email():
    """Test email configuration"""
    if not current_user.is_admin:
        abort(403)
    
    if email_service:
        success = email_service.send_test_email(current_user.email)
        if success:
            return jsonify({'success': True, 'message': 'Test email sent'})
        else:
            return jsonify({'success': False, 'message': 'Failed to send email'})
    else:
        return jsonify({'success': False, 'message': 'Email service not configured'})

@app.route('/api/test/runpod')
@login_required
def test_runpod():
    """Test RunPod connection"""
    if not current_user.is_admin:
        abort(403)
    
    if runpod_client:
        status = runpod_client.get_pod_status()
        return jsonify(status)
    else:
        return jsonify({'error': 'RunPod not configured'})

@app.route('/api/test/storage')
@login_required
def test_storage():
    """Test R2 storage connection"""
    if not current_user.is_admin:
        abort(403)
    
    if storage_client:
        usage = storage_client.get_storage_usage()
        return jsonify(usage)
    else:
        return jsonify({'error': 'R2 storage not configured'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': True,
            'runpod': runpod_client is not None,
            'storage': storage_client is not None,
            'email': email_service is not None,
            'job_processor': job_processor is not None and job_processor.processing
        }
    })

# Error handlers

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('error.html', 
                         title='Page Not Found',
                         message='The page you are looking for does not exist.'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('error.html',
                         title='Server Error',
                         message='An internal error occurred. Please try again later.'), 500

# CLI commands

@app.cli.command('init-db')
def init_db_command():
    """Initialize the database"""
    initialize_app()
    print('Database initialized!')

@app.cli.command('test-services')
def test_services_command():
    """Test all services"""
    print('Testing services...')
    
    # Test database
    try:
        User.query.first()
        print('✓ Database connection OK')
    except Exception as e:
        print(f'✗ Database error: {e}')
    
    # Test RunPod
    if runpod_client:
        status = runpod_client.get_pod_status()
        print(f"✓ RunPod status: {status.get('status', 'Unknown')}")
    else:
        print('✗ RunPod not configured')
    
    # Test R2
    if storage_client:
        usage = storage_client.get_storage_usage()
        print(f"✓ R2 storage: {usage['total_files']} files, {usage['total_size_gb']:.2f} GB")
    else:
        print('✗ R2 storage not configured')
    
    # Test email
    if email_service:
        print('✓ Email service configured')
    else:
        print('✗ Email service not configured')

# Application initialization

def initialize_app():
    """Initialize application"""
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Initialize authentication
        init_auth(app)
        
        # Initialize services
        init_services()
        
        # Setup storage lifecycle rules if configured
        if storage_client:
            storage_client.setup_lifecycle_rules(app.config['FILE_RETENTION_DAYS'])
        
        logger.info("Application initialized successfully")

# Main entry point

if __name__ == '__main__':
    # Initialize app before running
    initialize_app()
    
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)
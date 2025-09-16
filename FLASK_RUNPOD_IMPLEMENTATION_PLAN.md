# Flask RunPod Application - Implementation Plan

## 📌 Project Overview

This document provides a comprehensive implementation plan for building a Flask web application that wraps the advanced 3D pose analysis pipeline with on-demand RunPod GPU processing. The application allows users to upload videos, processes them through the SMPL-X pipeline on RunPod GPUs, and returns PKL mesh data and comprehensive XLSX ergonomic analysis.

## 🎯 Core Requirements

- **Input**: MP4 video files (max 5GB, 30 minutes)
- **Processing**: MediaPipe → SMPL-X fitting → Skin-based angle calculation → Ergonomic analysis
- **Output**: PKL file (mesh data) + XLSX file (comprehensive ergonomic analysis)
- **Users**: 10 internal users, basic auth, FIFO queue
- **Infrastructure**: Free-tier hosting, RunPod A5000 GPU, 7-day file retention
- **Features**: Real-time progress tracking, email notifications, job history

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Browser   │────▶│   Flask App      │────▶│  RunPod GPU     │
│   (User)    │◀────│   (Render.com)   │◀────│  (A5000 Pod)    │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │                          │
                            ▼                          ▼
                    ┌──────────────┐         ┌──────────────────┐
                    │  SQLite DB   │         │  Cloudflare R2   │
                    │  (Jobs/Logs) │         │  (File Storage)  │
                    └──────────────┘         └──────────────────┘
```

## 📋 Technology Stack

### Hosting Platform: **Render.com**
- **Free Tier**: 750 hours/month
- **Why**: Best free Flask hosting with persistent disk for SQLite
- **Features**: Background workers, environment variables, custom domains
- **Alternative considered**: Vercel (complex for Flask), Railway (limited free), PythonAnywhere (restrictive)

### Storage: **Cloudflare R2**
- **Free Tier**: 10GB storage, 1M Class A operations, 10M Class B operations
- **Critical Advantage**: NO egress fees (unlimited free downloads)
- **Features**: S3-compatible API, lifecycle rules for 7-day auto-deletion
- **Alternative considered**: AWS S3 (egress fees), Backblaze B2 (10GB limit with egress)

### GPU Compute: **RunPod**
- **GPU**: A5000 (~$0.79/hour)
- **Mode**: Persistent pod with 5-minute idle timeout
- **Environment**: Pre-configured conda `mesh_pipeline`
- **Storage**: SMPL-X models pre-loaded

### Database: **SQLite**
- Local file-based database
- Perfect for low-volume internal application
- No external database service needed

### Real-time Updates: **Server-Sent Events (SSE)**
- Simpler than WebSockets
- Native browser support
- Perfect for progress updates

### Email: **Gmail SMTP or SendGrid Free**
- Gmail: 500 emails/day free
- SendGrid: 100 emails/day free forever

## 📁 Project Structure

```
flask_runpod_app/
├── app.py                      # Main Flask application
├── config.py                   # Configuration management
├── models.py                   # SQLite database models
├── auth.py                     # User authentication (adapted from testw4)
├── 
├── core/
│   ├── __init__.py
│   ├── runpod_client.py      # RunPod API wrapper
│   ├── storage_client.py     # Cloudflare R2 operations
│   ├── job_processor.py      # Background job processing
│   ├── email_service.py      # Email notifications
│   └── progress_tracker.py   # SSE progress updates
│
├── static/
│   ├── css/
│   │   └── style.css          # UI styling (from testw4)
│   ├── js/
│   │   ├── upload.js          # Drag-drop upload
│   │   └── progress.js        # SSE progress display
│   └── favicon.ico
│
├── templates/
│   ├── base.html              # Base template with navigation
│   ├── login.html             # User login (from testw4)
│   ├── upload.html            # Video upload interface
│   ├── progress.html          # Real-time progress display
│   ├── history.html           # Job history with downloads
│   └── error.html             # Error display with logs
│
├── runpod_scripts/            # Scripts that run on RunPod
│   ├── process_video.py       # Main processing orchestrator
│   └── setup_environment.sh   # Environment setup script
│
├── migrations/
│   └── init_db.py            # Database initialization
│
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore
├── render.yaml               # Render.com deployment config
├── README.md                 # Setup and deployment guide
└── tests/
    └── test_pipeline.py      # Basic tests
```

## 💾 Database Schema

```sql
-- Users table (from testw4, hardcoded credentials)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Jobs table for processing queue
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    status TEXT DEFAULT 'queued', -- queued|processing|completed|failed
    video_filename TEXT NOT NULL,
    video_size_mb REAL,
    total_frames INTEGER,
    processed_frames INTEGER DEFAULT 0,
    progress_percent REAL DEFAULT 0,
    processing_stage TEXT, -- uploading|mediapipe|smplx|angles|downloading
    time_elapsed_seconds REAL,
    time_remaining_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Files table for output tracking
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    file_type TEXT NOT NULL, -- pkl|xlsx
    filename TEXT NOT NULL,
    r2_key TEXT NOT NULL,
    r2_url TEXT NOT NULL,
    size_mb REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs (id)
);

-- Logs table for debugging
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    level TEXT NOT NULL, -- info|warning|error
    message TEXT NOT NULL,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs (id)
);

-- RunPod usage tracking
CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    pod_id TEXT,
    gpu_type TEXT DEFAULT 'A5000',
    processing_time_seconds REAL,
    cost_usd REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs (id)
);
```

## 📝 Implementation Phases

### Phase 1: Foundation Setup (Day 1-2)

#### 1.1 Project Initialization
```bash
# Create GitHub repository
git init flask_runpod_app
cd flask_runpod_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Initialize Flask project structure
mkdir -p app/core app/static/css app/static/js app/templates
mkdir -p runpod_scripts migrations tests
```

#### 1.2 Copy and Adapt from testw4
- [ ] Copy authentication system from testw4
- [ ] Copy email templates and adapt
- [ ] Copy CSS styles and UI components
- [ ] Adapt user management for 10 hardcoded users

#### 1.3 Environment Configuration
```python
# .env.example
FLASK_SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///app.db

# RunPod Configuration
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_POD_ID=your-existing-pod-id
RUNPOD_POD_TEMPLATE=your-template-id

# Cloudflare R2 Configuration
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=pose-analysis-files
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com

# Email Configuration (Gmail)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com

# Application Settings
MAX_UPLOAD_SIZE_MB=5120  # 5GB
MAX_VIDEO_DURATION_SECONDS=1800  # 30 minutes
JOB_RETRY_LIMIT=10
POD_IDLE_TIMEOUT_SECONDS=300  # 5 minutes
FILE_RETENTION_DAYS=7
```

### Phase 2: RunPod Integration (Day 2-3)

#### 2.1 RunPod Client Implementation
```python
# core/runpod_client.py
import runpod
import time
import json
from typing import Dict, Optional

class RunPodClient:
    def __init__(self, api_key: str, pod_id: str):
        self.api_key = api_key
        self.pod_id = pod_id
        runpod.api_key = api_key
        self.pod = None
        self.last_activity = None
        
    def ensure_pod_running(self) -> bool:
        """Start pod if not running, return True when ready"""
        pod_status = runpod.get_pod(self.pod_id)
        
        if pod_status['status'] != 'RUNNING':
            runpod.start_pod(self.pod_id)
            # Wait for pod to be ready
            for _ in range(60):  # 5 minute timeout
                time.sleep(5)
                status = runpod.get_pod(self.pod_id)
                if status['status'] == 'RUNNING':
                    self.pod = status
                    break
                    
        self.last_activity = time.time()
        return self.pod is not None
        
    def execute_processing(self, video_path: str, job_id: int) -> Dict:
        """Execute the processing pipeline on RunPod"""
        command = f"""
        cd /workspace/pracovni_poloha_mesh
        conda activate mesh_pipeline
        python run_production_simple.py {video_path} --output_dir /tmp/job_{job_id}
        python create_combined_angles_csv_skin.py /tmp/job_{job_id}/output_meshes.pkl /tmp/job_{job_id}/skin_angles.csv
        python ergonomic_time_analysis.py /tmp/job_{job_id}/skin_angles.csv /tmp/job_{job_id}/analysis.xlsx
        """
        
        result = runpod.run_pod_command(self.pod_id, command)
        self.last_activity = time.time()
        return result
        
    def check_idle_timeout(self):
        """Shutdown pod if idle for too long"""
        if self.last_activity and (time.time() - self.last_activity) > 300:
            runpod.stop_pod(self.pod_id)
            self.pod = None
```

#### 2.2 Processing Script for RunPod
```python
# runpod_scripts/process_video.py
#!/usr/bin/env python
"""
Main processing script that runs on RunPod GPU
Orchestrates the entire pipeline and reports progress
"""
import sys
import os
import json
import time
from pathlib import Path

def report_progress(stage, percent, message):
    """Send progress updates that Flask can parse"""
    print(f"PROGRESS|{stage}|{percent}|{message}", flush=True)

def main(video_path, output_dir, job_id):
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Stage 1: MediaPipe and SMPL-X fitting
        report_progress("processing", 10, "Starting MediaPipe detection...")
        
        import sys
        sys.path.append('/workspace/pracovni_poloha_mesh')
        
        # Import after path setup
        from run_production_simple import main as run_production
        from create_combined_angles_csv_skin import create_combined_angles_csv_skin
        from ergonomic_time_analysis import main as ergonomic_analysis
        
        # Run production pipeline
        report_progress("processing", 20, "Running SMPL-X fitting...")
        pkl_path = run_production(video_path, str(output_dir))
        
        # Calculate angles
        report_progress("processing", 60, "Calculating skin-based angles...")
        csv_path = output_dir / "skin_angles.csv"
        create_combined_angles_csv_skin(pkl_path, str(csv_path))
        
        # Generate ergonomic analysis
        report_progress("processing", 80, "Generating ergonomic analysis...")
        xlsx_path = output_dir / "ergonomic_analysis.xlsx"
        ergonomic_analysis(str(csv_path), str(xlsx_path))
        
        # Report success
        report_progress("completed", 100, "Processing completed successfully")
        
        result = {
            "status": "success",
            "pkl_path": str(pkl_path),
            "xlsx_path": str(xlsx_path)
        }
        
        print(f"RESULT|{json.dumps(result)}", flush=True)
        
    except Exception as e:
        report_progress("failed", 0, str(e))
        print(f"ERROR|{str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: process_video.py <video_path> <output_dir> <job_id>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2], sys.argv[3])
```

### Phase 3: Cloudflare R2 Integration (Day 3-4)

#### 3.1 Storage Client Implementation
```python
# core/storage_client.py
import boto3
from botocore.config import Config
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

class R2StorageClient:
    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        
    def upload_file(self, file_path: str, key: str, expires_days: int = 7) -> str:
        """Upload file to R2 and return public URL"""
        # Upload file
        with open(file_path, 'rb') as f:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=f,
                Metadata={
                    'expires': (datetime.now() + timedelta(days=expires_days)).isoformat()
                }
            )
        
        # Generate public URL
        url = f"{self.endpoint_url}/{self.bucket_name}/{key}"
        return url
        
    def download_file(self, key: str, destination: str) -> bool:
        """Download file from R2"""
        try:
            self.s3_client.download_file(self.bucket_name, key, destination)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
            
    def delete_file(self, key: str) -> bool:
        """Delete file from R2"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            print(f"Deletion failed: {e}")
            return False
            
    def setup_lifecycle_rules(self):
        """Setup automatic deletion after 7 days"""
        lifecycle_config = {
            'Rules': [{
                'ID': 'delete-after-7-days',
                'Status': 'Enabled',
                'Expiration': {
                    'Days': 7
                }
            }]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
```

### Phase 4: Job Queue System (Day 4-5)

#### 4.1 Job Processor Implementation
```python
# core/job_processor.py
import threading
import queue
import time
import traceback
from typing import Optional
import re

class JobProcessor:
    def __init__(self, app, runpod_client, storage_client, email_service):
        self.app = app
        self.runpod_client = runpod_client
        self.storage_client = storage_client
        self.email_service = email_service
        self.job_queue = queue.Queue()
        self.processing = False
        self.current_job = None
        
    def start(self):
        """Start the background job processor"""
        self.processing = True
        thread = threading.Thread(target=self._process_jobs, daemon=True)
        thread.start()
        
    def add_job(self, job_id: int):
        """Add a job to the processing queue"""
        self.job_queue.put(job_id)
        
    def _process_jobs(self):
        """Main job processing loop"""
        while self.processing:
            try:
                # Get next job from queue
                if not self.job_queue.empty():
                    job_id = self.job_queue.get()
                    self._process_single_job(job_id)
                else:
                    # Check for idle timeout
                    self.runpod_client.check_idle_timeout()
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Job processor error: {e}")
                traceback.print_exc()
                
    def _process_single_job(self, job_id: int):
        """Process a single job with retry logic"""
        with self.app.app_context():
            from models import Job, File, Log, db
            
            job = Job.query.get(job_id)
            if not job:
                return
                
            # Update job status
            job.status = 'processing'
            job.started_at = datetime.now()
            db.session.commit()
            
            # Retry logic
            for attempt in range(job.retry_count, 10):
                try:
                    # Ensure pod is running
                    self._update_progress(job_id, 'starting', 5, 'Starting RunPod GPU...')
                    if not self.runpod_client.ensure_pod_running():
                        raise Exception("Failed to start RunPod")
                    
                    # Upload video to pod
                    self._update_progress(job_id, 'uploading', 10, 'Uploading video to GPU...')
                    video_path = self._upload_video_to_pod(job)
                    
                    # Execute processing
                    self._update_progress(job_id, 'processing', 20, 'Processing video...')
                    result = self._execute_and_monitor(job_id, video_path)
                    
                    # Download results
                    self._update_progress(job_id, 'downloading', 90, 'Downloading results...')
                    files = self._download_and_store_results(job_id, result)
                    
                    # Mark as completed
                    job.status = 'completed'
                    job.completed_at = datetime.now()
                    job.progress_percent = 100
                    db.session.commit()
                    
                    # Send email notification
                    self.email_service.send_completion_email(job, files)
                    
                    # Success - exit retry loop
                    break
                    
                except Exception as e:
                    job.retry_count = attempt + 1
                    job.error_message = str(e)
                    
                    Log.create(job_id, 'error', f"Attempt {attempt + 1} failed: {e}")
                    
                    if attempt >= 9:
                        job.status = 'failed'
                        db.session.commit()
                        self.email_service.send_failure_email(job, str(e))
                    else:
                        # Exponential backoff
                        time.sleep(2 ** attempt)
                        
    def _execute_and_monitor(self, job_id: int, video_path: str) -> dict:
        """Execute processing and monitor progress"""
        import subprocess
        
        # Run processing on pod
        command = f"python /workspace/runpod_scripts/process_video.py {video_path} /tmp/job_{job_id} {job_id}"
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor output for progress
        for line in process.stdout:
            if line.startswith("PROGRESS|"):
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    stage, percent, message = parts[1], float(parts[2]), parts[3]
                    self._update_progress(job_id, stage, percent, message)
                    
            elif line.startswith("RESULT|"):
                result_json = line.strip().split("|", 1)[1]
                return json.loads(result_json)
                
            elif line.startswith("ERROR|"):
                error_msg = line.strip().split("|", 1)[1]
                raise Exception(f"Processing error: {error_msg}")
                
        process.wait()
        if process.returncode != 0:
            raise Exception(f"Processing failed with return code {process.returncode}")
            
    def _update_progress(self, job_id: int, stage: str, percent: float, message: str):
        """Update job progress in database and notify SSE"""
        from models import Job, db
        
        job = Job.query.get(job_id)
        if job:
            job.processing_stage = stage
            job.progress_percent = percent
            
            # Calculate time estimates
            if job.started_at and percent > 0:
                elapsed = (datetime.now() - job.started_at).total_seconds()
                job.time_elapsed_seconds = elapsed
                if percent < 100:
                    job.time_remaining_seconds = (elapsed / percent) * (100 - percent)
                    
            db.session.commit()
            
            # Send SSE update
            self._send_sse_update(job_id, {
                'stage': stage,
                'percent': percent,
                'message': message,
                'elapsed': job.time_elapsed_seconds,
                'remaining': job.time_remaining_seconds
            })
```

### Phase 5: User Interface (Day 5-6)

#### 5.1 Upload Interface
```html
<!-- templates/upload.html -->
{% extends "base.html" %}
{% block content %}
<div class="upload-container">
    <h2>Upload Video for Analysis</h2>
    
    <div id="drop-zone" class="drop-zone">
        <p>Drag & drop your video here or click to browse</p>
        <input type="file" id="file-input" accept="video/mp4" style="display: none;">
        <div id="file-info" style="display: none;">
            <p>Selected: <span id="file-name"></span></p>
            <p>Size: <span id="file-size"></span> MB</p>
            <button id="upload-btn" class="btn-primary">Start Processing</button>
        </div>
    </div>
    
    <div class="requirements">
        <h3>Requirements:</h3>
        <ul>
            <li>Format: MP4</li>
            <li>Max size: 5 GB</li>
            <li>Max duration: 30 minutes</li>
            <li>Processing: Ultra quality, skin-based analysis</li>
        </ul>
    </div>
</div>

<script src="{{ url_for('static', filename='js/upload.js') }}"></script>
{% endblock %}
```

#### 5.2 Progress Display with SSE
```javascript
// static/js/progress.js
class ProgressTracker {
    constructor(jobId) {
        this.jobId = jobId;
        this.eventSource = null;
        this.startTracking();
    }
    
    startTracking() {
        this.eventSource = new EventSource(`/api/progress/${this.jobId}`);
        
        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateUI(data);
        };
        
        this.eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            this.eventSource.close();
        };
    }
    
    updateUI(data) {
        // Update progress bar
        document.getElementById('progress-bar').style.width = `${data.percent}%`;
        document.getElementById('progress-percent').textContent = `${Math.round(data.percent)}%`;
        
        // Update stage
        document.getElementById('current-stage').textContent = data.stage;
        document.getElementById('stage-message').textContent = data.message;
        
        // Update time
        if (data.elapsed) {
            document.getElementById('time-elapsed').textContent = this.formatTime(data.elapsed);
        }
        if (data.remaining) {
            document.getElementById('time-remaining').textContent = this.formatTime(data.remaining);
        }
        
        // Handle completion
        if (data.percent >= 100) {
            this.eventSource.close();
            this.showDownloadLinks();
        }
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    showDownloadLinks() {
        fetch(`/api/job/${this.jobId}/files`)
            .then(response => response.json())
            .then(files => {
                const container = document.getElementById('download-container');
                container.innerHTML = '<h3>Files Ready for Download:</h3>';
                
                files.forEach(file => {
                    const link = document.createElement('a');
                    link.href = file.url;
                    link.className = 'download-link';
                    link.textContent = `Download ${file.type.toUpperCase()} (${file.size_mb} MB)`;
                    container.appendChild(link);
                });
                
                container.style.display = 'block';
            });
    }
}
```

### Phase 6: Email Notifications (Day 6)

#### 6.1 Email Service Implementation
```python
# core/email_service.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

class EmailService:
    def __init__(self, smtp_server, smtp_port, username, password, from_email):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        
    def send_email(self, to_email: str, subject: str, html_body: str):
        """Send an HTML email"""
        msg = MIMEMultipart('alternative')
        msg['From'] = self.from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
            
    def send_completion_email(self, job, files):
        """Send completion notification with download links"""
        html = f"""
        <html>
            <body>
                <h2>Your pose analysis is ready!</h2>
                <p>Video: {job.video_filename}</p>
                <p>Processing time: {job.time_elapsed_seconds/60:.1f} minutes</p>
                
                <h3>Download your files:</h3>
                <ul>
                    {''.join([f'<li><a href="{f.r2_url}">{f.file_type.upper()} ({f.size_mb:.1f} MB)</a></li>' for f in files])}
                </ul>
                
                <p>Files will be available for 7 days.</p>
                
                <hr>
                <p style="font-size: 12px; color: #666;">
                    Processed on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </p>
            </body>
        </html>
        """
        
        user = job.user
        self.send_email(user.email, "Pose Analysis Completed", html)
        
    def send_failure_email(self, job, error_message):
        """Send failure notification with error details"""
        html = f"""
        <html>
            <body>
                <h2>Processing Failed</h2>
                <p>Video: {job.video_filename}</p>
                <p>Error: {error_message}</p>
                
                <p>Please contact support with Job ID: {job.id}</p>
                
                <h3>Error Log:</h3>
                <pre style="background: #f0f0f0; padding: 10px;">
                    {error_message}
                </pre>
            </body>
        </html>
        """
        
        user = job.user
        self.send_email(user.email, "Pose Analysis Failed", html)
```

### Phase 7: Main Flask Application (Day 7)

#### 7.1 Main App Implementation
```python
# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime

from config import Config
from models import db, User, Job, File, Log
from auth import init_auth
from core.runpod_client import RunPodClient
from core.storage_client import R2StorageClient
from core.job_processor import JobProcessor
from core.email_service import EmailService

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Initialize authentication
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize services
runpod_client = RunPodClient(
    app.config['RUNPOD_API_KEY'],
    app.config['RUNPOD_POD_ID']
)

storage_client = R2StorageClient(
    app.config['R2_ACCOUNT_ID'],
    app.config['R2_ACCESS_KEY_ID'],
    app.config['R2_SECRET_ACCESS_KEY'],
    app.config['R2_BUCKET_NAME']
)

email_service = EmailService(
    app.config['SMTP_SERVER'],
    app.config['SMTP_PORT'],
    app.config['SMTP_USERNAME'],
    app.config['SMTP_PASSWORD'],
    app.config['EMAIL_FROM']
)

job_processor = JobProcessor(app, runpod_client, storage_client, email_service)

# Routes
@app.route('/')
@login_required
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """Handle video upload and create job"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Validate file
    if not video.filename.lower().endswith('.mp4'):
        return jsonify({'error': 'Only MP4 files allowed'}), 400
        
    # Check file size (5GB max)
    video.seek(0, os.SEEK_END)
    size_mb = video.tell() / (1024 * 1024)
    video.seek(0)
    
    if size_mb > 5120:
        return jsonify({'error': 'File too large (max 5GB)'}), 400
        
    # Save video temporarily
    filename = secure_filename(f"{uuid.uuid4()}_{video.filename}")
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(temp_path)
    
    # Create job
    job = Job(
        user_id=current_user.id,
        video_filename=video.filename,
        video_size_mb=size_mb,
        status='queued'
    )
    db.session.add(job)
    db.session.commit()
    
    # Add to processing queue
    job_processor.add_job(job.id)
    
    # Log
    Log.create(job.id, 'info', f'Job created for {video.filename}')
    
    return jsonify({
        'job_id': job.id,
        'redirect': url_for('progress', job_id=job.id)
    })

@app.route('/progress/<int:job_id>')
@login_required
def progress(job_id):
    """Show processing progress page"""
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        abort(403)
    return render_template('progress.html', job=job)

@app.route('/api/progress/<int:job_id>')
@login_required
def progress_stream(job_id):
    """SSE endpoint for progress updates"""
    def generate():
        job = Job.query.get(job_id)
        if not job or job.user_id != current_user.id:
            return
            
        while job.status in ['queued', 'processing']:
            # Send current status
            data = {
                'stage': job.processing_stage or 'queued',
                'percent': job.progress_percent or 0,
                'message': f"Processing {job.video_filename}",
                'elapsed': job.time_elapsed_seconds,
                'remaining': job.time_remaining_seconds,
                'frames_processed': job.processed_frames,
                'total_frames': job.total_frames
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # Wait before next update
            time.sleep(1)
            
            # Refresh job from database
            db.session.refresh(job)
            
        # Send final status
        final_data = {
            'stage': job.status,
            'percent': 100 if job.status == 'completed' else 0,
            'message': 'Processing completed' if job.status == 'completed' else 'Processing failed'
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        
    return Response(generate(), mimetype="text/event-stream")

@app.route('/api/job/<int:job_id>/files')
@login_required
def get_job_files(job_id):
    """Get download links for job files"""
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        abort(403)
        
    files = File.query.filter_by(job_id=job_id).all()
    return jsonify([{
        'type': f.file_type,
        'filename': f.filename,
        'url': f.r2_url,
        'size_mb': f.size_mb
    } for f in files])

@app.route('/history')
@login_required
def history():
    """Show job history for current user"""
    jobs = Job.query.filter_by(user_id=current_user.id)\
                    .order_by(Job.created_at.desc())\
                    .limit(50)\
                    .all()
    return render_template('history.html', jobs=jobs)

@app.route('/admin/stats')
@login_required
def admin_stats():
    """Admin dashboard with usage statistics"""
    if current_user.username != 'admin':
        abort(403)
        
    stats = {
        'total_jobs': Job.query.count(),
        'completed_jobs': Job.query.filter_by(status='completed').count(),
        'failed_jobs': Job.query.filter_by(status='failed').count(),
        'total_processing_time': db.session.query(db.func.sum(Job.time_elapsed_seconds)).scalar() or 0,
        'total_cost': db.session.query(db.func.sum(UsageStats.cost_usd)).scalar() or 0
    }
    
    return render_template('admin_stats.html', stats=stats)

# Initialize database and start job processor
@app.before_first_request
def initialize():
    db.create_all()
    init_auth(app)  # Initialize hardcoded users from testw4
    job_processor.start()
    storage_client.setup_lifecycle_rules()

if __name__ == '__main__':
    app.run(debug=True)
```

### Phase 8: Deployment Configuration (Day 8)

#### 8.1 Render.com Configuration
```yaml
# render.yaml
services:
  - type: web
    name: flask-runpod-app
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: DATABASE_URL
        value: sqlite:///app.db
      - key: RUNPOD_API_KEY
        sync: false
      - key: RUNPOD_POD_ID
        sync: false
      - key: R2_ACCOUNT_ID
        sync: false
      - key: R2_ACCESS_KEY_ID
        sync: false
      - key: R2_SECRET_ACCESS_KEY
        sync: false
      - key: R2_BUCKET_NAME
        value: pose-analysis-files
      - key: SMTP_USERNAME
        sync: false
      - key: SMTP_PASSWORD
        sync: false
```

#### 8.2 Requirements File
```txt
# requirements.txt
Flask==2.3.3
Flask-Login==0.6.2
Flask-SQLAlchemy==3.0.5
gunicorn==21.2.0
python-dotenv==1.0.0
runpod==1.3.0
boto3==1.28.57
opencv-python-headless==4.8.1.78
numpy==1.24.3
pandas==2.0.3
openpyxl==3.1.2
requests==2.31.0
Werkzeug==2.3.7
email-validator==2.0.0
```

## 💰 Cost Analysis

### Monthly Cost Breakdown:
| Service | Usage | Cost |
|---------|-------|------|
| Render.com | 750 hrs/month free tier | $0 |
| Cloudflare R2 | 10GB storage, no egress | $0 |
| RunPod A5000 | ~10 hrs/month @ $0.79/hr | $7.90 |
| Email (Gmail) | 500/day free | $0 |
| **Total** | | **~$8/month** |

### Per-Video Cost:
- Average 30-minute video processing: ~30 minutes GPU time
- Cost: ~$0.40 per video
- Storage: Free (within 10GB limit)

## 🔐 Security Considerations

1. **API Keys**: Store all sensitive keys in environment variables
2. **File Access**: Use signed URLs for time-limited access (optional enhancement)
3. **User Auth**: Basic auth with hashed passwords (from testw4)
4. **Rate Limiting**: Natural rate limiting through FIFO queue
5. **Input Validation**: File size, type, and duration checks
6. **SQL Injection**: Using SQLAlchemy ORM prevents SQL injection

## 📊 Monitoring & Logging

1. **Application Logs**: Stored in SQLite `logs` table
2. **RunPod Logs**: Captured and stored per job
3. **Usage Tracking**: Cost and time tracking in `usage_stats` table
4. **Health Checks**: Endpoint for uptime monitoring
5. **Error Reporting**: Comprehensive error logs accessible to admin

## 🚀 Deployment Steps

1. **Setup Cloudflare R2**:
   - Create Cloudflare account
   - Create R2 bucket
   - Generate API credentials
   - Configure lifecycle rules

2. **Prepare RunPod**:
   - Ensure pod has all scripts
   - Test processing pipeline
   - Note pod ID

3. **Deploy to Render**:
   - Push code to GitHub
   - Connect GitHub to Render
   - Configure environment variables
   - Deploy application

4. **Post-Deployment**:
   - Test end-to-end flow
   - Verify email notifications
   - Check file downloads
   - Monitor first few jobs

## 📝 Testing Checklist

- [ ] User login with hardcoded credentials
- [ ] Video upload (small test file)
- [ ] Queue management with multiple uploads
- [ ] Progress tracking via SSE
- [ ] RunPod pod startup from cold
- [ ] Processing pipeline execution
- [ ] File upload to R2
- [ ] Download links generation
- [ ] Email notifications
- [ ] Job retry on failure
- [ ] Pod idle timeout
- [ ] 7-day file deletion
- [ ] Admin statistics page
- [ ] Error logging and display

## 🔄 Future Enhancements

1. **Phase 2 Features**:
   - Visualization video generation
   - Video preview before processing
   - Processing options UI
   - Batch upload support

2. **Optimizations**:
   - Video compression before upload
   - Distributed processing for large videos
   - Caching of common processing results
   - Progressive video upload

3. **Advanced Features**:
   - API endpoint for programmatic access
   - Webhook notifications
   - Processing templates/presets
   - Comparison between multiple analyses

## 📚 Additional Resources

- [RunPod API Documentation](https://docs.runpod.io/api)
- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2)
- [Render.com Flask Guide](https://render.com/docs/deploy-flask)
- [Flask-SSE Tutorial](https://flask-sse.readthedocs.io)

---

**Note**: This plan is designed for an internal business application with 10 users. For production/commercial use, additional security, scalability, and reliability features would be required.
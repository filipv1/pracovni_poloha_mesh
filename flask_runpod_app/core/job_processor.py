"""
Job Processor with Queue Management
Handles background job processing for video analysis
"""
import os
import threading
import queue
import time
import json
import traceback
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class JobProcessor:
    """Background job processor with FIFO queue"""
    
    def __init__(self, app, runpod_client, storage_client, email_service):
        self.app = app
        self.runpod_client = runpod_client
        self.storage_client = storage_client
        self.email_service = email_service
        self.job_queue = queue.Queue()
        self.processing = False
        self.current_job = None
        self.worker_thread = None
        self.progress_callbacks = {}
        
    def start(self):
        """Start the background job processor"""
        if self.processing:
            logger.warning("Job processor already running")
            return
            
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.worker_thread.start()
        logger.info("Job processor started")
        
    def stop(self):
        """Stop the job processor"""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Job processor stopped")
        
    def add_job(self, job_id: int):
        """Add a job to the processing queue"""
        self.job_queue.put(job_id)
        logger.info(f"Job {job_id} added to queue (position: {self.job_queue.qsize()})")
        
    def register_progress_callback(self, job_id: int, callback):
        """Register a callback for progress updates"""
        self.progress_callbacks[job_id] = callback
        
    def unregister_progress_callback(self, job_id: int):
        """Remove progress callback"""
        if job_id in self.progress_callbacks:
            del self.progress_callbacks[job_id]
            
    def get_queue_position(self, job_id: int) -> int:
        """Get position of job in queue"""
        # This is approximate since queue.Queue doesn't provide direct access
        if self.current_job and self.current_job == job_id:
            return 0
        
        # Check if in queue (approximate)
        queue_list = list(self.job_queue.queue)
        if job_id in queue_list:
            return queue_list.index(job_id) + 1
        
        return -1
    
    def _process_jobs(self):
        """Main job processing loop"""
        while self.processing:
            try:
                # Get next job from queue (with timeout to allow checking stop flag)
                try:
                    job_id = self.job_queue.get(timeout=5)
                    self.current_job = job_id
                    self._process_single_job(job_id)
                    self.current_job = None
                except queue.Empty:
                    # No jobs in queue, check for idle timeout
                    if self.runpod_client:
                        idle_timeout = self.app.config.get('POD_IDLE_TIMEOUT_SECONDS', 300)
                        self.runpod_client.check_idle_timeout(idle_timeout)
                    continue
                    
            except Exception as e:
                logger.error(f"Job processor error: {e}")
                traceback.print_exc()
                time.sleep(5)
    
    def _process_single_job(self, job_id: int):
        """Process a single job with retry logic"""
        with self.app.app_context():
            from models import db, Job, File, Log, UsageStats
            
            job = Job.query.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            logger.info(f"Starting processing job {job_id}: {job.video_filename}")
            
            # Update job status
            job.status = 'processing'
            job.started_at = datetime.utcnow()
            db.session.commit()
            
            # Retry logic
            max_retries = self.app.config.get('JOB_RETRY_LIMIT', 10)
            
            for attempt in range(job.retry_count, max_retries):
                try:
                    # Process the job
                    self._update_progress(job_id, 'starting', 0, 'Initializing processing...')
                    
                    # Ensure RunPod is running
                    self._update_progress(job_id, 'starting', 5, 'Starting GPU instance...')
                    if self.runpod_client:
                        success, message = self.runpod_client.ensure_pod_running()
                        if not success:
                            raise Exception(f"Failed to start RunPod: {message}")
                    
                    # Upload video to processing location
                    self._update_progress(job_id, 'uploading', 10, 'Uploading video...')
                    video_path = self._prepare_video_for_processing(job)
                    
                    # Execute processing
                    self._update_progress(job_id, 'processing', 20, 'Processing video...')
                    result = self._execute_processing(job_id, video_path)
                    
                    # Download and store results
                    self._update_progress(job_id, 'downloading', 85, 'Downloading results...')
                    files = self._store_results(job_id, result)
                    
                    # Update job as completed
                    job.status = 'completed'
                    job.completed_at = datetime.utcnow()
                    job.progress_percent = 100
                    job.processing_stage = 'completed'
                    
                    # Calculate processing time and cost
                    if job.started_at:
                        processing_time = (job.completed_at - job.started_at).total_seconds()
                        job.time_elapsed_seconds = processing_time
                        
                        # Record usage stats
                        gpu_rate = 0.79  # A5000 rate per hour
                        cost = (processing_time / 3600) * gpu_rate
                        
                        usage = UsageStats(
                            job_id=job_id,
                            pod_id=self.runpod_client.pod_id if self.runpod_client else None,
                            gpu_type='A5000',
                            processing_time_seconds=processing_time,
                            cost_usd=round(cost, 4)
                        )
                        db.session.add(usage)
                    
                    db.session.commit()
                    
                    # Send success notification
                    if self.email_service:
                        self.email_service.send_completion_email(job, files)
                    
                    # Log success
                    Log.create(job_id, 'info', f'Processing completed successfully')
                    
                    self._update_progress(job_id, 'completed', 100, 'Processing completed!')
                    logger.info(f"Job {job_id} completed successfully")
                    
                    # Success - exit retry loop
                    break
                    
                except Exception as e:
                    job.retry_count = attempt + 1
                    job.error_message = str(e)
                    db.session.commit()
                    
                    Log.create(job_id, 'error', f"Attempt {attempt + 1} failed", str(e))
                    logger.error(f"Job {job_id} attempt {attempt + 1} failed: {e}")
                    
                    if attempt >= max_retries - 1:
                        # Max retries reached
                        job.status = 'failed'
                        job.completed_at = datetime.utcnow()
                        db.session.commit()
                        
                        if self.email_service:
                            self.email_service.send_failure_email(job, str(e))
                        
                        self._update_progress(job_id, 'failed', 0, f'Processing failed: {str(e)}')
                        logger.error(f"Job {job_id} failed after {max_retries} attempts")
                    else:
                        # Exponential backoff before retry
                        wait_time = min(2 ** attempt, 60)
                        logger.info(f"Retrying job {job_id} in {wait_time} seconds...")
                        time.sleep(wait_time)
    
    def _prepare_video_for_processing(self, job) -> str:
        """Prepare video for processing (upload to RunPod or R2)"""
        # Get the uploaded video path
        upload_path = os.path.join(self.app.config['UPLOAD_FOLDER'], f"job_{job.id}_input.mp4")
        
        if not os.path.exists(upload_path):
            raise FileNotFoundError(f"Input video not found: {upload_path}")
        
        # If using RunPod with SSH, upload directly
        if self.runpod_client and os.environ.get('RUNPOD_SSH_KEY'):
            remote_path = f"/workspace/inputs/job_{job.id}_input.mp4"
            success, message = self.runpod_client.upload_file(upload_path, remote_path)
            if success:
                return remote_path
            else:
                raise Exception(f"Failed to upload to RunPod: {message}")
        
        # Otherwise, upload to R2 and return URL
        if self.storage_client:
            key = f"inputs/job_{job.id}/input.mp4"
            success, url_or_error = self.storage_client.upload_file(upload_path, key)
            if success:
                return url_or_error
            else:
                raise Exception(f"Failed to upload to R2: {url_or_error}")
        
        # Fallback to local path
        return upload_path
    
    def _execute_processing(self, job_id: int, video_path: str) -> Dict:
        """Execute the processing pipeline"""
        result = {
            'pkl_path': None,
            'xlsx_path': None,
            'status': 'processing'
        }
        
        # If we have RunPod with SSH
        if self.runpod_client and os.environ.get('RUNPOD_SSH_KEY'):
            result = self._execute_on_runpod(job_id, video_path)
        else:
            # Fallback to local processing (for testing)
            result = self._execute_locally(job_id, video_path)
        
        return result
    
    def _execute_on_runpod(self, job_id: int, video_path: str) -> Dict:
        """Execute processing on RunPod GPU"""
        output_dir = f"/workspace/outputs/job_{job_id}"
        
        # Create output directory
        self.runpod_client.execute_command(f"mkdir -p {output_dir}")
        
        # Build the processing command
        command = f"""
        cd /workspace/pracovni_poloha_mesh && \\
        source ~/miniconda3/bin/activate trunk_analysis && \\
        python runpod_scripts/process_video.py {video_path} {output_dir} {job_id}
        """
        
        # Execute and monitor progress
        success, stdout, stderr = self.runpod_client.execute_command(command)
        
        # Parse output for progress updates
        for line in stdout.split('\n'):
            if line.startswith('PROGRESS|'):
                parts = line.split('|')
                if len(parts) >= 4:
                    stage = parts[1]
                    percent = float(parts[2])
                    message = parts[3]
                    self._update_progress(job_id, stage, percent, message)
            elif line.startswith('RESULT|'):
                result_json = line.split('|', 1)[1]
                return json.loads(result_json)
        
        if not success:
            raise Exception(f"Processing failed: {stderr}")
        
        # Return expected output paths
        return {
            'status': 'success',
            'pkl_path': f"{output_dir}/output_meshes.pkl",
            'xlsx_path': f"{output_dir}/ergonomic_analysis.xlsx"
        }
    
    def _execute_locally(self, job_id: int, video_path: str) -> Dict:
        """Execute processing locally (for testing without RunPod)"""
        output_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
        
        try:
            # Simulate processing stages
            stages = [
                ('mediapipe', 30, 'Running MediaPipe detection...'),
                ('smplx', 50, 'Fitting SMPL-X model...'),
                ('angles', 70, 'Calculating angles...'),
                ('analysis', 85, 'Generating ergonomic analysis...')
            ]
            
            for stage, percent, message in stages:
                self._update_progress(job_id, stage, percent, message)
                time.sleep(2)  # Simulate processing time
            
            # Create dummy output files for testing
            pkl_path = output_dir / "output_meshes.pkl"
            xlsx_path = output_dir / "ergonomic_analysis.xlsx"
            
            # Create dummy files
            pkl_path.write_bytes(b"dummy pkl data")
            xlsx_path.write_bytes(b"dummy xlsx data")
            
            return {
                'status': 'success',
                'pkl_path': str(pkl_path),
                'xlsx_path': str(xlsx_path)
            }
            
        except Exception as e:
            raise Exception(f"Local processing failed: {e}")
    
    def _store_results(self, job_id: int, result: Dict) -> list:
        """Store processing results and create File records"""
        from models import db, File
        
        files = []
        
        # Store PKL file
        if result.get('pkl_path'):
            pkl_key = f"outputs/job_{job_id}/meshes.pkl"
            pkl_file = self._store_file(
                result['pkl_path'],
                pkl_key,
                job_id,
                'pkl',
                'meshes.pkl'
            )
            if pkl_file:
                files.append(pkl_file)
        
        # Store XLSX file
        if result.get('xlsx_path'):
            xlsx_key = f"outputs/job_{job_id}/analysis.xlsx"
            xlsx_file = self._store_file(
                result['xlsx_path'],
                xlsx_key,
                job_id,
                'xlsx',
                'ergonomic_analysis.xlsx'
            )
            if xlsx_file:
                files.append(xlsx_file)
        
        db.session.commit()
        return files
    
    def _store_file(self, source_path: str, storage_key: str, job_id: int, file_type: str, filename: str):
        """Store a single file and create database record"""
        from models import db, File
        
        # Download from RunPod if needed
        local_path = source_path
        if self.runpod_client and source_path.startswith('/workspace/'):
            local_path = os.path.join(tempfile.gettempdir(), os.path.basename(source_path))
            success, message = self.runpod_client.download_file(source_path, local_path)
            if not success:
                logger.error(f"Failed to download from RunPod: {message}")
                return None
        
        # Upload to R2
        if self.storage_client and os.path.exists(local_path):
            success, url_or_error = self.storage_client.upload_file(local_path, storage_key)
            if success:
                # Get file size
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                
                # Create File record
                file_record = File(
                    job_id=job_id,
                    file_type=file_type,
                    filename=filename,
                    r2_key=storage_key,
                    r2_url=url_or_error,
                    size_mb=round(file_size_mb, 2),
                    expires_at=datetime.utcnow() + timedelta(days=7)
                )
                db.session.add(file_record)
                
                # Clean up local file if it was downloaded
                if local_path != source_path and os.path.exists(local_path):
                    os.remove(local_path)
                
                return file_record
            else:
                logger.error(f"Failed to upload to R2: {url_or_error}")
        
        return None
    
    def _update_progress(self, job_id: int, stage: str, percent: float, message: str):
        """Update job progress and notify listeners"""
        with self.app.app_context():
            from models import db, Job
            
            job = Job.query.get(job_id)
            if job:
                job.processing_stage = stage
                job.progress_percent = percent
                
                # Calculate time estimates
                if job.started_at and percent > 0 and percent < 100:
                    elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                    job.time_elapsed_seconds = elapsed
                    job.time_remaining_seconds = (elapsed / percent) * (100 - percent)
                
                db.session.commit()
            
            # Notify progress callback if registered
            if job_id in self.progress_callbacks:
                try:
                    self.progress_callbacks[job_id]({
                        'stage': stage,
                        'percent': percent,
                        'message': message,
                        'elapsed': job.time_elapsed_seconds if job else None,
                        'remaining': job.time_remaining_seconds if job else None
                    })
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")
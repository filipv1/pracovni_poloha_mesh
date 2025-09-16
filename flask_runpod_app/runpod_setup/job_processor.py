#!/usr/bin/env python
"""
Job Processor for RunPod
Runs on pod, watches for jobs and processes them
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
import boto3
import requests
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobProcessor:
    def __init__(self):
        # R2/S3 configuration
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY']
        )
        self.bucket = os.environ.get('R2_BUCKET_NAME', 'flaskrunpod')
        
        # Paths
        self.workspace = Path('/workspace')
        self.repo_path = self.workspace / 'pose_analysis'
        self.models_path = self.workspace / 'models' / 'smplx'
        
    def run(self):
        """Main loop - watch for jobs and process them"""
        logger.info("Job processor started")
        logger.info(f"Workspace: {self.workspace}")
        logger.info(f"Repository: {self.repo_path}")
        logger.info(f"Models: {self.models_path}")
        
        idle_count = 0
        max_idle = 60  # 5 minutes of idle = 60 * 5 second checks
        
        while True:
            try:
                # Check for pending jobs
                job = self.get_next_job()
                
                if job:
                    idle_count = 0
                    logger.info(f"Processing job: {job['id']}")
                    self.process_job(job)
                else:
                    idle_count += 1
                    if idle_count > max_idle:
                        logger.info("No jobs for 5 minutes, shutting down pod...")
                        self.shutdown_pod()
                        break
                    
                # Wait before next check
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(10)
    
    def get_next_job(self):
        """Get next job from queue (R2/S3)"""
        try:
            # List objects in jobs/pending/
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix='jobs/pending/',
                MaxKeys=1
            )
            
            if 'Contents' not in response or not response['Contents']:
                return None
            
            # Get first job
            job_key = response['Contents'][0]['Key']
            
            # Download job config
            job_obj = self.s3_client.get_object(Bucket=self.bucket, Key=job_key)
            job = json.loads(job_obj['Body'].read())
            
            # Move job to processing
            processing_key = job_key.replace('pending', 'processing')
            self.s3_client.copy_object(
                Bucket=self.bucket,
                CopySource=f"{self.bucket}/{job_key}",
                Key=processing_key
            )
            self.s3_client.delete_object(Bucket=self.bucket, Key=job_key)
            
            return job
            
        except Exception as e:
            logger.error(f"Error getting job: {e}")
            return None
    
    def process_job(self, job):
        """Process a single job"""
        job_id = job['id']
        temp_dir = Path(f"/tmp/job_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Download video
            video_path = temp_dir / "input.mp4"
            logger.info(f"Downloading video: {job['video_key']}")
            self.s3_client.download_file(
                self.bucket,
                job['video_key'],
                str(video_path)
            )
            
            # 2. Run 3D pipeline
            output_dir = temp_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            logger.info("Running 3D pipeline...")
            cmd = [
                sys.executable,
                str(self.repo_path / 'production_3d_pipeline_clean.py'),
                str(video_path),
                '--output_dir', str(output_dir),
                '--quality', job.get('quality', 'high'),
                '--device', 'cuda'
            ]
            
            # Set environment for SMPL-X models
            env = os.environ.copy()
            env['SMPLX_MODELS'] = str(self.models_path)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Pipeline failed: {result.stderr}")
            
            # 3. Run angle calculation
            pkl_file = output_dir / "meshes.pkl"
            if pkl_file.exists():
                logger.info("Calculating angles...")
                xlsx_file = output_dir / "analysis.xlsx"
                
                cmd_angles = [
                    sys.executable,
                    str(self.repo_path / 'create_combined_angles_csv_skin.py'),
                    str(pkl_file),
                    str(xlsx_file)
                ]
                
                subprocess.run(cmd_angles, cwd=str(self.repo_path), check=True)
            
            # 4. Upload results
            logger.info("Uploading results...")
            results = []
            for file_path in output_dir.glob('*'):
                if file_path.suffix in ['.xlsx', '.pkl', '.mp4', '.obj']:
                    result_key = f"results/{job_id}/{file_path.name}"
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket,
                        result_key
                    )
                    results.append({
                        'filename': file_path.name,
                        'key': result_key,
                        'size': file_path.stat().st_size
                    })
            
            # 5. Mark job as completed
            self.complete_job(job, results)
            logger.info(f"Job {job_id} completed successfully")
            
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id} timed out")
            self.fail_job(job, "Processing timeout")
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            traceback.print_exc()
            self.fail_job(job, str(e))
        finally:
            # Cleanup temp files
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def complete_job(self, job, results):
        """Mark job as completed"""
        job['status'] = 'completed'
        job['results'] = results
        job['completed_at'] = time.time()
        
        # Save to completed folder
        completed_key = f"jobs/completed/{job['id']}.json"
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=completed_key,
            Body=json.dumps(job)
        )
        
        # Remove from processing
        processing_key = f"jobs/processing/{job['id']}.json"
        self.s3_client.delete_object(Bucket=self.bucket, Key=processing_key)
        
        # Send completion webhook if configured
        if 'webhook_url' in job:
            self.send_webhook(job['webhook_url'], job)
    
    def fail_job(self, job, error):
        """Mark job as failed"""
        job['status'] = 'failed'
        job['error'] = error
        job['failed_at'] = time.time()
        
        # Save to failed folder
        failed_key = f"jobs/failed/{job['id']}.json"
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=failed_key,
            Body=json.dumps(job)
        )
        
        # Remove from processing
        processing_key = f"jobs/processing/{job['id']}.json"
        self.s3_client.delete_object(Bucket=self.bucket, Key=processing_key)
    
    def send_webhook(self, url, data):
        """Send completion webhook"""
        try:
            requests.post(url, json=data, timeout=10)
        except:
            pass  # Don't fail job if webhook fails
    
    def shutdown_pod(self):
        """Shutdown the pod to save money"""
        # This will cause pod to stop
        logger.info("Shutting down pod...")
        sys.exit(0)


if __name__ == '__main__':
    processor = JobProcessor()
    processor.run()
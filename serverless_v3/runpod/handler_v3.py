"""
RunPod Handler V4 - Extended Pipeline with 4 outputs (PKL, CSV, Excel, Videos)
Version: 4.0.0
Release: 2025-01-23
"""
import runpod
import os
import sys
import json
import uuid
import threading
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from s3_utils import StorageClient, JobManager
from config.config import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video_async(job_id, video_key, quality='medium', user_email=None):
    """Process video asynchronously with extended pipeline

    Args:
        job_id: Unique job identifier
        video_key: S3/R2 key of uploaded video
        quality: Processing quality
        user_email: Optional user email for notifications
    """
    storage = StorageClient()
    job_manager = JobManager(storage)

    try:
        logger.info(f"Starting async processing for job {job_id}")

        # Update status to processing
        job_manager.update_job_status(job_id, status='processing', progress=5)

        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download video from R2/S3
            logger.info(f"Downloading video from {video_key}")
            video_filename = video_key.split('/')[-1]
            video_path = temp_path / video_filename

            if not storage.download_file(video_key, video_path):
                raise Exception(f"Failed to download video {video_key}")

            job_manager.update_job_status(job_id, progress=10)

            # Import and run processing pipeline
            try:
                # Add parent directory to path to import processing module
                parent_dir = Path(__file__).parent.parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))

                import run_production_simple_p
                import create_combined_angles_csv_skin
                import ergonomic_time_analysis

                # Create output directory
                output_dir = temp_path / "output"
                output_dir.mkdir(exist_ok=True)

                # Set up arguments for processing
                original_argv = sys.argv
                sys.argv = [
                    'run_production_simple_p.py',
                    str(video_path),
                    str(output_dir),
                    '--quality', quality
                ]

                logger.info(f"Step 1/3: Running production pipeline for {video_filename}")

                # Run the processing
                run_production_simple_p.main()

                sys.argv = original_argv

                # Find the output PKL file
                pkl_files = list(output_dir.glob("*.pkl"))
                if not pkl_files:
                    raise Exception("No PKL file generated")

                pkl_path = pkl_files[0]
                logger.info(f"PKL file generated: {pkl_path}")

                job_manager.update_job_status(job_id, progress=30)

                # Step 2: Generate angles CSV
                logger.info("Step 2/3: Generating angles CSV")
                csv_path = output_dir / f"{job_id}_angles.csv"

                create_combined_angles_csv_skin.create_combined_angles_csv_skin(
                    pkl_file=str(pkl_path),
                    output_csv=str(csv_path),
                    lumbar_vertex=5614,  # Using default vertex
                    video_path=str(video_path)
                )

                logger.info(f"CSV generated: {csv_path}")
                job_manager.update_job_status(job_id, progress=45)

                # Step 3: Generate ergonomic analysis Excel
                logger.info("Step 3/3: Generating ergonomic analysis")
                excel_path = output_dir / f"{job_id}_ergonomic_analysis.xlsx"

                analyzer = ergonomic_time_analysis.ErgonomicTimeAnalyzer(str(csv_path))
                analyzer.run_analysis(output_excel=str(excel_path))

                logger.info(f"Excel generated: {excel_path}")
                job_manager.update_job_status(job_id, progress=70)

                # Upload all results to R2/S3
                logger.info("Uploading all results to storage")
                results = {}

                # Upload PKL
                pkl_key = f"{RESULTS_PREFIX}{job_id}/mesh.pkl"
                if not storage.upload_file(pkl_path, pkl_key):
                    raise Exception("Failed to upload PKL file")
                results['pkl_url'] = storage.generate_presigned_url(pkl_key, operation='get', expiry=DOWNLOAD_URL_EXPIRY)

                # Upload CSV
                csv_key = f"{RESULTS_PREFIX}{job_id}/angles.csv"
                if not storage.upload_file(csv_path, csv_key):
                    raise Exception("Failed to upload CSV file")
                results['csv_url'] = storage.generate_presigned_url(csv_key, operation='get', expiry=DOWNLOAD_URL_EXPIRY)

                # Upload Excel
                excel_key = f"{RESULTS_PREFIX}{job_id}/ergonomic_analysis.xlsx"
                if not storage.upload_file(excel_path, excel_key):
                    raise Exception("Failed to upload Excel file")
                results['excel_url'] = storage.generate_presigned_url(excel_key, operation='get', expiry=DOWNLOAD_URL_EXPIRY)

                job_manager.update_job_status(job_id, progress=95)

                # Update final status with all download URLs
                job_manager.update_job_status(
                    job_id,
                    status='completed',
                    progress=100,
                    results=results
                )

                logger.info(f"Job {job_id} completed successfully")

                # Clean up uploaded video (optional)
                # storage.delete_object(video_key)

            except ImportError as e:
                logger.error(f"Failed to import processing module: {e}")
                raise Exception(f"Processing module not available: {e}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update status with error
        job_manager.update_job_status(
            job_id,
            status='failed',
            error=str(e)
        )


def handler(job):
    """RunPod handler for V3 architecture

    Supports multiple actions:
    - start_processing: Start async video processing
    - get_status: Get job status
    - generate_upload_url: Generate presigned URL for upload
    - generate_download_url: Generate presigned URL for download
    """
    logger.info("=== HANDLER V4 STARTED (Extended Pipeline) ===")

    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "start_processing")

        logger.info(f"Action: {action}")
        logger.info(f"Input: {json.dumps(job_input, indent=2)}")

        storage = StorageClient()
        job_manager = JobManager(storage)

        if action == "start_processing":
            # Start async processing
            video_key = job_input.get("video_key")
            if not video_key:
                return {"output": {"status": "error", "error": "video_key is required"}}

            # Generate unique job ID
            job_id = str(uuid.uuid4())

            # Create initial job status
            job_manager.create_job_status(job_id, 'accepted')

            # Start processing in background thread
            thread = threading.Thread(
                target=process_video_async,
                args=(
                    job_id,
                    video_key,
                    job_input.get("quality", DEFAULT_QUALITY),
                    job_input.get("user_email")
                ),
                daemon=True
            )
            thread.start()

            return {
                "output": {
                    "status": "success",
                    "job_id": job_id,
                    "message": "Processing started"
                }
            }

        elif action == "get_status":
            # Get job status
            job_id = job_input.get("job_id")
            if not job_id:
                return {"output": {"status": "error", "error": "job_id is required"}}

            status = job_manager.get_job_status(job_id)
            if not status:
                return {"output": {"status": "error", "error": "Job not found"}}

            # For backward compatibility, add download_url for old PKL-only format
            if status['status'] == 'completed' and status.get('result_key') and not status.get('results'):
                status['download_url'] = storage.generate_presigned_url(
                    status['result_key'],
                    operation='get',
                    expiry=DOWNLOAD_URL_EXPIRY
                )

            # Return in RunPod format with output key
            return {
                "output": {
                    "status": "success",
                    "job_status": status
                }
            }

        elif action == "generate_upload_url":
            # Generate presigned URL for upload
            filename = job_input.get("filename", f"video_{str(uuid.uuid4())[:8]}.mp4")
            video_key = f"{UPLOADS_PREFIX}{filename}"

            upload_url = storage.generate_presigned_url(
                video_key,
                operation='put',
                expiry=UPLOAD_URL_EXPIRY
            )

            if not upload_url:
                return {"output": {"status": "error", "error": "Failed to generate upload URL"}}

            return {
                "output": {
                    "status": "success",
                    "upload_url": upload_url,
                    "video_key": video_key
                }
            }

        elif action == "generate_download_url":
            # Generate presigned URL for download
            result_key = job_input.get("result_key")
            if not result_key:
                return {"output": {"status": "error", "error": "result_key is required"}}

            download_url = storage.generate_presigned_url(
                result_key,
                operation='get',
                expiry=DOWNLOAD_URL_EXPIRY
            )

            if not download_url:
                return {"output": {"status": "error", "error": "Failed to generate download URL"}}

            return {
                "output": {
                    "status": "success",
                    "download_url": download_url
                }
            }

        else:
            return {"output": {"status": "error", "error": f"Unknown action: {action}"}}

    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "output": {
                "status": "error",
                "error": str(e)
            }
        }


# RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker V4 (Extended Pipeline)")
    logger.info("Version: 4.0.0 - Outputs: PKL, CSV, Excel, 4 Videos")
    runpod.serverless.start({"handler": handler})
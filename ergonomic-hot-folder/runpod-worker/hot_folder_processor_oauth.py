#!/usr/bin/env python3
"""
Hot Folder Processor for RunPod - OAuth Version
Processes videos from Google Drive using OAuth2 authentication and automatically shuts down when complete
Version: 2.0.0 (OAuth Migration)
"""

import os
import sys
import json
import time
import logging
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Add parent directory to path for imports
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Google Drive OAuth client
from google_drive_oauth_client import GoogleDriveOAuthClient

# Import pipeline components
try:
    import run_production_simple_p
    import create_combined_angles_csv_skin
    import ergonomic_time_analysis
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Pipeline components not available: {e}")
    PIPELINE_AVAILABLE = False


class HotFolderProcessor:
    """Main processor for hot folder pipeline with OAuth authentication"""

    def __init__(self):
        """Initialize processor with environment configuration"""
        # Google Drive folders from environment
        self.folder_processing = os.getenv('GOOGLE_DRIVE_FOLDER_PROCESSING', '')
        self.folder_completed = os.getenv('GOOGLE_DRIVE_FOLDER_COMPLETED', '')
        self.folder_archive = os.getenv('GOOGLE_DRIVE_FOLDER_ARCHIVE', '')
        self.folder_logs = os.getenv('GOOGLE_DRIVE_FOLDER_LOGS', '')

        # Processing configuration
        self.processing_quality = os.getenv('PROCESSING_QUALITY', 'medium')
        self.delete_after_processing = os.getenv('DELETE_VIDEOS_AFTER_PROCESSING', 'true').lower() == 'true'
        self.auto_shutdown_minutes = int(os.getenv('AUTO_SHUTDOWN_AFTER_MINUTES', '5'))

        # RunPod configuration
        self.pod_id = os.getenv('RUNPOD_POD_ID', 'unknown')
        self.runpod_api_key = os.getenv('RUNPOD_API_KEY', '')

        # OAuth configuration
        self.oauth_credentials_path = os.getenv(
            'OAUTH_CREDENTIALS_PATH',
            '/workspace/persistent/oauth/oauth_credentials.json'
        )
        self.oauth_token_dir = os.getenv(
            'OAUTH_TOKEN_DIR',
            '/workspace/persistent/oauth'
        )

        # Validate required folders
        required_folders = [
            ('PROCESSING', self.folder_processing),
            ('COMPLETED', self.folder_completed),
            ('ARCHIVE', self.folder_archive)
        ]

        missing_folders = [name for name, folder_id in required_folders if not folder_id]
        if missing_folders:
            raise ValueError(f"Missing required folder IDs: {', '.join(missing_folders)}")

        # Initialize Google Drive client
        self.drive_client = None
        self.init_drive_client()

        # Processing state
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.processed_count = 0
        self.failed_count = 0

        logger.info("="*60)
        logger.info("HOT FOLDER PROCESSOR STARTED (OAuth Version)")
        logger.info(f"Pod ID: {self.pod_id}")
        logger.info(f"Processing folder: {self.folder_processing}")
        logger.info(f"Completed folder: {self.folder_completed}")
        logger.info(f"Archive folder: {self.folder_archive}")
        logger.info(f"OAuth credentials: {self.oauth_credentials_path}")
        logger.info(f"Token directory: {self.oauth_token_dir}")
        logger.info(f"Auto-shutdown after: {self.auto_shutdown_minutes} minutes idle")
        logger.info("="*60)

    def init_drive_client(self):
        """Initialize Google Drive client with OAuth2 authentication"""
        try:
            # Validate OAuth credentials exist
            if not os.path.exists(self.oauth_credentials_path):
                raise FileNotFoundError(f"OAuth credentials not found: {self.oauth_credentials_path}")

            # Ensure token directory exists
            Path(self.oauth_token_dir).mkdir(parents=True, exist_ok=True)

            # Initialize OAuth client
            self.drive_client = GoogleDriveOAuthClient(
                credentials_path=self.oauth_credentials_path,
                token_dir=self.oauth_token_dir
            )

            logger.info("Google Drive OAuth client initialized successfully")

            # Log authentication info
            auth_info = self.drive_client.get_auth_info()
            logger.info(f"Token expires in: {auth_info.get('expires_in_minutes', 'unknown')} minutes")
            logger.info(f"Has refresh token: {auth_info.get('has_refresh_token', False)}")

        except Exception as e:
            logger.error(f"Failed to initialize Google Drive OAuth client: {e}")
            logger.error("Possible solutions:")
            logger.error(f"1. Ensure {self.oauth_credentials_path} exists")
            logger.error(f"2. Run initial OAuth authorization if no token exists")
            logger.error(f"3. Check OAuth credentials are valid for this project")
            raise

    def get_processing_files(self):
        """Get list of files to process from Google Drive"""
        try:
            # Get files from processing folder
            files = self.drive_client.list_files(self.folder_processing)

            # Filter for video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = [
                f for f in files
                if any(f['name'].lower().endswith(ext) for ext in video_extensions)
            ]

            logger.info(f"Found {len(video_files)} video files to process")

            return video_files

        except Exception as e:
            logger.error(f"Error getting processing files: {e}")
            return []

    def download_video(self, file_info, temp_dir):
        """Download video file from Google Drive"""
        try:
            file_id = file_info['id']
            filename = file_info['name']

            local_path = os.path.join(temp_dir, filename)

            logger.info(f"Downloading {filename} from Google Drive...")

            if self.drive_client.download_file(file_id, local_path):
                file_size = os.path.getsize(local_path)
                logger.info(f"Downloaded {filename} ({file_size:,} bytes)")
                return local_path
            else:
                logger.error(f"Failed to download {filename}")
                return None

        except Exception as e:
            logger.error(f"Error downloading video {file_info.get('name', 'unknown')}: {e}")
            return None

    def process_video(self, file_info):
        """Process a single video file through the complete pipeline"""
        filename = file_info['name']
        file_id = file_info['id']

        logger.info(f"Starting processing: {filename}")

        # Create temporary directory for this video
        with tempfile.TemporaryDirectory(prefix='ergonomic_') as temp_dir:
            try:
                # Step 1: Download video
                local_video_path = self.download_video(file_info, temp_dir)
                if not local_video_path:
                    return False

                # Step 2: Run pipeline
                logger.info(f"Running pipeline for {filename}...")
                success = self.run_pipeline(local_video_path, temp_dir)

                if not success:
                    logger.error(f"Pipeline failed for {filename}")
                    # Move file back to processing folder (it might have been deleted)
                    try:
                        if file_info.get('id'):
                            self.drive_client.move_file(file_info['id'], self.folder_processing)
                    except:
                        pass
                    return False

                # Step 3: Upload results
                logger.info(f"Uploading results for {filename}...")
                uploaded_files = self.upload_results(filename, temp_dir)

                logger.info(f"Uploaded {len(uploaded_files)} files: {', '.join(uploaded_files)}")

                # Step 4: Clean up original video
                if self.delete_after_processing:
                    logger.info(f"Deleting processed video {filename}")
                    self.drive_client.delete_file(file_id)
                else:
                    # Move to archive
                    logger.info(f"Moving {filename} to archive")
                    self.drive_client.move_file(file_id, self.folder_archive)

                # Step 5: Log success
                self.log_processing_result("SUCCESS", filename, len(uploaded_files))

                logger.info(f"Completed processing: {filename}")
                return True

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                logger.error(traceback.format_exc())

                # Log failure
                self.log_processing_result("FAILED", filename, 0, str(e))

                return False

    def run_pipeline(self, video_path, output_dir):
        """Run the complete ergonomic analysis pipeline"""
        try:
            if not PIPELINE_AVAILABLE:
                logger.error("Pipeline components not available")
                return False

            video_name = Path(video_path).stem

            # Step 1: Run main production pipeline (FIXED!)
            logger.info("Step 1: Running main production pipeline...")

            # Správné volání MasterPipeline (ne neexistující funkci)
            pipeline = run_production_simple_p.MasterPipeline(
                smplx_path="models/smplx",
                device='cpu',  # Safer on RunPod
                gender='neutral'
            )

            result_dict = pipeline.execute_parallel_pipeline(
                video_path,
                output_dir=output_dir,
                quality=self.processing_quality
            )

            if not result_dict:
                logger.error("Main pipeline failed")
                return False

            # Step 2: Generate combined angles CSV
            logger.info("Step 2: Generating combined angles CSV...")
            pkl_file = result_dict.get('mesh_file')
            angles_csv = os.path.join(output_dir, f"{video_name}_angles.csv")

            if pkl_file and os.path.exists(pkl_file):
                try:
                    result_2 = create_combined_angles_csv_skin.create_combined_angles_csv(
                        str(pkl_file),
                        angles_csv
                    )
                except Exception as e:
                    logger.warning(f"Angle CSV generation failed: {e}")
                    result_2 = True
            else:
                logger.warning("PKL file not found, skipping angle analysis")
                result_2 = True

            # Step 3: Generate ergonomic time analysis
            logger.info("Step 3: Running ergonomic time analysis...")
            try:
                result_3 = ergonomic_time_analysis.analyze_ergonomic_time(
                    video_path,
                    angles_csv if result_2 and os.path.exists(angles_csv) else None,
                    output_dir
                )
            except Exception as e:
                logger.warning(f"Ergonomic time analysis failed: {e}")
                result_3 = True  # Continue even if this fails

            logger.info("Pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def upload_results(self, video_filename, results_dir):
        """Upload all result files to Google Drive"""
        try:
            uploaded_files = []

            # Create results folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{Path(video_filename).stem}_results_{timestamp}"

            results_folder_id = self.drive_client.create_folder(folder_name, self.folder_completed)
            if not results_folder_id:
                logger.error("Failed to create results folder")
                return uploaded_files

            # Upload all files from results directory
            for item in Path(results_dir).iterdir():
                if item.is_file():
                    file_id = self.drive_client.upload_file(
                        str(item),
                        results_folder_id,
                        item.name
                    )
                    if file_id:
                        uploaded_files.append(item.name)
                    else:
                        logger.warning(f"Failed to upload {item.name}")

            return uploaded_files

        except Exception as e:
            logger.error(f"Error uploading results: {e}")
            return []

    def log_processing_result(self, status, filename, file_count, error_message=None):
        """Log processing result to Google Drive logs"""
        try:
            if not self.folder_logs:
                return

            # Get existing logs
            existing_logs = self.drive_client.list_files(
                self.folder_logs,
                name_contains="processing_log"
            )

            # Create log entry
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "pod_id": self.pod_id,
                "status": status,
                "filename": filename,
                "uploaded_files": file_count,
                "error_message": error_message
            }

            # Create log filename
            today = datetime.now().strftime("%Y-%m-%d")
            log_filename = f"processing_log_{today}.json"

            # Upload log entry
            log_content = json.dumps(log_entry, indent=2)
            temp_log_path = f"/tmp/{log_filename}"

            with open(temp_log_path, 'w') as f:
                f.write(log_content)

            self.drive_client.upload_file(temp_log_path, self.folder_logs, log_filename)
            os.remove(temp_log_path)

            logger.info(f"Logged processing result: {status}")

        except Exception as e:
            logger.warning(f"Failed to log processing result: {e}")

    def check_idle_shutdown(self):
        """Check if processor should shutdown due to inactivity"""
        idle_time = datetime.now() - self.last_activity
        idle_minutes = idle_time.total_seconds() / 60

        if idle_minutes >= self.auto_shutdown_minutes:
            logger.info(f"Auto-shutdown triggered after {idle_minutes:.1f} minutes of inactivity")
            return True

        return False

    def shutdown_pod(self):
        """Initiate pod shutdown"""
        logger.info("Initiating pod shutdown...")

        # Final log
        runtime = datetime.now() - self.start_time
        self.log_processing_result(
            "SHUTDOWN",
            "processor",
            self.processed_count,
            f"Runtime: {runtime}, Processed: {self.processed_count}, Failed: {self.failed_count}"
        )

        # Shutdown command
        try:
            if os.path.exists('/sbin/shutdown'):
                subprocess.run(['/sbin/shutdown', '-h', 'now'], check=False)
            elif os.path.exists('/usr/bin/sudo'):
                subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)
            else:
                logger.warning("No shutdown command available, exiting process")
                sys.exit(0)
        except Exception as e:
            logger.error(f"Shutdown command failed: {e}")
            sys.exit(0)

    def run(self):
        """Main processing loop"""
        try:
            logger.info("Starting hot folder monitoring...")

            while True:
                try:
                    # Get files to process
                    files_to_process = self.get_processing_files()

                    if not files_to_process:
                        logger.info("No files to process")

                        # Check for idle shutdown
                        if self.check_idle_shutdown():
                            self.shutdown_pod()
                            break

                        # Wait before next check
                        logger.info("Waiting 30 seconds before next check...")
                        time.sleep(30)
                        continue

                    # Process each file
                    for file_info in files_to_process:
                        self.last_activity = datetime.now()

                        if self.process_video(file_info):
                            self.processed_count += 1
                        else:
                            self.failed_count += 1

                    # Brief pause between processing cycles
                    time.sleep(5)

                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Processing complete. Processed: {self.processed_count}, Failed: {self.failed_count}")
            logger.info("Initiating pod shutdown...")
            self.shutdown_pod()


def main():
    """Main entry point"""
    try:
        processor = HotFolderProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"Failed to start processor: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
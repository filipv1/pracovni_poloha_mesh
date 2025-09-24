
import os
import uuid
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import runpod
import subprocess
import json
from pathlib import Path

# --- Configuration ---
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)

# --- Utility Functions ---

def get_job_status(job_id):
    """Fetches the status of a job from S3."""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME, Key=f"jobs/{job_id}.json"
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None  # Job status does not exist yet
        raise

def update_job_status(job_id, status, data=None):
    """Updates the status of a job in S3."""
    current_status = get_job_status(job_id) or {}
    current_status["status"] = status
    if data:
        current_status.update(data)
    
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=f"jobs/{job_id}.json",
        Body=json.dumps(current_status).encode("utf-8"),
        ContentType="application/json",
    )

# --- Asynchronous Job Function ---

def run_job(job):
    """
    The main function executed asynchronously by RunPod.
    Downloads, processes, and uploads the results.
    """
    job_details = job["input"]
    job_id = job_details["job_id"]
    file_key = job_details["file_key"]

    try:
        update_job_status(job_id, "processing")

        # Setup local directories
        video_path_obj = Path(file_key)
        video_name = video_path_obj.name
        video_stem = video_path_obj.stem
        
        input_dir = Path(f"/tmp/inputs/{job_id}")
        output_dir = Path(f"/tmp/outputs/{job_id}")
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        local_video_path = input_dir / video_name

        # Download input video from S3
        print(f"Downloading {file_key} to {local_video_path}...")
        s3_client.download_file(S3_BUCKET_NAME, file_key, str(local_video_path))
        print("Download complete.")

        # This script will act as a bridge to call the main processing script
        # This is more robust than importing and calling main directly in a serverless env
        runner_script = """
import sys
# Add the script's directory to the path to find main
sys.path.append('/workspace/serverless_v2/processing_script')
from main import main
import json

if __name__ == '__main__':
    results = main(
        video_path=sys.argv[1],
        output_dir=sys.argv[2]
    )
    # Print results to stdout to be captured
    print(json.dumps(results))
"""
        runner_path = input_dir / "runner.py"
        with open(runner_path, 'w') as f:
            f.write(runner_script)

        # Execute the processing script as a subprocess
        print("Starting processing subprocess...")
        # IMPORTANT: Activate the correct conda environment if needed!
        # This command assumes the script can run in the base env.
        # If not, you might need: ["/path/to/conda", "run", "-n", "my_env", "python", ...]
        process = subprocess.run(
            ["python", str(runner_path), str(local_video_path), str(output_dir)],
            capture_output=True,
            text=True,
            check=True # Will raise CalledProcessError on non-zero exit codes
        )
        print("Processing complete.")
        
        # The script prints the result paths as the last line of stdout
        result_line = process.stdout.strip().split('\n')[-1]
        result_paths = json.loads(result_line)
        local_pkl_path = result_paths["pkl_path"]
        local_video_out_path = result_paths["video_path"]

        # Upload results to S3
        result_pkl_key = f"outputs/{job_id}/{Path(local_pkl_path).name}"
        result_video_key = f"outputs/{job_id}/{Path(local_video_out_path).name}"

        print(f"Uploading {local_pkl_path} to {result_pkl_key}...")
        s3_client.upload_file(local_pkl_path, S3_BUCKET_NAME, result_pkl_key)

        print(f"Uploading {local_video_out_path} to {result_video_key}...")
        s3_client.upload_file(local_video_out_path, S3_BUCKET_NAME, result_video_key)
        print("Uploads complete.")

        # Final status update
        update_job_status(
            job_id,
            "complete",
            {"result_pkl": result_pkl_key, "result_video": result_video_key},
        )

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        update_job_status(job_id, "failed", {"error": str(e)})
    finally:
        # Clean up local files
        # import shutil
        # shutil.rmtree(input_dir, ignore_errors=True)
        # shutil.rmtree(output_dir, ignore_errors=True)
        pass # In serverless, the temp storage is ephemeral anyway

# --- Handler Endpoints ---

def start_upload(job):
    """
    Generates a presigned URL for the client to upload a file to S3.
    """
    job_input = job.get("input", {})
    file_name = job_input.get("fileName")
    
    if not file_name:
        return {"error": "fileName is required."}

    # Generate a unique key for the S3 object
    file_key = f"inputs/{uuid.uuid4()}/{file_name}"

    try:
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": file_key},
            ExpiresIn=3600,  # URL expires in 1 hour
        )
        return {"presignedUrl": presigned_url, "fileKey": file_key}
    except ClientError as e:
        return {"error": f"Failed to generate presigned URL: {e}"}

def start_processing(job):
    """
    Starts the asynchronous processing of a file that is already in S3.
    """
    job_input = job.get("input", {})
    file_key = job_input.get("fileKey")

    if not file_key:
        return {"error": "fileKey is required."}

    job_id = str(uuid.uuid4())

    # Set initial status
    update_job_status(job_id, "pending")

    # Run the main job asynchronously
    runpod.serverless.run_async(job_id, run_job, {"file_key": file_key, "job_id": job_id})

    return {"job_id": job_id}

def status(job):
    """
    Checks the status of a running job.
    """
    job_input = job.get("input", {})
    job_id = job_input.get("job_id")

    if not job_id:
        return {"error": "job_id is required."}

    job_status = get_job_status(job_id)

    if not job_status:
        return {"error": "Job not found."}

    # If complete, generate download URLs
    if job_status.get("status") == "complete":
        try:
            result_pkl_key = job_status.get("result_pkl")
            result_video_key = job_status.get("result_video")

            if result_pkl_key:
                pkl_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": S3_BUCKET_NAME, "Key": result_pkl_key},
                    ExpiresIn=3600,
                )
                job_status["result_pkl_url"] = pkl_url

            if result_video_key:
                video_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": S3_BUCKET_NAME, "Key": result_video_key},
                    ExpiresIn=3600,
                )
                job_status["result_video_url"] = video_url

        except ClientError as e:
            job_status["error"] = f"Failed to generate download URLs: {e}"

    return job_status


# --- RunPod Handler ---

def handler(job):
    """
    The main entry point for RunPod.
    Routes requests to the appropriate function based on the input.
    """
    job_input = job.get("input", {})
    endpoint = job_input.get("endpoint")

    if endpoint == "start_upload":
        return start_upload(job)
    elif endpoint == "start_processing":
        return start_processing(job)
    elif endpoint == "status":
        return status(job)
    else:
        return {"error": f"Invalid endpoint: {endpoint}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

"""
RunPod Serverless Handler for 3D Pose Analysis
"""

import runpod
import os
import json
import subprocess
import tempfile
import boto3
from typing import Dict, Any

def download_file(url: str, destination: str):
    """Download file from URL or S3"""
    if url.startswith("s3://"):
        # Download from S3
        bucket, key = url[5:].split("/", 1)
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, destination)
    else:
        # Download from HTTP
        import requests
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def upload_file(file_path: str, bucket: str, key: str):
    """Upload file to S3"""
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    return f"s3://{bucket}/{key}"

def process_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video through 3D pose analysis pipeline
    
    Expected input:
    {
        "input": {
            "video_url": "https://... or s3://...",
            "output_bucket": "bucket-name",
            "output_key": "path/to/output.xlsx",
            "quality": "high",  # ultra, high, medium
            "user_email": "user@example.com"
        }
    }
    """
    try:
        job_input = job['input']
        video_url = job_input['video_url']
        output_bucket = job_input.get('output_bucket', 'flaskrunpod')
        output_key = job_input.get('output_key', f"results/{job['id']}.xlsx")
        quality = job_input.get('quality', 'high')
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download video
            video_path = os.path.join(tmpdir, "input.mp4")
            print(f"Downloading video from {video_url}")
            download_file(video_url, video_path)
            
            # Run 3D pipeline
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing video with quality: {quality}")
            
            # Run the actual processing script
            cmd = [
                "python", "/workspace/production_3d_pipeline_clean.py",
                video_path,
                "--output_dir", output_dir,
                "--quality", quality,
                "--device", "cuda"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            if result.returncode != 0:
                raise Exception(f"Processing failed: {result.stderr}")
            
            # Run angle calculation
            pkl_file = os.path.join(output_dir, "meshes.pkl")
            xlsx_file = os.path.join(output_dir, "analysis.xlsx")
            
            if os.path.exists(pkl_file):
                # Generate angle analysis
                cmd_angles = [
                    "python", "/workspace/create_combined_angles_csv_skin.py",
                    pkl_file, xlsx_file
                ]
                subprocess.run(cmd_angles, cwd="/workspace")
            
            # Upload results
            results = []
            for filename in os.listdir(output_dir):
                if filename.endswith(('.xlsx', '.pkl', '.mp4')):
                    local_path = os.path.join(output_dir, filename)
                    s3_key = f"{output_key}/{filename}"
                    url = upload_file(local_path, output_bucket, s3_key)
                    results.append({
                        "filename": filename,
                        "url": url,
                        "size": os.path.getsize(local_path)
                    })
            
            return {
                "status": "completed",
                "results": results,
                "message": "Video processed successfully"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": f"Processing failed: {str(e)}"
        }

# RunPod handler
runpod.serverless.start({
    "handler": process_video
})
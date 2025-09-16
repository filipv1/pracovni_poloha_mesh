#!/bin/bash
# Setup script for RunPod with conda (no Docker)

echo "==================================="
echo "RunPod Setup with Conda & GitHub"
echo "==================================="

# 1. Clone repository from GitHub
echo "Step 1: Cloning repository..."
if [ ! -d "/workspace/pose_analysis" ]; then
    git clone https://github.com/YOUR_USERNAME/pose_analysis.git /workspace/pose_analysis
else
    cd /workspace/pose_analysis
    git pull
fi

# 2. Setup conda environment
echo "Step 2: Setting up Conda environment..."
if [ ! -d "/workspace/conda/envs/pose_analysis" ]; then
    conda env create -f /workspace/pose_analysis/flask_runpod_app/runpod_setup/environment.yml -p /workspace/conda/envs/pose_analysis
fi

# 3. Activate environment
source activate /workspace/conda/envs/pose_analysis

# 4. Download SMPL-X models if not present
echo "Step 3: Checking SMPL-X models..."
if [ ! -d "/workspace/models/smplx" ]; then
    mkdir -p /workspace/models/smplx
    echo "Please upload SMPL-X models to /workspace/models/smplx/"
    # Or download from your storage:
    # wget -O /workspace/models/smplx/SMPLX_NEUTRAL.npz YOUR_URL
fi

# 5. Create processing service
echo "Step 4: Creating processing service..."
cat > /workspace/process_service.py << 'EOF'
"""
Simple processing service for RunPod
Watches for jobs and processes them
"""
import os
import json
import time
import boto3
import subprocess
from pathlib import Path

def watch_for_jobs():
    """Watch S3 bucket for new jobs"""
    s3 = boto3.client('s3')
    bucket = os.environ.get('R2_BUCKET_NAME', 'flaskrunpod')
    
    while True:
        try:
            # Check for pending jobs
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix='jobs/pending/'
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    job_key = obj['Key']
                    process_job(s3, bucket, job_key)
                    
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(5)

def process_job(s3, bucket, job_key):
    """Process a single job"""
    # Download job config
    job_file = '/tmp/job.json'
    s3.download_file(bucket, job_key, job_file)
    
    with open(job_file, 'r') as f:
        job = json.load(f)
    
    # Download video
    video_key = job['video_key']
    video_path = '/tmp/input.mp4'
    s3.download_file(bucket, video_key, video_path)
    
    # Run processing
    output_dir = '/tmp/output'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'python', '/workspace/pose_analysis/production_3d_pipeline_clean.py',
        video_path,
        '--output_dir', output_dir,
        '--quality', job.get('quality', 'high')
    ]
    
    subprocess.run(cmd, check=True)
    
    # Upload results
    for file in Path(output_dir).glob('*'):
        result_key = f"results/{job['id']}/{file.name}"
        s3.upload_file(str(file), bucket, result_key)
    
    # Mark job as completed
    s3.delete_object(Bucket=bucket, Key=job_key)
    completed_key = job_key.replace('pending', 'completed')
    s3.put_object(Bucket=bucket, Key=completed_key, Body=json.dumps(job))

if __name__ == '__main__':
    watch_for_jobs()
EOF

# 6. Start the service
echo "Step 5: Starting processing service..."
python /workspace/process_service.py &

echo "==================================="
echo "Setup complete!"
echo "Pod is ready to process jobs"
echo "===================================" 
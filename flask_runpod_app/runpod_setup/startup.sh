#!/bin/bash
# Startup script for RunPod pod with Network Volume
# This runs every time pod starts (10-30 seconds)
# Conda env already exists on Network Volume from initial setup

echo "======================================="
echo "Starting Pose Analysis Pod"
echo "======================================="

# Check if this is first run (no conda env yet)
if [ ! -d "/workspace/conda/envs/pose_analysis" ]; then
    echo "First run detected - setting up environment..."
    bash /workspace/initial_setup.sh
    exit 0
fi

# Regular startup (conda already exists)
echo "Using existing conda environment from Network Volume"

# 1. Update code from GitHub (fast)
echo "Updating code from GitHub..."
cd /workspace
if [ ! -d "pose_analysis" ]; then
    git clone https://github.com/YOUR_USERNAME/pose_analysis.git
fi
cd pose_analysis
git pull

# 2. Activate conda (instant - already on volume)
source /workspace/conda/bin/activate pose_analysis

# 3. Quick pip update for any new packages (usually skipped)
pip install -r requirements_quick.txt --upgrade --quiet

# 4. Start job processor
echo "Starting job processor..."
python /workspace/pose_analysis/flask_runpod_app/runpod_setup/job_processor.py

echo "======================================="
echo "Pod Ready - Waiting for jobs"
echo "======================================="
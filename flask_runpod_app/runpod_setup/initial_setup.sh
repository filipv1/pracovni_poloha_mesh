#!/bin/bash
# Initial setup for Network Volume - runs only once
# This takes 20-30 minutes but only needs to run once per Network Volume

echo "======================================="
echo "INITIAL SETUP - Network Volume"
echo "This will take 20-30 minutes"
echo "======================================="

# 1. Install Miniconda to Network Volume
if [ ! -d "/workspace/conda" ]; then
    echo "Installing Miniconda to Network Volume..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/conda
    rm /tmp/miniconda.sh
    
    # Initialize conda
    /workspace/conda/bin/conda init bash
    source ~/.bashrc
fi

# 2. Create conda environment (slowest part - 15-20 min)
echo "Creating conda environment (this takes 15-20 minutes)..."
/workspace/conda/bin/conda env create -f /workspace/environment.yml -p /workspace/conda/envs/pose_analysis

# 3. Activate environment
source /workspace/conda/bin/activate pose_analysis

# 4. Download SMPL-X models (if not present)
if [ ! -d "/workspace/models/smplx" ]; then
    echo "Downloading SMPL-X models..."
    mkdir -p /workspace/models/smplx
    
    # Download from your storage or RunPod's cache
    # Option 1: From your R2/S3
    # aws s3 cp s3://your-bucket/smplx-models/ /workspace/models/smplx/ --recursive
    
    # Option 2: From public URL (if you have one)
    # wget https://your-storage.com/SMPLX_NEUTRAL.npz -O /workspace/models/smplx/SMPLX_NEUTRAL.npz
    # wget https://your-storage.com/SMPLX_MALE.npz -O /workspace/models/smplx/SMPLX_MALE.npz
    # wget https://your-storage.com/SMPLX_FEMALE.npz -O /workspace/models/smplx/SMPLX_FEMALE.npz
    
    echo "Please manually upload SMPL-X models to /workspace/models/smplx/"
fi

# 5. Clone repository
echo "Cloning repository..."
cd /workspace
git clone https://github.com/YOUR_USERNAME/pose_analysis.git

# 6. Test installation
echo "Testing installation..."
cd /workspace/pose_analysis
python -c "import mediapipe; import torch; import smplx; print('All packages installed successfully!')"

# 7. Create marker file
echo "$(date)" > /workspace/.initial_setup_complete

echo "======================================="
echo "INITIAL SETUP COMPLETE!"
echo "Network Volume is ready for use"
echo "Future pod starts will be fast (30 seconds)"
echo "======================================="

# Now run normal startup
bash /workspace/startup.sh
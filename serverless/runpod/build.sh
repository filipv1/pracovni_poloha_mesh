#!/bin/bash

# Build script for RunPod Docker image
# Stages files from parent directory and builds Docker image

set -e

echo "Preparing Docker build context..."

# Create temporary build directory
BUILD_DIR="build_context"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

# Copy Dockerfile and requirements
cp Dockerfile $BUILD_DIR/
cp requirements-frozen.txt $BUILD_DIR/
cp handler.py $BUILD_DIR/
cp handler_local_test.py $BUILD_DIR/

# Copy required Python files from parent directory
cp ../../run_production_simple.py $BUILD_DIR/
cp ../../production_3d_pipeline_clean.py $BUILD_DIR/
cp ../../create_combined_angles_csv_skin.py $BUILD_DIR/
cp ../../trunk_angle_calculator_skin.py $BUILD_DIR/
cp ../../neck_angle_calculator_skin.py $BUILD_DIR/
cp ../../arm_angle_calculator.py $BUILD_DIR/

# Copy SMPL-X models if they exist
if [ -d "../../models/smplx" ]; then
    echo "Copying SMPL-X models..."
    mkdir -p $BUILD_DIR/models/smplx
    cp ../../models/smplx/*.npz $BUILD_DIR/models/smplx/ 2>/dev/null || true
fi

# Update Dockerfile to copy local files
cat > $BUILD_DIR/Dockerfile.final << 'EOF'
# RunPod Serverless Docker Image for Ergonomic Analysis Pipeline
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

# Copy all application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "-u", "handler.py"]
EOF

# Build Docker image
cd $BUILD_DIR
echo "Building Docker image..."
docker build -f Dockerfile.final -t ergonomic-analyzer:latest .

# Cleanup
cd ..
rm -rf $BUILD_DIR

echo "Build complete! Image: ergonomic-analyzer:latest"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ergonomic-analyzer:latest YOUR_USERNAME/ergonomic-analyzer:latest"
echo "  docker push YOUR_USERNAME/ergonomic-analyzer:latest"
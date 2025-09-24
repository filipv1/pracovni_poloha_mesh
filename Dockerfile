# V4 RunPod Serverless Docker Image - Extended Pipeline Version
# Version 4.0.0 - 2025-01-23
# Adds: CSV generation, Excel analysis, 4 videos output
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies (same as working version)
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

# Copy requirements first for better caching
COPY requirements_v3.txt .

# Update pip and install Python dependencies
# Use --ignore-installed to avoid conflicts with pre-installed packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir boto3==1.34.0 botocore==1.34.0 && \
    pip install --no-cache-dir mediapipe==0.10.8 && \
    pip install --no-cache-dir -r requirements_v3.txt || true

# Copy all application files
COPY . /app/

# Copy Python modules from docker_files
COPY docker_files/*.py /app/

# Copy SMPL-X models
COPY docker_files/models /app/models

# Create necessary directories
RUN mkdir -p /app/models/smplx

# Note: SMPL-X models need to be added separately or downloaded at runtime
# due to their large size (104MB+)

# Set environment variables for R2/S3 (will be overridden by RunPod secrets)
ENV STORAGE_PROVIDER=r2
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV SMPLX_MODELS_DIR=/app/models/smplx

# RunPod handler
CMD ["python", "-u", "runpod/handler_v3.py"]
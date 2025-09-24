# RunPod Serverless Docker Image for Ergonomic Analysis Pipeline
# Uses official RunPod PyTorch base image for CUDA compatibility

FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
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

# Copy requirements first for better caching
COPY requirements-frozen.txt .

# Update pip first and install Python dependencies
# Use --ignore-installed to avoid conflicts with pre-installed packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements-frozen.txt

# Create directory for SMPL-X models
RUN mkdir -p /app/models/smplx

# Copy SMPL-X NPZ models (104MB each, total ~312MB)
# PKL files not needed - NPZ models confirmed sufficient
COPY models/smplx/*.npz /app/models/smplx/

# Copy all Python files
COPY *.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV SMPLX_MODELS_DIR=/app/models/smplx

# RunPod handler will be the entry point
CMD ["python", "-u", "handler.py"]
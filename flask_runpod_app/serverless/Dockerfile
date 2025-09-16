# RunPod Serverless Docker Image for 3D Pose Analysis
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python packages
COPY requirements_serverless.txt .
RUN pip install --no-cache-dir -r requirements_serverless.txt

# Download SMPL-X models (you need to provide these)
# Option 1: Copy from local if you have them
# COPY models/smplx /workspace/models/smplx

# Option 2: Download from your storage
# RUN wget https://your-storage/smplx-models.tar.gz && \
#     tar -xzf smplx-models.tar.gz -C /workspace/models/ && \
#     rm smplx-models.tar.gz

# Copy processing scripts
COPY ../production_3d_pipeline_clean.py .
COPY ../trunk_angle_calculator_skin.py .
COPY ../neck_angle_calculator_skin.py .
COPY ../create_combined_angles_csv_skin.py .

# Copy handler
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
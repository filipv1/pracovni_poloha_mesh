# SMPL-X Docker Setup Documentation

## 📅 Implementation Date: September 23, 2025

## 🎯 Overview

This document describes the process of integrating SMPL-X models into the RunPod Docker container and resolving related issues that prevented the V3 serverless architecture from functioning properly.

## 🚨 The Problem

The RunPod deployment was failing with the following error:
```
FileNotFoundError: /app/models/smplx/SMPLX_NEUTRAL.npz
```

The SMPL-X model files were missing from the Docker image, which are essential for:
- Converting MediaPipe landmarks to SMPL-X format
- Generating 3D human body meshes
- Performing ergonomic analysis calculations

## ✅ The Solution

### 1. Model Files Requirement

The following SMPL-X model files are required:
- `SMPLX_NEUTRAL.npz` (104MB) - Gender-neutral model
- `SMPLX_MALE.npz` (optional)
- `SMPLX_FEMALE.npz` (optional)

These files must be obtained from the official SMPL-X website and cannot be distributed with the code due to licensing restrictions.

### 2. Docker Build Context Issue

**Problem**: The Dockerfile was in `serverless_v3/` but needed to access `models/smplx/` from the parent directory.

**Solution**: Build Docker image from parent directory with correct context:
```bash
# Build from parent directory (pracovni_poloha_mesh/)
cd ..
docker build -f serverless_v3/Dockerfile -t vaclavikmasa/ergonomic-analysis-v3:v3-with-models .
```

### 3. Dockerfile Modifications

Added the following to `serverless_v3/Dockerfile`:
```dockerfile
# Create model directory
RUN mkdir -p /app/models/smplx

# Copy SMPL-X model - only NEUTRAL needed (104MB)
COPY models/smplx/SMPLX_NEUTRAL.npz /app/models/smplx/

# Set environment variable
ENV SMPLX_MODELS_DIR=/app/models/smplx
```

## 🔧 Setup Process

### Step 1: Prepare Model Files
```bash
# Create models directory in project root
mkdir -p models/smplx

# Copy SMPLX_NEUTRAL.npz to this directory
# (Download from https://smpl-x.is.tue.mpg.de/)
cp /path/to/SMPLX_NEUTRAL.npz models/smplx/
```

### Step 2: Build Docker Image

#### Automated Method (Recommended)
```bash
cd serverless_v3
rebuild_with_models.bat
```

This batch script:
1. Changes to parent directory
2. Builds image with correct context
3. Pushes to Docker Hub
4. Shows completion message

#### Manual Method
```bash
# From project root (pracovni_poloha_mesh/)
docker build -f serverless_v3/Dockerfile -t vaclavikmasa/ergonomic-analysis-v3:v3-with-models .
docker push vaclavikmasa/ergonomic-analysis-v3:v3-with-models
```

### Step 3: Update RunPod Endpoint

1. Log in to RunPod dashboard
2. Navigate to Serverless > Endpoints
3. Select endpoint `d1mtcfjymab45g`
4. Update container image to: `vaclavikmasa/ergonomic-analysis-v3:v3-with-models`
5. Save and restart workers

## 📦 Docker Image Details

### Base Image
- `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
- Size: ~6GB

### With SMPL-X Models
- Additional: +104MB (SMPLX_NEUTRAL.npz)
- Total size: ~6.1GB

### Image Tags History
- `v3latestv2` - Initial version with bug fixes
- `v3-with-models` - Includes SMPL-X model files (WORKING)

## 🛠️ Troubleshooting

### Issue: Docker build fails with "file not found"
**Cause**: Building from wrong directory
**Solution**: Always build from parent directory with `-f serverless_v3/Dockerfile`

### Issue: Model still not found after rebuild
**Possible causes**:
1. RunPod using cached old image
2. Model file not in correct location
3. Environment variable not set

**Solutions**:
1. Force RunPod to pull new image (change tag or restart endpoint)
2. Verify `models/smplx/SMPLX_NEUTRAL.npz` exists
3. Check `SMPLX_MODELS_DIR` environment variable in RunPod settings

### Issue: Permission denied when accessing model
**Solution**: Ensure model file has correct permissions in Dockerfile:
```dockerfile
RUN chmod 644 /app/models/smplx/SMPLX_NEUTRAL.npz
```

## 📂 File Structure

```
pracovni_poloha_mesh/
├── models/
│   └── smplx/
│       └── SMPLX_NEUTRAL.npz    # Required model file
├── serverless_v3/
│   ├── Dockerfile                # Docker configuration
│   ├── rebuild_with_models.bat   # Automated build script
│   └── runpod/
│       └── handler_v3.py         # Uses SMPL-X model
└── production_3d_pipeline_clean.py  # Main processing pipeline
```

## 🚀 Verification

### Local Testing
```bash
# Test if model loads correctly
docker run -it vaclavikmasa/ergonomic-analysis-v3:v3-with-models python -c "
import os
import numpy as np
path = '/app/models/smplx/SMPLX_NEUTRAL.npz'
print(f'Model exists: {os.path.exists(path)}')
if os.path.exists(path):
    data = np.load(path, allow_pickle=True)
    print(f'Model loaded successfully, keys: {list(data.keys())[:5]}...')
"
```

### RunPod Testing
Use the V3 frontend to upload a test video and verify processing completes without model errors.

## 📝 Important Notes

1. **Model License**: SMPL-X models have specific licensing terms. Ensure compliance before deployment.

2. **Storage Impact**: Adding models increases Docker image size. Consider using RunPod network volumes for models in production.

3. **Build Time**: Full rebuild takes 5-10 minutes depending on network speed.

4. **Caching**: Docker layer caching helps with subsequent builds. Model COPY is near the end to maximize cache usage.

## 🔄 Update Process

When updating the processing code without changing models:
```bash
# Use quick rebuild (doesn't re-upload models layer)
cd serverless_v3
quick_rebuild.bat
```

When models need to be updated:
```bash
# Full rebuild with models
cd serverless_v3
rebuild_with_models.bat
```

## ✨ Benefits of This Setup

1. **Self-contained**: No external model downloads needed at runtime
2. **Faster startup**: Models pre-loaded in image
3. **Reliable**: No network issues fetching models
4. **Versioned**: Models are part of the Docker image version

## 🐛 Known Limitations

1. **Image size**: 6GB+ is large for a Docker image
2. **Build context**: Must be careful about build directory
3. **Model updates**: Requires full image rebuild

## 🔮 Future Improvements

1. **Network Volume**: Store models in RunPod network volume to reduce image size
2. **Multi-stage build**: Optimize Docker layers for faster builds
3. **Model selection**: Allow runtime selection of NEUTRAL/MALE/FEMALE models
4. **Compression**: Use compressed model format to reduce size

---

**Status**: ✅ RESOLVED AND WORKING
**Docker Image**: `vaclavikmasa/ergonomic-analysis-v3:v3-with-models`
**RunPod Endpoint**: `d1mtcfjymab45g`

---

*This setup ensures the SMPL-X models are always available for the processing pipeline.*
# RunPod Deployment V4.2 - Fixed Version

## Quick Deploy Instructions

### 1. Build and Push Docker Image
```bash
cd serverless_v3
build_v4.2.bat
```

This will:
- Copy required Python modules from parent directory
- Copy SMPL-X NEUTRAL model (108MB)
- Build Docker image
- Push to Docker Hub as `vaclavikmasa/ergonomic-analysis:v4.2`
- Clean up temporary files

### 2. Update RunPod
1. Go to RunPod console
2. Navigate to your serverless endpoint
3. Update Docker image to: `vaclavikmasa/ergonomic-analysis:v4.2`
4. Restart the endpoint

## What's Fixed in V4.2
- All required Python modules are now included in the Docker image
- SMPL-X NEUTRAL model is bundled (108MB)
- Proper file paths configured for RunPod environment

## Files Created
- `prepare_docker_files.bat` - Copies required files before build
- `build_v4.2.bat` - Complete build and push workflow
- `cleanup_docker_files.bat` - Removes temporary files
- Modified `Dockerfile` - Correctly copies files from docker_files/
- Modified `.dockerignore` - Allows docker_files/ to be included

## Manual Build (if needed)
```bash
# Step 1: Prepare files
prepare_docker_files.bat

# Step 2: Build Docker image
docker build -t vaclavikmasa/ergonomic-analysis:v4.2 .

# Step 3: Push to Docker Hub
docker push vaclavikmasa/ergonomic-analysis:v4.2

# Step 4: Clean up
cleanup_docker_files.bat
```

## Verification
The Docker image includes:
- 7 Python modules for processing
- 1 SMPL-X model (NEUTRAL, 108MB)
- All RunPod handler code
- All dependencies from requirements_v3.txt
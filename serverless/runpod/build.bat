@echo off
REM Build script for RunPod Docker image
REM Stages files from parent directory and builds Docker image

echo Preparing Docker build context...

REM Create temporary build directory
set BUILD_DIR=build_context
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
mkdir %BUILD_DIR%

REM Copy Dockerfile and requirements
copy Dockerfile %BUILD_DIR%\
copy requirements-frozen.txt %BUILD_DIR%\
copy handler.py %BUILD_DIR%\
copy handler_local_test.py %BUILD_DIR%\

REM Copy required Python files from parent directory
copy ..\..\run_production_simple.py %BUILD_DIR%\
copy ..\..\production_3d_pipeline_clean.py %BUILD_DIR%\
copy ..\..\create_combined_angles_csv_skin.py %BUILD_DIR%\
copy ..\..\trunk_angle_calculator_skin.py %BUILD_DIR%\
copy ..\..\neck_angle_calculator_skin.py %BUILD_DIR%\
copy ..\..\arm_angle_calculator.py %BUILD_DIR%\

REM Copy SMPL-X models if they exist
if exist "..\..\models\smplx" (
    echo Copying SMPL-X models...
    mkdir %BUILD_DIR%\models\smplx
    copy ..\..\models\smplx\*.npz %BUILD_DIR%\models\smplx\ >nul 2>&1
)

REM Create final Dockerfile
(
echo # RunPod Serverless Docker Image for Ergonomic Analysis Pipeline
echo FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
echo.
echo WORKDIR /app
echo.
echo # Install system dependencies
echo RUN apt-get update ^&^& apt-get install -y \
echo     ffmpeg \
echo     libgl1-mesa-glx \
echo     libglib2.0-0 \
echo     libsm6 \
echo     libxext6 \
echo     libxrender-dev \
echo     libgomp1 \
echo     wget \
echo     ^&^& apt-get clean \
echo     ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo # Copy and install requirements
echo COPY requirements-frozen.txt .
echo RUN pip install --no-cache-dir -r requirements-frozen.txt
echo.
echo # Copy all application files
echo COPY . .
echo.
echo # Set environment variables
echo ENV PYTHONUNBUFFERED=1
echo ENV CUDA_VISIBLE_DEVICES=0
echo.
echo CMD ["python", "-u", "handler.py"]
) > %BUILD_DIR%\Dockerfile

REM Build Docker image
cd %BUILD_DIR%
echo Building Docker image...
docker build -t ergonomic-analyzer:latest .

REM Cleanup
cd ..
rmdir /s /q %BUILD_DIR%

echo Build complete! Image: ergonomic-analyzer:latest
echo.
echo To push to Docker Hub:
echo   docker tag ergonomic-analyzer:latest YOUR_USERNAME/ergonomic-analyzer:latest
echo   docker push YOUR_USERNAME/ergonomic-analyzer:latest
@echo off
echo ============================================
echo Building and pushing Docker Image V4.8
echo ============================================
echo Version 4.8 - Fixed Unhealthy Worker (PyTorch3D install)
echo.

REM Step 1: Prepare files
echo Step 1: Preparing Docker files...
call prepare_docker_files.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to prepare Docker files!
    exit /b 1
)

echo.
echo Step 2: Building Docker image...
docker build -t vaclavikmasa/ergonomic-analysis:v4.8 .
if %ERRORLEVEL% NEQ 0 (
    echo Docker build failed!
    exit /b 1
)

echo.
echo Step 3: Tagging as latest...
docker tag vaclavikmasa/ergonomic-analysis:v4.8 vaclavikmasa/ergonomic-analysis:latest

echo.
echo Step 4: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4.8
if %ERRORLEVEL% NEQ 0 (
    echo Docker push failed!
    exit /b 1
)

docker push vaclavikmasa/ergonomic-analysis:latest
if %ERRORLEVEL% NEQ 0 (
    echo Docker push latest failed!
    exit /b 1
)

echo.
echo Step 5: Cleaning up Docker files...
call cleanup_docker_files.bat

echo.
echo ============================================
echo SUCCESS! V4.8 built and pushed to Docker Hub
echo ============================================
echo.
echo Image: vaclavikmasa/ergonomic-analysis:v4.8
echo.
echo CHANGES IN V4.8:
echo - FIX: Correctly install pytorch3d via a direct Dockerfile command.
echo - This resolves the 'unhealthy worker' issue from v4.7.
echo.
echo Next steps for RunPod:
echo 1. Go to RunPod console
echo 2. Update serverless endpoint to use v4.8
echo 3. Restart the endpoint
echo.
pause
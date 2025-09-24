@echo off
echo ============================================
echo Building and pushing Docker Image V4.6
echo ============================================
echo Version 4.6 - Fixed missing module bug
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
docker build -t vaclavikmasa/ergonomic-analysis:v4.6 .
if %ERRORLEVEL% NEQ 0 (
    echo Docker build failed!
    exit /b 1
)

echo.
echo Step 3: Tagging as latest...
docker tag vaclavikmasa/ergonomic-analysis:v4.6 vaclavikmasa/ergonomic-analysis:latest

echo.
echo Step 4: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4.6
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
echo SUCCESS! V4.6 built and pushed to Docker Hub
echo ============================================
echo.
echo Image: vaclavikmasa/ergonomic-analysis:v4.6
echo.
echo CHANGES IN V4.6:
echo - FIX: Included generate_4videos_from_pkl.py in the Docker image.
echo - This resolves the 'module not found' error from the previous version.
echo.
echo Next steps for RunPod:
echo 1. Go to RunPod console
echo 2. Update serverless endpoint to use v4.6
echo 3. Restart the endpoint
echo.
pause
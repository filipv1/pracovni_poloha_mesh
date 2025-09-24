@echo off
echo ============================================
echo Building and pushing Docker Image V4.3
echo ============================================
echo Version 4.3 - Without video generation
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
docker build -t vaclavikmasa/ergonomic-analysis:v4.3 .
if %ERRORLEVEL% NEQ 0 (
    echo Docker build failed!
    exit /b 1
)

echo.
echo Step 3: Tagging as latest...
docker tag vaclavikmasa/ergonomic-analysis:v4.3 vaclavikmasa/ergonomic-analysis:latest

echo.
echo Step 4: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4.3
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
echo SUCCESS! V4.3 built and pushed to Docker Hub
echo ============================================
echo.
echo Image: vaclavikmasa/ergonomic-analysis:v4.3
echo.
echo CHANGES IN V4.3:
echo - Removed video generation (problematic PKL format)
echo - Only outputs: PKL, CSV, XLSX
echo - Faster processing without video generation
echo.
echo Next steps for RunPod:
echo 1. Go to RunPod console
echo 2. Update serverless endpoint to use v4.3
echo 3. Restart the endpoint
echo.
pause
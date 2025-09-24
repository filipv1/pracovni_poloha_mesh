@echo off
echo Force update Docker image with new tag

set TAG=v3latestv2
set IMAGE_NAME=vaclavikmasa/ergonomic-analysis-v3

echo Building with tag: %TAG%
cd ..
docker build -f serverless_v3/Dockerfile -t %IMAGE_NAME%:%TAG% .

echo Pushing to Docker Hub...
docker push %IMAGE_NAME%:%TAG%

echo.
echo ========================================
echo DONE! New image pushed as:
echo %IMAGE_NAME%:%TAG%
echo.
echo NOW UPDATE RUNPOD:
echo 1. Go to RunPod dashboard
echo 2. Edit endpoint
echo 3. Change container image to:
echo    vaclavikmasa/ergonomic-analysis-v3:v3latestv2
echo 4. Click Update
echo ========================================
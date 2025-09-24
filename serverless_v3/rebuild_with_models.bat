@echo off
echo ============================================================
echo     REBUILDING V3 WITH SMPL-X MODELS
echo ============================================================
echo.
echo This will add ~104MB to the Docker image (SMPLX_NEUTRAL.npz)
echo.

set TAG=v3-with-models
set IMAGE_NAME=vaclavikmasa/ergonomic-analysis-v3

echo Building image with SMPL-X models...
cd ..
docker build -f serverless_v3/Dockerfile -t %IMAGE_NAME%:%TAG% .

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo Pushing to Docker Hub...
docker push %IMAGE_NAME%:%TAG%

echo.
echo ============================================================
echo            DONE! New image ready:
echo     %IMAGE_NAME%:%TAG%
echo.
echo     UPDATE RUNPOD ENDPOINT TO USE THIS IMAGE!
echo ============================================================
echo.
pause
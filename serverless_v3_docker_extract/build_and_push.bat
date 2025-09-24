@echo off
echo ============================================
echo Building and pushing Docker Image V4.5-STABLE
echo ============================================
echo Production version - Fixed missing angle calculators v4.5
echo.

REM Step 1: Copy SMPL-X models if available
echo Step 1: Checking for SMPL-X models...
if not exist "models" mkdir models
if not exist "models\smplx" mkdir models\smplx

if exist "..\models\smplx\SMPLX_NEUTRAL.npz" (
    echo Copying SMPLX_NEUTRAL.npz...
    copy "..\models\smplx\SMPLX_NEUTRAL.npz" "models\smplx\" >nul
    echo ✅ SMPL-X model copied
) else (
    echo ⚠️  SMPL-X model not found - will be downloaded at runtime
)

echo.
echo Step 2: Building Docker image...
docker build -t vaclavikmasa/ergonomic-analysis:v4.5-stable .
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Tagging as latest...
docker tag vaclavikmasa/ergonomic-analysis:v4.5-stable vaclavikmasa/ergonomic-analysis:latest

echo.
echo Step 4: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4.5-stable
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker push failed!
    pause
    exit /b 1
)

docker push vaclavikmasa/ergonomic-analysis:latest
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker push latest failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! V4.5-STABLE built and pushed to Docker Hub
echo ============================================
echo.
echo Image: vaclavikmasa/ergonomic-analysis:v4.5-stable
echo.
echo PRODUCTION FEATURES:
echo - 3-step pipeline: PKL + CSV + Excel
echo - Skin-based angle calculations
echo - Czech ergonomic analysis
echo - Stable RunPod handler
echo - FIXED: All angle calculator modules included
echo.
echo Next steps for RunPod:
echo 1. Go to RunPod console
echo 2. Update serverless endpoint to use v4.5-stable
echo 3. Test with sample video
echo.
pause
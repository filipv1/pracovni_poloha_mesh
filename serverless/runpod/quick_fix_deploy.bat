@echo off
REM Quick fix and deploy script
REM Fastest way to fix and deploy to RunPod

echo ========================================
echo QUICK FIX AND DEPLOY
echo ========================================
echo.
echo This will:
echo 1. Use handler_fixed.py (no progress_update bugs)
echo 2. Build Docker image
echo 3. Push to Docker Hub
echo 4. RunPod will auto-update in 1-2 minutes
echo.
pause

echo.
echo Step 1: Using fixed handler...
copy /Y handler_fixed.py handler.py
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy handler_fixed.py
    pause
    exit /b 1
)
echo [OK] Handler copied

echo.
echo Step 2: Building Docker image...
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed
    echo.
    echo Common fixes:
    echo - Make sure Docker Desktop is running
    echo - Check Dockerfile syntax
    echo - Ensure all required .py files are in this directory
    pause
    exit /b 1
)
echo [OK] Docker image built

echo.
echo Step 3: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Docker push failed
    echo.
    echo Common fixes:
    echo - Run: docker login
    echo - Check internet connection
    echo - Verify Docker Hub credentials
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] DEPLOYMENT COMPLETE!
echo ========================================
echo.
echo RunPod will automatically update in 1-2 minutes
echo.
echo To test:
echo 1. Wait 2 minutes
echo 2. Run: powershell .\test_runpod_api.ps1 -TestType submit
echo 3. Or use the web interface
echo.
pause
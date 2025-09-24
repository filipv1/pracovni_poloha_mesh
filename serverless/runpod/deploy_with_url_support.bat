@echo off
REM Deploy with URL support for large files
REM Supports files > 10MB via temporary URL storage

echo ========================================
echo DEPLOY WITH URL SUPPORT FOR LARGE FILES
echo ========================================
echo.
echo This deployment includes:
echo - URL download support for large videos
echo - file.io/transfer.sh integration
echo - Fixed handler without progress_update bugs
echo.
pause

echo.
echo Step 1: Building Docker image...
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed
    pause
    exit /b 1
)
echo [OK] Docker image built

echo.
echo Step 2: Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Docker push failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] DEPLOYMENT COMPLETE!
echo ========================================
echo.
echo Features enabled:
echo - Small files (<7MB): Direct base64 upload
echo - Large files (>7MB): URL upload via file.io
echo - Maximum file size: 100MB
echo.
echo RunPod will update in 1-2 minutes
echo.
echo To test:
echo 1. Start proxy: python proxy-server-url.py
echo 2. Open: index-with-proxy.html
echo 3. Upload any size video (up to 100MB)
echo.
pause
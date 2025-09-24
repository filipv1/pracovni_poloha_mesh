@echo off
REM Login to Docker Hub to avoid rate limits

echo ========================================
echo DOCKER HUB LOGIN AND PUSH
echo ========================================
echo.
echo Logging in avoids rate limits!
echo.

echo Step 1: Login to Docker Hub...
docker login
if %errorlevel% neq 0 (
    echo [ERROR] Docker login failed
    echo.
    echo Please create account at: https://hub.docker.com
    pause
    exit /b 1
)

echo.
echo Step 2: Push image (authenticated = no limits)...
docker push vaclavikmasa/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Push failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Pushed with authentication!
echo ========================================
echo.
echo RunPod should now be able to pull without limits
echo (if RunPod is also logged in)
echo.
pause
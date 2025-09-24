@echo off
REM Push to GitHub Container Registry (no rate limits)
REM Free for public repositories

echo ========================================
echo PUSH TO GITHUB CONTAINER REGISTRY
echo ========================================
echo.
echo GitHub registry has NO rate limits!
echo.

echo Step 1: Login to GitHub registry...
echo You need a GitHub Personal Access Token with:
echo - read:packages
echo - write:packages
echo - delete:packages (optional)
echo.
echo Create token at: https://github.com/settings/tokens
echo.
set /p GITHUB_TOKEN="Enter your GitHub token: "
set /p GITHUB_USER="Enter your GitHub username: "

echo %GITHUB_TOKEN% | docker login ghcr.io -u %GITHUB_USER% --password-stdin
if %errorlevel% neq 0 (
    echo [ERROR] GitHub login failed
    pause
    exit /b 1
)

echo.
echo Step 2: Tag image for GitHub registry...
docker tag vaclavikmasa/ergonomic-analyzer:latest ghcr.io/%GITHUB_USER%/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Tagging failed
    pause
    exit /b 1
)

echo.
echo Step 3: Push to GitHub registry...
docker push ghcr.io/%GITHUB_USER%/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Push failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Pushed to GitHub Registry!
echo ========================================
echo.
echo Image URL: ghcr.io/%GITHUB_USER%/ergonomic-analyzer:latest
echo.
echo Update this in RunPod console:
echo 1. Go to RunPod endpoint settings
echo 2. Change Docker image to: ghcr.io/%GITHUB_USER%/ergonomic-analyzer:latest
echo 3. Save and restart endpoint
echo.
pause
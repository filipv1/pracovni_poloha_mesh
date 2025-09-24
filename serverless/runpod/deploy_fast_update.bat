@echo off
REM Fast deploy - only updates changed layers
REM Uses optimized Dockerfile for better caching

echo ========================================
echo FAST UPDATE DEPLOYMENT
echo ========================================
echo.
echo Only updates changed code (not dependencies)
echo.

REM Use optimized Dockerfile if it exists
if exist Dockerfile.optimized (
    echo Using optimized Dockerfile...
    docker build -f Dockerfile.optimized -t vaclavikmasa/ergonomic-analyzer:latest .
) else (
    echo Using standard Dockerfile...
    docker build -t vaclavikmasa/ergonomic-analyzer:latest .
)

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo Pushing to Docker Hub (only changed layers)...
docker push vaclavikmasa/ergonomic-analyzer:latest

if %errorlevel% neq 0 (
    echo [ERROR] Push failed
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Fast update complete!
echo Only changed layers were uploaded.
echo.
pause
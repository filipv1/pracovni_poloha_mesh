@echo off
REM Test and Deploy script for RunPod handler
REM Windows compatible - no emojis, no diacritics

echo ========================================
echo RunPod Handler Test and Deploy
echo ========================================

:menu
echo.
echo 1. Test handlers locally
echo 2. Test in Docker container
echo 3. Build and push Docker image
echo 4. Quick deploy (build + push)
echo 5. Exit
echo.
set /p choice="Select option (1-5): "

if "%choice%"=="1" goto test_local
if "%choice%"=="2" goto test_docker
if "%choice%"=="3" goto build_push
if "%choice%"=="4" goto quick_deploy
if "%choice%"=="5" exit /b
echo Invalid choice
goto menu

:test_local
echo.
echo Testing handlers locally...
echo ----------------------------------------
python test_handler_local.py
if %errorlevel% neq 0 (
    echo [ERROR] Tests failed
) else (
    echo [OK] Tests passed
)
pause
goto menu

:test_docker
echo.
echo Testing in Docker container...
echo ----------------------------------------
echo Building test image...
docker build -t test-handler .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed
    pause
    goto menu
)

echo Running minimal handler test...
docker run --rm test-handler python handler_minimal.py
if %errorlevel% neq 0 (
    echo [ERROR] Minimal handler test failed
) else (
    echo [OK] Minimal handler works
)
pause
goto menu

:build_push
echo.
echo Building and pushing Docker image...
echo ----------------------------------------
echo Which handler to use?
echo 1. handler_minimal.py (for testing)
echo 2. handler_fixed.py (full version)
echo 3. handler.py (original)
set /p handler_choice="Select (1-3): "

if "%handler_choice%"=="1" (
    copy /Y handler_minimal.py handler.py
    echo Using minimal handler
) else if "%handler_choice%"=="2" (
    copy /Y handler_fixed.py handler.py
    echo Using fixed handler
) else (
    echo Using original handler
)

echo.
echo Building Docker image...
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    pause
    goto menu
)

echo.
echo Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Push failed
) else (
    echo [OK] Image pushed successfully
    echo.
    echo IMPORTANT: RunPod will automatically pull the new image
    echo Wait 1-2 minutes before testing
)
pause
goto menu

:quick_deploy
echo.
echo Quick Deploy - Using handler_fixed.py
echo ----------------------------------------
copy /Y handler_fixed.py handler.py
echo Building image...
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    pause
    goto menu
)

echo Pushing image...
docker push vaclavikmasa/ergonomic-analyzer:latest
if %errorlevel% neq 0 (
    echo [ERROR] Push failed
) else (
    echo [SUCCESS] Deployed successfully!
    echo Wait 1-2 minutes for RunPod to update
)
pause
goto menu
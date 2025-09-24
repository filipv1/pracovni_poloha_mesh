@echo off
REM Prepare SMPL-X models for Docker build

echo ========================================
echo PREPARING SMPL-X MODELS
echo ========================================
echo.

REM Create models directory
if not exist "models\smplx" (
    mkdir models\smplx
    echo Created models\smplx directory
)

echo.
echo Copy your SMPL-X model files here:
echo   serverless\runpod\models\smplx\
echo.
echo Required files:
echo   - SMPLX_NEUTRAL.npz
echo   - SMPLX_MALE.npz
echo   - SMPLX_FEMALE.npz
echo.
echo From your local directory, copy them:
echo.

REM Try to find models in common locations
if exist "..\..\models\smplx\*.npz" (
    echo Found models in ..\..\models\smplx\
    copy /Y "..\..\models\smplx\*.npz" "models\smplx\"
    copy /Y "..\..\models\smplx\*.pkl" "models\smplx\" 2>nul
    echo Models copied!
) else if exist "C:\Users\vaclavik\ruce7\pracovni_poloha_mesh\models\smplx\*.npz" (
    echo Found models in main project directory
    copy /Y "C:\Users\vaclavik\ruce7\pracovni_poloha_mesh\models\smplx\*.npz" "models\smplx\"
    copy /Y "C:\Users\vaclavik\ruce7\pracovni_poloha_mesh\models\smplx\*.pkl" "models\smplx\" 2>nul
    echo Models copied!
) else (
    echo Models not found automatically!
    echo.
    echo Please copy manually:
    echo   FROM: Your SMPL-X models location
    echo   TO:   serverless\runpod\models\smplx\
    echo.
    explorer models\smplx
)

echo.
echo Checking models...
dir models\smplx\*.npz 2>nul

if not exist "models\smplx\SMPLX_NEUTRAL.npz" (
    echo.
    echo [WARNING] SMPLX_NEUTRAL.npz not found!
    echo Please copy the model files before building Docker image
    pause
    exit /b 1
)

echo.
echo [OK] Models ready for Docker build!
echo.
echo Next steps:
echo 1. Run: docker build -f Dockerfile.with-models -t vaclavikmasa/ergonomic-analyzer:with-models .
echo 2. Run: docker push vaclavikmasa/ergonomic-analyzer:with-models
echo 3. Update RunPod to use: vaclavikmasa/ergonomic-analyzer:with-models
echo.
pause
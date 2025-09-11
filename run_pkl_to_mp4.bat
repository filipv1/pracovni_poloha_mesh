@echo off
REM Quick test script for PKL to MP4 pipeline

echo =====================================
echo PKL to MP4 Pipeline Test
echo =====================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Default test with fpsmeshes.pkl if it exists
if exist "fpsmeshes.pkl" (
    echo Using test file: fpsmeshes.pkl
    python pkl_to_mp4_pipeline.py fpsmeshes.pkl test_output.mp4 --quality low
) else if exist "arm_meshes.pkl" (
    echo Using test file: arm_meshes.pkl
    python pkl_to_mp4_pipeline.py arm_meshes.pkl test_output.mp4 --quality low
) else (
    echo ERROR: No PKL file found for testing!
    echo Please provide a PKL file as argument.
    pause
    exit /b 1
)

echo.
echo =====================================
echo Pipeline completed!
echo Check test_output.mp4
echo =====================================
pause
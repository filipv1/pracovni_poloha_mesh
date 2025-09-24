@echo off
REM Compress video to fit RunPod 10MB limit (target 5MB for safety)

if "%1"=="" (
    echo Usage: compress_for_runpod.bat input_video.mp4
    echo.
    echo This will create a compressed version that works with RunPod
    pause
    exit /b 1
)

set INPUT=%1
set OUTPUT=%~n1_compressed.mp4

echo ========================================
echo VIDEO COMPRESSION FOR RUNPOD
echo ========================================
echo Input:  %INPUT%
echo Output: %OUTPUT%
echo Target: ~5MB (safe for RunPod)
echo ========================================
echo.

REM Option 1: First 30 seconds, lower quality
echo Creating 30-second preview...
ffmpeg -i "%INPUT%" -t 30 -vcodec libx264 -crf 28 -preset fast -vf scale=640:-1 -y "%OUTPUT%"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compression failed
    echo Make sure ffmpeg is installed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Compression complete!
echo ========================================

REM Check file size
for %%A in ("%OUTPUT%") do set SIZE=%%~zA
set /a SIZE_MB=%SIZE%/1048576

echo Output file: %OUTPUT%
echo Size: ~%SIZE_MB%MB

if %SIZE_MB% GTR 7 (
    echo.
    echo [WARNING] File still too large!
    echo Try shorter duration or lower resolution
)

echo.
echo You can now upload this file to the web app
echo.
pause
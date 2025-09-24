@echo off
echo ============================================
echo Cleaning up Docker build files
echo ============================================

if exist docker_files (
    echo Removing docker_files directory...
    rmdir /S /Q docker_files
    echo Cleaned up successfully!
) else (
    echo Nothing to clean up.
)

echo ============================================
echo Cleanup complete
echo ============================================
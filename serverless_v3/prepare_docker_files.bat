@echo off
echo ============================================
echo Preparing Docker files for V4.2 build
echo ============================================

REM Clean up previous docker_files directory
if exist docker_files (
    echo Cleaning up old docker_files directory...
    rmdir /S /Q docker_files
)

REM Create docker_files directory structure
echo Creating docker_files directory structure...
mkdir docker_files
mkdir docker_files\models
mkdir docker_files\models\smplx

REM Copy Python modules from parent directory
echo.
echo Copying Python modules...
copy ..\run_production_simple_p.py docker_files\ >nul
copy ..\create_combined_angles_csv_skin.py docker_files\ >nul
copy ..\ergonomic_time_analysis.py docker_files\ >nul
copy ..\generate_4videos_from_pkl.py docker_files\ >nul
copy ..\arm_angle_calculator.py docker_files\ >nul
copy ..\neck_angle_calculator_like_arm.py docker_files\ >nul
copy ..\neck_angle_calculator_skin.py docker_files\ >nul

REM Copy SMPL-X NEUTRAL model
echo.
echo Copying SMPL-X NEUTRAL model (108MB)...
copy ..\models\smplx\SMPLX_NEUTRAL.npz docker_files\models\smplx\ >nul

REM Verify all files were copied
echo.
echo Verifying copied files...
set ERROR=0

if not exist docker_files\run_production_simple_p.py (
    echo ERROR: run_production_simple_p.py not copied!
    set ERROR=1
)
if not exist docker_files\create_combined_angles_csv_skin.py (
    echo ERROR: create_combined_angles_csv_skin.py not copied!
    set ERROR=1
)
if not exist docker_files\ergonomic_time_analysis.py (
    echo ERROR: ergonomic_time_analysis.py not copied!
    set ERROR=1
)
if not exist docker_files\generate_4videos_from_pkl.py (
    echo ERROR: generate_4videos_from_pkl.py not copied!
    set ERROR=1
)
if not exist docker_files\arm_angle_calculator.py (
    echo ERROR: arm_angle_calculator.py not copied!
    set ERROR=1
)
if not exist docker_files\neck_angle_calculator_like_arm.py (
    echo ERROR: neck_angle_calculator_like_arm.py not copied!
    set ERROR=1
)
if not exist docker_files\neck_angle_calculator_skin.py (
    echo ERROR: neck_angle_calculator_skin.py not copied!
    set ERROR=1
)
if not exist docker_files\models\smplx\SMPLX_NEUTRAL.npz (
    echo ERROR: SMPLX_NEUTRAL.npz not copied!
    set ERROR=1
)

if %ERROR%==0 (
    echo.
    echo ============================================
    echo SUCCESS! All files prepared for Docker build
    echo ============================================
    echo.
    echo Directory structure:
    dir docker_files /B
    echo.
    dir docker_files\models\smplx /B
    echo.
    echo Ready for Docker build!
) else (
    echo.
    echo ============================================
    echo FAILED! Some files were not copied
    echo ============================================
    exit /b 1
)
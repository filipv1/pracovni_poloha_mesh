@echo off
REM Render OBJ files to MP4 directly (without PKL export)
REM Usage: render_obj_to_mp4.bat [output_name]

set OUTPUT=%1
if "%OUTPUT%"=="" set OUTPUT=output_video.mp4

echo =====================================
echo OBJ to MP4 Renderer
echo =====================================
echo.
echo Input: blender_export_skin_5614\*.obj
echo Output: %OUTPUT%
echo.

"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python blender_headless_render.py -- --obj_dir blender_export_skin_5614 --output %OUTPUT% --fps 25 --resolution_x 1280 --resolution_y 720 --samples 32

echo.
echo =====================================
echo Render complete!
echo Check for: %OUTPUT% (with frame numbers)
echo =====================================
pause
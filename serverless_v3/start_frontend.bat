@echo off
echo Starting local web server for frontend...
echo.
echo Frontend will be available at: http://localhost:8000
echo.
cd frontend
python -m http.server 8000
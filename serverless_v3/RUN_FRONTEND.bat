@echo off
cls
echo ============================================================
echo           V3 FRONTEND LAUNCHER
echo ============================================================
echo.
echo Starting local web server to avoid CORS issues...
echo.
echo After server starts, open your browser at:
echo.
echo   http://localhost:8000/index-configured.html
echo.
echo ============================================================
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

cd frontend
python -m http.server 8000
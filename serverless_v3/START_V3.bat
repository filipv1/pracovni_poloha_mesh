@echo off
cls
echo ============================================================
echo                 V3 SERVERLESS LAUNCHER
echo ============================================================
echo.
echo Tento skript spusti:
echo 1. Proxy server (resi CORS problemy)
echo 2. Frontend v prohlizeci
echo.
echo ============================================================
echo.

REM Install dependencies if needed
echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask flask-cors
)

echo.
echo Starting proxy server in new window...
start cmd /k "cd %~dp0 && python full_proxy.py"

timeout /t 3 >nul

echo Starting frontend server in new window...
start cmd /k "cd %~dp0\frontend && python -m http.server 8000"

timeout /t 3 >nul

echo.
echo ============================================================
echo                    V3 IS RUNNING!
echo ============================================================
echo.
echo Proxy server: http://localhost:5001
echo Frontend: http://localhost:8000/index-with-proxy.html
echo.
echo Opening browser...
start http://localhost:8000/index-with-proxy.html
echo.
echo ============================================================
echo Press any key to stop all services...
pause >nul

taskkill /f /im python.exe 2>nul
echo.
echo All services stopped.
@echo off
setlocal enabledelayedexpansion

echo =========================================
echo Ergonomic Analyzer - Serverless Deployment
echo =========================================

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found. Please copy .env.example and configure it.
    exit /b 1
)

REM Load environment variables from .env
for /f "tokens=1,2 delims==" %%a in (.env) do (
    set %%a=%%b
)

:menu
echo.
echo Select deployment target:
echo 1. Full deployment (RunPod + Cloudflare + Frontend)
echo 2. RunPod only
echo 3. Cloudflare Worker only
echo 4. Frontend only
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto full
if "%choice%"=="2" goto runpod
if "%choice%"=="3" goto cloudflare
if "%choice%"=="4" goto frontend
echo Invalid choice
goto menu

:full
call :deploy_runpod
set /p RUNPOD_ENDPOINT_ID="Enter RunPod Endpoint ID: "
echo RUNPOD_ENDPOINT_ID=%RUNPOD_ENDPOINT_ID%>> .env
call :deploy_cloudflare
call :deploy_frontend
goto done

:runpod
call :deploy_runpod
goto done

:cloudflare
call :deploy_cloudflare
goto done

:frontend
call :deploy_frontend
goto done

:deploy_runpod
echo.
echo 1. Deploying to RunPod...
echo --------------------------

cd runpod

echo Building Docker image...
docker build -t %DOCKER_USERNAME%/ergonomic-analyzer:latest .

echo Pushing to Docker Hub...
docker push %DOCKER_USERNAME%/ergonomic-analyzer:latest

echo √ Docker image deployed to Docker Hub
echo.
echo Next steps for RunPod:
echo 1. Go to https://runpod.io/console/serverless
echo 2. Create new endpoint
echo 3. Use image: %DOCKER_USERNAME%/ergonomic-analyzer:latest
echo 4. Set GPU: A10G or RTX 4090
echo 5. Copy the endpoint ID

cd ..
exit /b 0

:deploy_cloudflare
echo.
echo 2. Deploying Cloudflare Worker...
echo ---------------------------------

cd cloudflare

REM Check if wrangler is installed
where wrangler >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Wrangler CLI...
    npm install -g wrangler
)

echo Logging into Cloudflare...
wrangler login

echo Creating KV namespaces...
for /f "tokens=3 delims==" %%a in ('wrangler kv:namespace create "JOBS" --preview false ^| findstr "id"') do set JOBS_KV_ID=%%a
for /f "tokens=3 delims==" %%a in ('wrangler kv:namespace create "LOGS" --preview false ^| findstr "id"') do set LOGS_KV_ID=%%a

REM Update wrangler.toml
powershell -Command "(Get-Content wrangler.toml) -replace 'YOUR_JOBS_KV_ID', '%JOBS_KV_ID%' | Set-Content wrangler.toml"
powershell -Command "(Get-Content wrangler.toml) -replace 'YOUR_LOGS_KV_ID', '%LOGS_KV_ID%' | Set-Content wrangler.toml"
powershell -Command "(Get-Content wrangler.toml) -replace 'YOUR_SUBDOMAIN', '%CF_SUBDOMAIN%' | Set-Content wrangler.toml"
powershell -Command "(Get-Content wrangler.toml) -replace 'YOUR_USERNAME', '%GITHUB_USERNAME%' | Set-Content wrangler.toml"

echo Creating R2 bucket...
wrangler r2 bucket create ergonomic-results

echo Setting secrets...
echo %RUNPOD_API_KEY%| wrangler secret put RUNPOD_API_KEY
echo %RUNPOD_ENDPOINT_ID%| wrangler secret put RUNPOD_ENDPOINT_ID
echo %RESEND_API_KEY%| wrangler secret put RESEND_API_KEY
echo %AUTH_PASSWORD_HASH%| wrangler secret put AUTH_PASSWORD_HASH
echo %JWT_SECRET%| wrangler secret put JWT_SECRET

echo Deploying worker...
wrangler deploy

echo √ Cloudflare Worker deployed

cd ..
exit /b 0

:deploy_frontend
echo.
echo 3. Deploying Frontend to GitHub Pages...
echo ----------------------------------------

cd frontend

REM Update API URL in index.html
powershell -Command "(Get-Content index.html) -replace 'YOUR_SUBDOMAIN', '%CF_SUBDOMAIN%' | Set-Content index.html"

echo Please manually:
echo 1. Commit frontend/index.html to gh-pages branch
echo 2. Push to GitHub
echo 3. Enable GitHub Pages for gh-pages branch

echo √ Frontend ready for deployment
echo URL: https://%GITHUB_USERNAME%.github.io/ergonomic-analyzer

cd ..
exit /b 0

:done
echo.
echo =========================================
echo Deployment Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Test the frontend: https://%GITHUB_USERNAME%.github.io/ergonomic-analyzer
echo 2. Monitor logs: wrangler tail
echo 3. Check RunPod: https://runpod.io/console/serverless
echo.

endlocal
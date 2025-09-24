@echo OFF
set IMAGE_NAME=vaclavikmasa/ergonomic-analyzer:v4-s3-polling

echo Building Docker image: %IMAGE_NAME%

docker build -f serverless_v2/Dockerfile.v2 -t %IMAGE_NAME% .

echo.
echo Build complete.

@echo OFF
set IMAGE_NAME=vaclavikmasa/ergonomic-analyzer:v4-s3-polling

echo Pushing Docker image to Docker Hub: %IMAGE_NAME%

docker push %IMAGE_NAME%

echo.
echo Push complete.

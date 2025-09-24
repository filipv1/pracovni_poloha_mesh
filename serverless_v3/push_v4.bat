@echo off
echo ========================================
echo Pushing Docker Image V4 to Docker Hub
echo ========================================
echo.

echo Tagging additional versions...
docker tag vaclavikmasa/ergonomic-analysis:v4 vaclavikmasa/ergonomic-analysis:4.0.0
docker tag vaclavikmasa/ergonomic-analysis:v4 vaclavikmasa/ergonomic-analysis:latest

echo.
echo Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4
docker push vaclavikmasa/ergonomic-analysis:4.0.0
docker push vaclavikmasa/ergonomic-analysis:latest

echo.
echo ========================================
echo SUCCESS! Images pushed to Docker Hub
echo ========================================
echo.
echo Use in RunPod: vaclavikmasa/ergonomic-analysis:v4
echo.
echo IMPORTANT: Change from :v3 or :latest to :v4 in RunPod!
echo.
pause
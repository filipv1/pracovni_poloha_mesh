@echo off
echo ========================================
echo Pushing Docker Image V4.1 to Docker Hub
echo ========================================
echo Fixed: Handler path issue
echo.

echo Pushing to Docker Hub...
docker push vaclavikmasa/ergonomic-analysis:v4.1

echo Also tagging as latest...
docker tag vaclavikmasa/ergonomic-analysis:v4.1 vaclavikmasa/ergonomic-analysis:latest
docker push vaclavikmasa/ergonomic-analysis:latest

echo.
echo ========================================
echo SUCCESS! V4.1 pushed to Docker Hub
echo ========================================
echo.
echo IMPORTANT FOR RUNPOD:
echo ======================
echo Use: vaclavikmasa/ergonomic-analysis:v4.1
echo.
echo This version fixes the handler path issue!
echo.
pause
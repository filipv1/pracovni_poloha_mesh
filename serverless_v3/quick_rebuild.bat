@echo off
echo Quick rebuild and push Docker image for V3

cd ..
docker build -f serverless_v3/Dockerfile -t vaclavikmasa/ergonomic-analysis-v3:latest .
docker push vaclavikmasa/ergonomic-analysis-v3:latest

echo Done! Image pushed to Docker Hub
echo Update RunPod endpoint to use new image version
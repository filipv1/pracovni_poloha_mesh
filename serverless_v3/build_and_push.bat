@echo off

REM Build and push Docker image for V3 architecture (Windows)

set DOCKER_USERNAME=vaclavik
set IMAGE_NAME=ergonomic-analysis-v3
set TAG=latest

echo Building Docker image for V3 architecture...
docker build -f Dockerfile -t %IMAGE_NAME%:%TAG% ..

echo Tagging image...
docker tag %IMAGE_NAME%:%TAG% %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%

echo Logging in to Docker Hub...
docker login

echo Pushing image to Docker Hub...
docker push %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%

echo Done! Image available at: %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%

REM Also create a versioned tag
set VERSION_TAG=v3.0.0
docker tag %IMAGE_NAME%:%TAG% %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION_TAG%
docker push %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION_TAG%

echo Also tagged as: %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION_TAG%
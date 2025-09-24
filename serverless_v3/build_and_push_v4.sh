#!/bin/bash
# Build and push Docker image V4 to RunPod
# This forces RunPod to pull the new version

echo "========================================"
echo "Building RunPod Docker Image V4"
echo "Version: 4.0.0"
echo "Date: 2025-01-23"
echo "========================================"

# Docker Hub credentials (update with your credentials)
DOCKER_USER="your_docker_username"
DOCKER_REPO="ergonomic-analysis"
VERSION="4.0.0"
TAG="v4-extended-pipeline"

echo "Building Docker image..."
docker build -t $DOCKER_USER/$DOCKER_REPO:$TAG .
docker build -t $DOCKER_USER/$DOCKER_REPO:$VERSION .
docker build -t $DOCKER_USER/$DOCKER_REPO:latest .

echo "Logging in to Docker Hub..."
docker login

echo "Pushing images..."
docker push $DOCKER_USER/$DOCKER_REPO:$TAG
docker push $DOCKER_USER/$DOCKER_REPO:$VERSION
docker push $DOCKER_USER/$DOCKER_REPO:latest

echo "========================================"
echo "Docker images pushed successfully!"
echo "Use this in RunPod: $DOCKER_USER/$DOCKER_REPO:$TAG"
echo "========================================"

# Instructions for RunPod update
echo ""
echo "TO UPDATE RUNPOD:"
echo "1. Go to RunPod console"
echo "2. Stop your current serverless endpoint"
echo "3. Update Docker image to: $DOCKER_USER/$DOCKER_REPO:$TAG"
echo "4. Restart the endpoint"
echo ""
echo "This will force RunPod to pull the new V4 image with extended pipeline"
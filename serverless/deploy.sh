#!/bin/bash

# Deployment script for Ergonomic Analyzer Serverless

set -e

echo "========================================="
echo "Ergonomic Analyzer - Serverless Deployment"
echo "========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.example and configure it."
    exit 1
fi

# Load environment variables
source .env

# Function to deploy RunPod
deploy_runpod() {
    echo ""
    echo "1. Deploying to RunPod..."
    echo "--------------------------"

    cd runpod

    # Build Docker image
    echo "Building Docker image..."
    docker build -t ${DOCKER_USERNAME}/ergonomic-analyzer:latest .

    # Push to Docker Hub
    echo "Pushing to Docker Hub..."
    docker push ${DOCKER_USERNAME}/ergonomic-analyzer:latest

    echo "✓ Docker image deployed to Docker Hub"
    echo ""
    echo "Next steps for RunPod:"
    echo "1. Go to https://runpod.io/console/serverless"
    echo "2. Create new endpoint"
    echo "3. Use image: ${DOCKER_USERNAME}/ergonomic-analyzer:latest"
    echo "4. Set GPU: A10G or RTX 4090"
    echo "5. Copy the endpoint ID"

    cd ..
}

# Function to deploy Cloudflare Worker
deploy_cloudflare() {
    echo ""
    echo "2. Deploying Cloudflare Worker..."
    echo "---------------------------------"

    cd cloudflare

    # Check if wrangler is installed
    if ! command -v wrangler &> /dev/null; then
        echo "Installing Wrangler CLI..."
        npm install -g wrangler
    fi

    # Login to Cloudflare
    echo "Logging into Cloudflare..."
    wrangler login

    # Create KV namespaces if they don't exist
    echo "Creating KV namespaces..."
    JOBS_KV_ID=$(wrangler kv:namespace create "JOBS" --preview false | grep -oP 'id = "\K[^"]+')
    LOGS_KV_ID=$(wrangler kv:namespace create "LOGS" --preview false | grep -oP 'id = "\K[^"]+')

    # Update wrangler.toml with KV IDs
    sed -i "s/YOUR_JOBS_KV_ID/${JOBS_KV_ID}/g" wrangler.toml
    sed -i "s/YOUR_LOGS_KV_ID/${LOGS_KV_ID}/g" wrangler.toml
    sed -i "s/YOUR_SUBDOMAIN/${CF_SUBDOMAIN}/g" wrangler.toml
    sed -i "s/YOUR_USERNAME/${GITHUB_USERNAME}/g" wrangler.toml

    # Create R2 bucket
    echo "Creating R2 bucket..."
    wrangler r2 bucket create ergonomic-results

    # Set secrets
    echo "Setting secrets..."
    echo ${RUNPOD_API_KEY} | wrangler secret put RUNPOD_API_KEY
    echo ${RUNPOD_ENDPOINT_ID} | wrangler secret put RUNPOD_ENDPOINT_ID
    echo ${RESEND_API_KEY} | wrangler secret put RESEND_API_KEY
    echo ${AUTH_PASSWORD_HASH} | wrangler secret put AUTH_PASSWORD_HASH
    echo ${JWT_SECRET} | wrangler secret put JWT_SECRET

    # Deploy worker
    echo "Deploying worker..."
    wrangler deploy

    echo "✓ Cloudflare Worker deployed"

    cd ..
}

# Function to deploy frontend
deploy_frontend() {
    echo ""
    echo "3. Deploying Frontend to GitHub Pages..."
    echo "----------------------------------------"

    cd frontend

    # Update API URL in index.html
    sed -i "s/YOUR_SUBDOMAIN/${CF_SUBDOMAIN}/g" index.html

    # Create gh-pages branch if it doesn't exist
    if ! git ls-remote --heads origin gh-pages | grep -q gh-pages; then
        git checkout --orphan gh-pages
        git rm -rf .
        echo "# Ergonomic Analyzer Frontend" > README.md
        git add README.md
        git commit -m "Initial gh-pages commit"
        git push origin gh-pages
        git checkout serverless-migration
    fi

    # Copy frontend files to temporary directory
    cp -r . /tmp/ergonomic-frontend

    # Switch to gh-pages branch
    git checkout gh-pages

    # Copy files
    cp /tmp/ergonomic-frontend/index.html .

    # Commit and push
    git add .
    git commit -m "Deploy frontend"
    git push origin gh-pages

    # Switch back
    git checkout serverless-migration

    echo "✓ Frontend deployed to GitHub Pages"
    echo "URL: https://${GITHUB_USERNAME}.github.io/ergonomic-analyzer"

    cd ..
}

# Main deployment flow
echo ""
echo "Select deployment target:"
echo "1. Full deployment (RunPod + Cloudflare + Frontend)"
echo "2. RunPod only"
echo "3. Cloudflare Worker only"
echo "4. Frontend only"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        deploy_runpod
        read -p "Enter RunPod Endpoint ID: " RUNPOD_ENDPOINT_ID
        echo "RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}" >> .env
        deploy_cloudflare
        deploy_frontend
        ;;
    2)
        deploy_runpod
        ;;
    3)
        deploy_cloudflare
        ;;
    4)
        deploy_frontend
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Test the frontend: https://${GITHUB_USERNAME}.github.io/ergonomic-analyzer"
echo "2. Monitor logs: wrangler tail"
echo "3. Check RunPod: https://runpod.io/console/serverless"
echo ""
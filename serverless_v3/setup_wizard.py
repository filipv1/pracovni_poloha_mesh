#!/usr/bin/env python3
"""
V3 Setup Wizard - Automatická konfigurace RunPod + CloudFlare R2
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import webbrowser

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    """Add color to text for terminal output"""
    if sys.platform == 'win32':
        # Windows doesn't support ANSI colors by default
        return text
    return f"{color}{text}{Colors.ENDC}"

def print_header(text):
    """Print colored header"""
    print("\n" + "=" * 60)
    print(colored(text, Colors.HEADER + Colors.BOLD))
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(colored(f"[SUCCESS] {text}", Colors.GREEN))

def print_warning(text):
    """Print warning message"""
    print(colored(f"[WARNING] {text}", Colors.WARNING))

def print_error(text):
    """Print error message"""
    print(colored(f"[ERROR] {text}", Colors.FAIL))

def print_step(number, text):
    """Print numbered step"""
    print(f"\n{colored(f'Step {number}:', Colors.BLUE + Colors.BOLD)} {text}")

def ask_yes_no(question):
    """Ask yes/no question"""
    while True:
        answer = input(f"{question} (y/n): ").lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False

def create_env_file():
    """Create .env file with user input"""
    print_header("CLOUDFLARE R2 / AWS S3 CONFIGURATION")

    env_data = {}

    # Storage provider selection
    print("\nSelect storage provider:")
    print("1. CloudFlare R2 (recommended - free egress)")
    print("2. AWS S3")

    choice = input("Choice (1 or 2): ").strip()

    if choice == '1':
        env_data['STORAGE_PROVIDER'] = 'r2'
        print("\n" + colored("CloudFlare R2 Setup", Colors.BLUE))
        print("Go to: https://dash.cloudflare.com/sign-up")
        print("1. Create account if needed")
        print("2. Go to R2 section")
        print("3. Create bucket named: ergonomic-analysis")
        print("4. Get API credentials from R2 > Manage R2 API Tokens")

        input("\nPress Enter when you have the credentials...")

        env_data['R2_ACCOUNT_ID'] = input("Enter R2 Account ID: ").strip()
        env_data['R2_ACCESS_KEY_ID'] = input("Enter R2 Access Key ID: ").strip()
        env_data['R2_SECRET_ACCESS_KEY'] = input("Enter R2 Secret Access Key: ").strip()
        env_data['R2_BUCKET_NAME'] = input("Enter R2 Bucket Name [ergonomic-analysis]: ").strip() or 'ergonomic-analysis'

    else:
        env_data['STORAGE_PROVIDER'] = 's3'
        print("\n" + colored("AWS S3 Setup", Colors.BLUE))
        print("Go to: https://aws.amazon.com/console/")
        print("1. Create S3 bucket")
        print("2. Get IAM credentials with S3 access")

        input("\nPress Enter when you have the credentials...")

        env_data['S3_ACCESS_KEY_ID'] = input("Enter S3 Access Key ID: ").strip()
        env_data['S3_SECRET_ACCESS_KEY'] = input("Enter S3 Secret Access Key: ").strip()
        env_data['S3_BUCKET_NAME'] = input("Enter S3 Bucket Name [ergonomic-analysis]: ").strip() or 'ergonomic-analysis'
        env_data['S3_REGION'] = input("Enter S3 Region [us-east-1]: ").strip() or 'us-east-1'

    print_header("RUNPOD CONFIGURATION")

    print("\nGo to: https://www.runpod.io/console/user/settings")
    print("1. Sign up / Login")
    print("2. Go to Settings > API Keys")
    print("3. Create new API key")

    input("\nPress Enter when you have the API key...")

    env_data['RUNPOD_API_KEY'] = input("Enter RunPod API Key: ").strip()
    env_data['RUNPOD_ENDPOINT_ID'] = 'WILL_BE_SET_LATER'  # Set after deployment

    # Processing settings
    env_data['DEFAULT_QUALITY'] = 'medium'
    env_data['MAX_VIDEO_SIZE'] = '5368709120'
    env_data['JOB_STATUS_TTL'] = '86400'
    env_data['UPLOAD_URL_EXPIRY'] = '3600'
    env_data['DOWNLOAD_URL_EXPIRY'] = '86400'

    # Write .env file
    env_path = Path('.env')
    with open(env_path, 'w') as f:
        for key, value in env_data.items():
            f.write(f"{key}={value}\n")

    print_success(f".env file created at {env_path.absolute()}")

    return env_data

def build_docker_image():
    """Build and push Docker image"""
    print_header("DOCKER IMAGE BUILD")

    # Check if Docker is installed
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
    except:
        print_error("Docker not installed!")
        print("Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/")
        return False

    docker_username = input("\nEnter your Docker Hub username: ").strip()

    if not docker_username:
        print_warning("Skipping Docker build. You'll need to build manually later.")
        return False

    image_name = f"{docker_username}/ergonomic-analysis-v3"

    print(f"\nBuilding Docker image: {image_name}:latest")

    # Build image - ALWAYS build from current directory with correct context
    cmd = f"docker build -f Dockerfile -t {image_name}:latest .."
    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print_error("Docker build failed!")
        return False

    print_success("Docker image built successfully")

    # Login to Docker Hub
    print("\nLogging in to Docker Hub...")
    subprocess.run(['docker', 'login'], check=True)

    # Push image
    print(f"\nPushing image to Docker Hub...")
    cmd = f"docker push {image_name}:latest"
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print_error("Docker push failed!")
        return False

    print_success(f"Image pushed: {image_name}:latest")

    return image_name

def setup_runpod_endpoint(docker_image, env_data):
    """Guide user through RunPod endpoint setup"""
    print_header("RUNPOD ENDPOINT SETUP")

    print("\nOpen RunPod Console:")
    webbrowser.open("https://www.runpod.io/console/serverless")

    print("\nFollow these steps:")
    print("1. Click '+ New Endpoint'")
    print("2. Configure as follows:")
    print(f"   - Container Image: {docker_image}:latest")
    print("   - Select GPU: RTX 4090 24GB")
    print("   - Container Disk: 20 GB")
    print("   - Max Workers: 3")
    print("   - Idle Timeout: 5 seconds")
    print("   - Execution Timeout: 3600 seconds")

    print("\n3. Set Environment Variables:")

    if env_data['STORAGE_PROVIDER'] == 'r2':
        print(f"   STORAGE_PROVIDER=r2")
        print(f"   R2_ACCOUNT_ID={env_data['R2_ACCOUNT_ID']}")
        print(f"   R2_ACCESS_KEY_ID={env_data['R2_ACCESS_KEY_ID']}")
        print(f"   R2_SECRET_ACCESS_KEY={env_data['R2_SECRET_ACCESS_KEY']}")
        print(f"   R2_BUCKET_NAME={env_data['R2_BUCKET_NAME']}")
    else:
        print(f"   STORAGE_PROVIDER=s3")
        print(f"   S3_ACCESS_KEY_ID={env_data['S3_ACCESS_KEY_ID']}")
        print(f"   S3_SECRET_ACCESS_KEY={env_data['S3_SECRET_ACCESS_KEY']}")
        print(f"   S3_BUCKET_NAME={env_data['S3_BUCKET_NAME']}")
        print(f"   S3_REGION={env_data['S3_REGION']}")

    print("\n4. Click 'Deploy'")
    print("5. Copy the Endpoint ID from the dashboard")

    endpoint_id = input("\nEnter your RunPod Endpoint ID: ").strip()

    # Update .env file with endpoint ID
    env_path = Path('.env')
    content = env_path.read_text()
    content = content.replace('RUNPOD_ENDPOINT_ID=WILL_BE_SET_LATER',
                              f'RUNPOD_ENDPOINT_ID={endpoint_id}')
    env_path.write_text(content)

    print_success("RunPod endpoint configured")

    return endpoint_id

def configure_frontend(endpoint_id, api_key):
    """Configure frontend with RunPod credentials"""
    print_header("FRONTEND CONFIGURATION")

    frontend_path = Path('frontend/index.html')

    if not frontend_path.exists():
        print_error("Frontend file not found!")
        return False

    # Read frontend content with UTF-8 encoding
    content = frontend_path.read_text(encoding='utf-8')

    # Replace placeholders
    endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    content = content.replace('YOUR_ENDPOINT_ID', endpoint_id)
    content = content.replace('YOUR_API_KEY', api_key)
    content = content.replace('https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync', endpoint_url)

    # Save configured frontend with UTF-8 encoding
    configured_path = Path('frontend/index-configured.html')
    configured_path.write_text(content, encoding='utf-8')

    print_success(f"Frontend configured: {configured_path.absolute()}")

    return True

def test_setup():
    """Test the setup"""
    print_header("TESTING SETUP")

    print("\nRunning local tests...")
    result = subprocess.run([sys.executable, 'test_local.py'], capture_output=True, text=True)

    if result.returncode == 0:
        print_success("Local tests passed")
    else:
        print_warning("Some tests failed (this is normal for local testing)")
        print(result.stdout)

    return True

def create_deployment_summary():
    """Create deployment summary file"""
    summary = """
# V3 DEPLOYMENT SUMMARY

## Configuration Complete!

Your V3 serverless architecture is configured and ready to use.

## Files Created:
- `.env` - Environment configuration
- `frontend/index-configured.html` - Configured frontend

## Next Steps:

1. **Test locally:**
   Open `frontend/index-configured.html` in your browser

2. **Deploy frontend to hosting:**
   - GitHub Pages (free): Push to GitHub, enable Pages
   - Netlify (free): Drag & drop folder at netlify.com
   - Vercel (free): Deploy with Vercel CLI

3. **Monitor:**
   - RunPod Dashboard: Check worker logs and GPU usage
   - CloudFlare R2: Monitor storage usage

## Costs:
- RunPod: ~$0.078 per 10-minute video
- CloudFlare R2: $0.015/GB storage, FREE egress
- Frontend hosting: FREE

## Support:
- Check RunPod logs for processing errors
- Verify R2/S3 bucket permissions
- See DEPLOYMENT_README.md for detailed troubleshooting

## Testing:
1. Upload small test video first (<100MB)
2. Check processing completes
3. Download results
4. Then try larger videos

Good luck! 🚀
"""

    summary_path = Path('DEPLOYMENT_SUMMARY.txt')
    summary_path.write_text(summary)

    print_success(f"Deployment summary saved: {summary_path.absolute()}")

    return summary_path

def main():
    """Main setup wizard"""
    print_header("V3 SERVERLESS SETUP WIZARD")
    print("This wizard will guide you through setting up the V3 architecture")
    print("for processing videos with RunPod and CloudFlare R2/AWS S3")

    # Ensure we're in serverless_v3 directory
    if not Path('Dockerfile').exists():
        print("[ERROR] Please run this script from serverless_v3 directory!")
        print("Current directory:", os.getcwd())
        sys.exit(1)

    steps = [
        (1, "Configure Storage (R2/S3)", create_env_file),
        (2, "Build Docker Image", build_docker_image),
        (3, "Setup RunPod Endpoint", None),
        (4, "Configure Frontend", None),
        (5, "Test Setup", test_setup),
        (6, "Create Summary", create_deployment_summary)
    ]

    env_data = None
    docker_image = None
    endpoint_id = None

    for step_num, step_name, step_func in steps:
        print_step(step_num, step_name)

        if step_num == 1:
            env_data = create_env_file()
            if not env_data:
                print_error("Setup failed at step 1")
                return

        elif step_num == 2:
            docker_image = build_docker_image()
            if not docker_image:
                print_warning("Docker image not built. Build manually later.")
                docker_image = input("Enter existing Docker image name [vaclavik/ergonomic-analysis-v3]: ").strip()
                if not docker_image:
                    docker_image = "vaclavik/ergonomic-analysis-v3"

        elif step_num == 3:
            endpoint_id = setup_runpod_endpoint(docker_image, env_data)

        elif step_num == 4:
            success = configure_frontend(endpoint_id, env_data['RUNPOD_API_KEY'])
            if not success:
                print_error("Frontend configuration failed")

        elif step_func:
            step_func()

    print_header("SETUP COMPLETE!")
    print("\nYour V3 serverless architecture is ready!")
    print("\nOpen frontend/index-configured.html to start processing videos.")
    print("\nFor detailed information, see DEPLOYMENT_SUMMARY.txt")

    # Open frontend in browser
    if ask_yes_no("\nOpen frontend in browser now?"):
        frontend_path = Path('frontend/index-configured.html').absolute()
        webbrowser.open(f"file://{frontend_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
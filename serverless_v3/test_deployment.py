#!/usr/bin/env python3
"""
V3 Deployment Test - Ověření funkčnosti po nasazení
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# Load environment variables
def load_env():
    """Load .env file"""
    env_path = Path('.env')
    if not env_path.exists():
        print("[ERROR] .env file not found! Run setup_wizard.py first.")
        sys.exit(1)

    env_vars = {}
    with open(env_path) as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                env_vars[key] = value
                os.environ[key] = value

    return env_vars

def test_r2_connection(env_vars):
    """Test CloudFlare R2 / S3 connection"""
    print("\n[TEST] Testing storage connection...")

    try:
        import boto3
        from botocore.exceptions import ClientError

        if env_vars['STORAGE_PROVIDER'] == 'r2':
            client = boto3.client(
                's3',
                endpoint_url=f"https://{env_vars['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                aws_access_key_id=env_vars['R2_ACCESS_KEY_ID'],
                aws_secret_access_key=env_vars['R2_SECRET_ACCESS_KEY'],
                region_name='auto'
            )
            bucket_name = env_vars['R2_BUCKET_NAME']
        else:
            client = boto3.client(
                's3',
                aws_access_key_id=env_vars['S3_ACCESS_KEY_ID'],
                aws_secret_access_key=env_vars['S3_SECRET_ACCESS_KEY'],
                region_name=env_vars['S3_REGION']
            )
            bucket_name = env_vars['S3_BUCKET_NAME']

        # Try to list bucket contents
        response = client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print(f"[OK] Connected to {env_vars['STORAGE_PROVIDER'].upper()} bucket: {bucket_name}")

        # Test presigned URL generation
        test_key = f"test/test_{datetime.now().timestamp()}.txt"
        url = client.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket_name, 'Key': test_key},
            ExpiresIn=3600
        )
        print(f"[OK] Generated presigned URL successfully")

        return True

    except ImportError:
        print("[ERROR] boto3 not installed! Run: pip install boto3")
        return False
    except ClientError as e:
        print(f"[ERROR] Storage connection failed: {e}")
        print("\nCheck:")
        print("1. Credentials are correct")
        print("2. Bucket exists")
        print("3. Permissions are set correctly")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_runpod_connection(env_vars):
    """Test RunPod endpoint connection"""
    print("\n[TEST] Testing RunPod connection...")

    endpoint_id = env_vars.get('RUNPOD_ENDPOINT_ID')
    api_key = env_vars.get('RUNPOD_API_KEY')

    if not endpoint_id or endpoint_id == 'WILL_BE_SET_LATER':
        print("[WARNING] RunPod endpoint ID not configured yet")
        print("Complete RunPod setup first, then update .env file")
        return False

    if not api_key:
        print("[ERROR] RunPod API key not found in .env")
        return False

    # Test API connection
    url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            print(f"[OK] RunPod endpoint is healthy: {endpoint_id}")
            return True
        elif response.status_code == 401:
            print("[ERROR] Invalid RunPod API key")
            return False
        elif response.status_code == 404:
            print("[ERROR] RunPod endpoint not found. Check endpoint ID.")
            return False
        else:
            print(f"[WARNING] RunPod returned status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("[ERROR] RunPod connection timeout")
        return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to RunPod API")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_frontend_config():
    """Test if frontend is configured"""
    print("\n[TEST] Checking frontend configuration...")

    frontend_path = Path('frontend/index.html')
    configured_path = Path('frontend/index-configured.html')

    if not frontend_path.exists():
        print("[ERROR] Frontend file not found!")
        return False

    content = frontend_path.read_text()

    if 'YOUR_ENDPOINT_ID' in content or 'YOUR_API_KEY' in content:
        print("[WARNING] Frontend not configured yet")
        print("Run setup_wizard.py or manually edit frontend/index.html")

        if configured_path.exists():
            print(f"[INFO] Found configured version at: {configured_path}")
            return True

        return False

    print("[OK] Frontend is configured")
    return True

def test_docker_image():
    """Check if Docker image exists"""
    print("\n[TEST] Checking Docker image...")

    try:
        import subprocess
        result = subprocess.run(
            ['docker', 'images', 'ergonomic-analysis-v3'],
            capture_output=True,
            text=True
        )

        if 'ergonomic-analysis-v3' in result.stdout:
            print("[OK] Docker image found locally")
        else:
            print("[INFO] Docker image not found locally (will be pulled by RunPod)")

        return True

    except FileNotFoundError:
        print("[WARNING] Docker not installed locally")
        print("This is OK if image is already on Docker Hub")
        return True
    except Exception as e:
        print(f"[WARNING] Cannot check Docker: {e}")
        return True

def test_full_pipeline(env_vars):
    """Test complete pipeline with a small test"""
    print("\n[TEST] Testing full pipeline...")

    if env_vars.get('RUNPOD_ENDPOINT_ID') == 'WILL_BE_SET_LATER':
        print("[SKIP] RunPod not configured yet")
        return False

    endpoint_id = env_vars['RUNPOD_ENDPOINT_ID']
    api_key = env_vars['RUNPOD_API_KEY']

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Test getting upload URL
    print("\n1. Testing upload URL generation...")
    payload = {
        "input": {
            "action": "generate_upload_url",
            "filename": "test_video.mp4"
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"[ERROR] RunPod returned status {response.status_code}")
            print(response.text)
            return False

        data = response.json()

        if data.get('status') == 'COMPLETED':
            output = data.get('output', {})
            if output.get('status') == 'success':
                print("[OK] Upload URL generated successfully")
                print(f"    Video key: {output.get('video_key')}")
            else:
                print(f"[ERROR] {output.get('error')}")
                return False
        else:
            print(f"[ERROR] RunPod job failed: {data}")
            return False

    except Exception as e:
        print(f"[ERROR] Pipeline test failed: {e}")
        return False

    print("\n[OK] Pipeline test successful!")
    return True

def create_test_report(results):
    """Create test report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
V3 DEPLOYMENT TEST REPORT
Generated: {timestamp}

TEST RESULTS:
-------------
Storage Connection:  {'PASS' if results['storage'] else 'FAIL'}
RunPod Connection:   {'PASS' if results['runpod'] else 'FAIL'}
Frontend Config:     {'PASS' if results['frontend'] else 'FAIL'}
Docker Image:        {'PASS' if results['docker'] else 'FAIL'}
Full Pipeline:       {'PASS' if results['pipeline'] else 'FAIL'}

OVERALL STATUS: {'READY FOR PRODUCTION' if all(results.values()) else 'NEEDS ATTENTION'}
"""

    # Save report
    report_path = Path('test_report.txt')
    report_path.write_text(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    print(f"\nReport saved: {report_path.absolute()}")

    return all(results.values())

def main():
    """Main test runner"""
    print("=" * 60)
    print("V3 DEPLOYMENT TESTER")
    print("=" * 60)

    # Load environment
    env_vars = load_env()

    # Run tests
    results = {
        'storage': test_r2_connection(env_vars),
        'runpod': test_runpod_connection(env_vars),
        'frontend': test_frontend_config(),
        'docker': test_docker_image(),
        'pipeline': False
    }

    # Only test full pipeline if other tests pass
    if results['storage'] and results['runpod']:
        results['pipeline'] = test_full_pipeline(env_vars)

    # Create report
    success = create_test_report(results)

    if success:
        print("\n[SUCCESS] All tests passed! Your deployment is ready.")
        print("\nOpen frontend/index.html (or index-configured.html) to start processing!")
    else:
        print("\n[WARNING] Some tests failed. Check the report above.")
        print("\nCommon fixes:")
        print("1. Run setup_wizard.py for automatic configuration")
        print("2. Check credentials in .env file")
        print("3. Ensure RunPod endpoint is deployed")
        print("4. Verify CloudFlare R2 / S3 bucket exists")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
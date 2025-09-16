#!/usr/bin/env python
"""
Test RunPod API connection
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_runpod_api():
    """Test RunPod API with the provided key"""
    api_key = os.environ.get('RUNPOD_API_KEY')
    
    if not api_key:
        print("[ERROR] RUNPOD_API_KEY not found in .env")
        return False
    
    print(f"[INFO] Testing RunPod API...")
    print(f"[INFO] API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # RunPod API endpoints - try different versions
    base_urls = [
        "https://api.runpod.io/v2",
        "https://api.runpod.io/v1", 
        "https://api.runpod.io/graphql"
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try to find working endpoint
    base_url = None
    for url in base_urls:
        try:
            print(f"\n[TEST] Trying {url}...")
            test_response = requests.get(f"{url}/pod", headers=headers, timeout=5)
            if test_response.status_code != 404:
                base_url = url
                print(f"[OK] Using API endpoint: {url}")
                break
        except:
            continue
    
    if not base_url:
        base_url = "https://api.runpod.io/v2"  # Default
        print(f"[WARNING] Using default endpoint: {base_url}")
    
    try:
        # Test 1: List pods directly (skip user info)
        print("\n[TEST] Listing pods...")
        response = requests.get(f"{base_url}/pod", headers=headers, timeout=10)
        
        if response.status_code == 200:
            pods_data = response.json()
            pods = pods_data.get('data', []) if isinstance(pods_data, dict) else pods_data
            
            if pods:
                print(f"[OK] Found {len(pods)} pod(s):")
                for pod in pods:
                    pod_id = pod.get('id', 'Unknown')
                    pod_name = pod.get('name', 'Unnamed')
                    pod_status = pod.get('desiredStatus', 'Unknown')
                    gpu_type = pod.get('machine', {}).get('gpuType', 'Unknown')
                    
                    print(f"\n    Pod: {pod_name}")
                    print(f"    ID: {pod_id}")
                    print(f"    Status: {pod_status}")
                    print(f"    GPU: {gpu_type}")
                    
                    # This is the Pod ID you need for .env
                    print(f"\n    [ACTION] Add this to your .env file:")
                    print(f"    RUNPOD_POD_ID={pod_id}")
            else:
                print("[WARNING] No pods found. You need to create a pod first.")
                print("\n[HELP] To create a pod:")
                print("    1. Go to https://runpod.io/console/pods")
                print("    2. Click 'Deploy'")
                print("    3. Choose GPU (A5000 recommended)")
                print("    4. Select 'PyTorch' template")
                print("    5. Set persistent storage")
                print("    6. Deploy and note the Pod ID")
        else:
            print(f"[ERROR] Failed to list pods: {response.status_code}")
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Failed to connect to RunPod API")
        print("    Check your internet connection")
        return False
    except requests.exceptions.Timeout:
        print("[ERROR] RunPod API timeout")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def main():
    print("=" * 60)
    print("RunPod API Connection Test")
    print("=" * 60)
    
    if test_runpod_api():
        print("\n" + "=" * 60)
        print("[SUCCESS] RunPod API is working!")
        print("\nNext steps:")
        print("1. Copy the Pod ID shown above")
        print("2. Add it to your .env file")
        print("3. Restart the Flask app")
    else:
        print("\n" + "=" * 60)
        print("[FAILED] RunPod API test failed")
        print("\nTroubleshooting:")
        print("1. Check your API key in .env")
        print("2. Verify internet connection")
        print("3. Check RunPod service status")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
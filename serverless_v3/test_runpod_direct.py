#!/usr/bin/env python3
"""
Direct RunPod API test - test if everything works
"""

import requests
import json
import time

# Your RunPod credentials
ENDPOINT_ID = "d1mtcfjymab45g"
API_KEY = "YOUR_RUNPOD_API_KEY_HERE"
ENDPOINT_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

def test_generate_upload_url():
    """Test generating upload URL"""
    print("\n=== TEST 1: Generate Upload URL ===")

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        "input": {
            "action": "generate_upload_url",
            "filename": "test_video.mp4"
        }
    }

    print(f"Calling: {ENDPOINT_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'COMPLETED':
            output = data.get('output', {})
            if output.get('status') == 'success':
                print(f"\n✅ SUCCESS! Upload URL generated")
                print(f"Video key: {output.get('video_key')}")
                print(f"Upload URL: {output.get('upload_url')[:100]}...")
                return output.get('upload_url'), output.get('video_key')
            else:
                print(f"\n❌ FAILED: {output.get('error')}")
        else:
            print(f"\n❌ Job failed: {data}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")

    return None, None

def test_start_processing(video_key):
    """Test starting processing"""
    print("\n=== TEST 2: Start Processing ===")

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        "input": {
            "action": "start_processing",
            "video_key": video_key,
            "quality": "medium"
        }
    }

    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'COMPLETED':
            output = data.get('output', {})
            if output.get('status') == 'success':
                print(f"\n✅ SUCCESS! Processing started")
                print(f"Job ID: {output.get('job_id')}")
                return output.get('job_id')
            else:
                print(f"\n❌ FAILED: {output.get('error')}")
        else:
            print(f"\n❌ Job failed: {data}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")

    return None

def test_get_status(job_id):
    """Test getting job status"""
    print("\n=== TEST 3: Get Job Status ===")

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        "input": {
            "action": "get_status",
            "job_id": job_id
        }
    }

    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'COMPLETED':
            output = data.get('output', {})
            if output.get('status') == 'success':
                print(f"\n✅ SUCCESS! Got job status")
                job_status = output.get('job_status', {})
                print(f"Job Status: {job_status.get('status')}")
                print(f"Progress: {job_status.get('progress')}%")
                return True
            else:
                print(f"\n❌ FAILED: {output.get('error')}")
        else:
            print(f"\n❌ Job failed: {data}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")

    return False

def main():
    print("=" * 60)
    print("RUNPOD V3 DIRECT API TEST")
    print("=" * 60)

    # Test 1: Generate upload URL
    upload_url, video_key = test_generate_upload_url()

    if not upload_url:
        print("\n❌ Generate upload URL failed!")
        return

    # Test 2: Start processing (with fake video key)
    job_id = test_start_processing(video_key or "uploads/test.mp4")

    if not job_id:
        print("\n❌ Start processing failed!")
        print("This might be normal if video doesn't exist in R2")

    # Test 3: Get status (with fake job ID if needed)
    if job_id:
        test_get_status(job_id)
    else:
        print("\nTesting with fake job ID...")
        test_get_status("test-job-123")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
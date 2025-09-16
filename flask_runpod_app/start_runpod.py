#!/usr/bin/env python
"""
Start RunPod pod
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def start_pod():
    """Start the RunPod pod"""
    api_key = os.environ.get('RUNPOD_API_KEY')
    pod_id = os.environ.get('RUNPOD_POD_ID')
    
    if not api_key or not pod_id:
        print("[ERROR] RUNPOD_API_KEY or RUNPOD_POD_ID not found in .env")
        return False
    
    print(f"[INFO] Starting pod: {pod_id}")
    
    # GraphQL mutation to start pod
    url = "https://api.runpod.io/graphql"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{api_key}"
    }
    
    # GraphQL mutation to resume pod
    mutation = """
    mutation {
        podResume(input: { podId: "%s" }) {
            id
            desiredStatus
            machineId
        }
    }
    """ % pod_id
    
    try:
        print("[INFO] Sending start command...")
        response = requests.post(
            url,
            json={"query": mutation},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "errors" in data:
                print(f"[ERROR] Failed to start pod: {data['errors']}")
                return False
            
            if "data" in data and data["data"]:
                pod_data = data["data"].get("podResume", {})
                if pod_data:
                    print(f"[OK] Pod starting!")
                    print(f"    Pod ID: {pod_data.get('id')}")
                    print(f"    Status: {pod_data.get('desiredStatus')}")
                    print(f"    Machine: {pod_data.get('machineId')}")
                    print("\n[INFO] Pod will take 1-2 minutes to be ready")
                    print("[INFO] Check status at: https://runpod.io/console/pods")
                    return True
                else:
                    print("[ERROR] No pod data returned")
            else:
                print(f"[ERROR] Invalid response: {json.dumps(data, indent=2)}")
        else:
            print(f"[ERROR] Failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"[ERROR] Failed to start pod: {e}")
        
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("RunPod Pod Starter")
    print("=" * 60)
    
    if start_pod():
        print("\n[SUCCESS] Pod is starting!")
        print("\nNext steps:")
        print("1. Wait 1-2 minutes for pod to be ready")
        print("2. Restart the Flask app: python app.py")
        print("3. The app will now use GPU processing!")
    else:
        print("\n[FAILED] Could not start pod")
        print("\nTroubleshooting:")
        print("1. Check your RunPod account has credits")
        print("2. Check pod status at: https://runpod.io/console/pods")
        print("3. You may need to manually start the pod from the console")
    
    print("=" * 60)
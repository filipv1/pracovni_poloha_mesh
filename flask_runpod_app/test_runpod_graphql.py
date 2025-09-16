#!/usr/bin/env python
"""
Test RunPod API using GraphQL endpoint
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_runpod_graphql():
    """Test RunPod GraphQL API"""
    api_key = os.environ.get('RUNPOD_API_KEY')
    
    if not api_key:
        print("[ERROR] RUNPOD_API_KEY not found in .env")
        return False
    
    print(f"[INFO] Testing RunPod GraphQL API...")
    print(f"[INFO] API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # RunPod GraphQL endpoint
    url = "https://api.runpod.io/graphql"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{api_key}"  # Try without Bearer prefix
    }
    
    # GraphQL query to get user info and pods
    query = """
    query {
        myself {
            id
            email
            pods {
                id
                name
                desiredStatus
                machineId
            }
        }
    }
    """
    
    try:
        print("\n[TEST] Sending GraphQL query...")
        response = requests.post(
            url,
            json={"query": query},
            headers=headers,
            timeout=10
        )
        
        print(f"[INFO] Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if "errors" in data:
                print(f"[ERROR] GraphQL errors: {data['errors']}")
                
                # Try with Bearer prefix
                headers["Authorization"] = f"Bearer {api_key}"
                print("\n[TEST] Trying with Bearer prefix...")
                response = requests.post(
                    url,
                    json={"query": query},
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
            if "data" in data and data["data"]:
                user_data = data["data"].get("myself", {})
                
                if user_data:
                    print(f"\n[OK] API Key is valid!")
                    print(f"    User ID: {user_data.get('id')}")
                    print(f"    Email: {user_data.get('email')}")
                    
                    # Check for regular pods
                    pods = user_data.get("pods", [])
                    if pods:
                        print(f"\n[OK] Found {len(pods)} pod(s):")
                        for pod in pods:
                            print(f"\n    Pod Name: {pod.get('name')}")
                            print(f"    Pod ID: {pod.get('id')}")
                            print(f"    Status: {pod.get('desiredStatus')}")
                            
                            # This is the Pod ID you need
                            print(f"\n    [ACTION] Add this to your .env file:")
                            print(f"    RUNPOD_POD_ID={pod.get('id')}")
                    else:
                        print("\n[WARNING] No regular pods found.")
                    
                    # Check for serverless endpoints
                    serverless = user_data.get("serverlessEndpoints", [])
                    if serverless:
                        print(f"\n[OK] Found {len(serverless)} serverless endpoint(s):")
                        for endpoint in serverless:
                            print(f"\n    Endpoint Name: {endpoint.get('name')}")
                            print(f"    Endpoint ID: {endpoint.get('id')}")
                            print(f"    Status: {endpoint.get('status')}")
                    else:
                        print("\n[WARNING] No serverless endpoints found.")
                    
                    if not pods and not serverless:
                        print("\n[HELP] To create resources:")
                        print("    For GPU Pod:")
                        print("    1. Go to https://runpod.io/console/pods")
                        print("    2. Click 'Deploy'")
                        print("    3. Choose GPU type")
                        print("    4. Deploy and note the Pod ID")
                        print("\n    For Serverless:")
                        print("    1. Go to https://runpod.io/console/serverless")
                        print("    2. Create new endpoint")
                        print("    3. Deploy your container")
                else:
                    print("[ERROR] No user data returned")
            else:
                print(f"[ERROR] Invalid response: {json.dumps(data, indent=2)}")
                
        elif response.status_code == 401:
            print("[ERROR] Authentication failed - API key may be invalid")
        elif response.status_code == 403:
            print("[ERROR] Forbidden - API key may lack permissions")
        else:
            print(f"[ERROR] Unexpected status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Failed to connect to RunPod API")
        return False
    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RunPod GraphQL API Test")
    print("=" * 60)
    
    test_runpod_graphql()
    
    print("\n" + "=" * 60)
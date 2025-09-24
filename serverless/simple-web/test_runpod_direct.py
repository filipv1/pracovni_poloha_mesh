"""
Direct RunPod API test to diagnose issues
"""

import requests
import json

# RunPod configuration
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

print("=" * 60)
print("RUNPOD DIAGNOSTIC TEST")
print("=" * 60)

# Test health endpoint
print("\n1. Testing health endpoint...")
try:
    response = requests.get(
        f'{RUNPOD_BASE_URL}/health',
        headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")

    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Raw Response: {response.text}")

    if response.status_code == 200:
        print("\n✅ Endpoint is healthy")
        if 'workers' in data:
            print(f"Workers ready: {data['workers'].get('ready', 0)}")
            print(f"Workers running: {data['workers'].get('running', 0)}")
    elif response.status_code == 429:
        print("\n⚠️  RATE LIMITED - Too many requests")
        print("Wait a few minutes before trying again")
    elif response.status_code == 401:
        print("\n❌ AUTHENTICATION ERROR")
        print("Check your API key")
    elif response.status_code == 503:
        print("\n⚠️  SERVICE UNAVAILABLE")
        print("Endpoint is throttled or workers are not ready")
    else:
        print(f"\n❌ ERROR: {response.status_code}")

except Exception as e:
    print(f"\n❌ Request failed: {e}")

# Test if endpoint exists
print("\n2. Testing endpoint info...")
try:
    # Try to get endpoint info (different API)
    response = requests.get(
        f'https://api.runpod.ai/v2/{ENDPOINT_ID}',
        headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        print("✅ Endpoint exists and is accessible")
    else:
        print(f"Response: {response.text}")

except Exception as e:
    print(f"Request failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

print("""
If you see 'throttled' status:
1. Wait 5-10 minutes for RunPod to recover
2. Check RunPod console for any errors
3. Workers might be starting up after Docker pull

If you see 503 Service Unavailable:
- Endpoint is temporarily down
- Workers are restarting
- Docker image pull might have failed

If you see 429 Rate Limited:
- Too many requests in short time
- Wait a few minutes

Solutions:
1. Go to https://www.runpod.io/console/serverless
2. Click on your endpoint (dfcn3rqntfybuk)
3. Check the 'Logs' tab for errors
4. Try 'Refresh Workers' button
5. If needed, stop and start the endpoint
""")
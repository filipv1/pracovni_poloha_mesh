"""
Debug handler to test what's causing crashes
"""

print("=== HANDLER STARTING ===")

try:
    print("1. Importing runpod...")
    import runpod
    print("   OK: runpod imported")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    print("2. Importing requests...")
    import requests
    print("   OK: requests imported")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    print("3. Importing base64...")
    import base64
    print("   OK: base64 imported")
except Exception as e:
    print(f"   ERROR: {e}")

def handler(job):
    """Ultra minimal handler"""
    print("=== HANDLER CALLED ===")
    return {
        "status": "success",
        "message": "Debug handler working",
        "test": "minimal"
    }

if __name__ == "__main__":
    print("4. Starting RunPod serverless...")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"ERROR starting serverless: {e}")
        import traceback
        traceback.print_exc()
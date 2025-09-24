"""
Minimal RunPod handler for testing
Tests basic functionality without complex processing
"""

import runpod
import base64
import json
import traceback
from pathlib import Path

def handler(job):
    """
    Minimal handler for testing RunPod integration
    """
    print("=== MINIMAL HANDLER STARTED ===")

    try:
        job_input = job["input"]
        print(f"Received input keys: {list(job_input.keys())}")

        # Test 1: Basic response without progress
        print("Creating basic response...")

        # Create fake results for testing
        fake_xlsx = base64.b64encode(b"fake excel data").decode('utf-8')
        fake_pkl = base64.b64encode(b"fake pkl data").decode('utf-8')

        result = {
            "status": "success",
            "message": "Minimal handler test successful",
            "xlsx_base64": fake_xlsx,
            "pkl_base64": fake_pkl,
            "statistics": {
                "test": "This is minimal handler",
                "frames_processed": 10
            }
        }

        print("=== MINIMAL HANDLER COMPLETED ===")
        return result

    except Exception as e:
        error_msg = f"Minimal handler error: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print("Starting RunPod serverless...")
    runpod.serverless.start({"handler": handler})
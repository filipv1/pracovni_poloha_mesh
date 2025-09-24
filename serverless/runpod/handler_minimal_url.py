"""
Minimal RunPod handler WITH URL support for large files
Tests both base64 and URL input
"""

import runpod
import base64
import json
import traceback
import requests
from pathlib import Path

def download_video(url):
    """Download video from URL"""
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        video_data = b""
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                video_data += chunk

        print(f"Downloaded {len(video_data)/1024/1024:.2f}MB")
        return video_data

    except Exception as e:
        print(f"Download error: {e}")
        return None

def handler(job):
    """
    Minimal handler with URL support for testing
    """
    print("=== MINIMAL HANDLER WITH URL SUPPORT ===")

    try:
        job_input = job["input"]
        print(f"Received input keys: {list(job_input.keys())}")

        # Check input type
        if "video_url" in job_input:
            print("Processing URL input...")
            video_data = download_video(job_input["video_url"])
            if video_data:
                print(f"Video downloaded: {len(video_data)/1024/1024:.2f}MB")
            else:
                return {
                    "status": "error",
                    "error": "Failed to download video from URL"
                }

        elif "video_base64" in job_input:
            print("Processing base64 input...")
            video_data = base64.b64decode(job_input["video_base64"])
            print(f"Video decoded: {len(video_data)/1024/1024:.2f}MB")

        else:
            print("No video input provided")
            video_data = b"test"

        # Create fake results
        fake_xlsx = base64.b64encode(b"fake excel data from URL handler").decode('utf-8')
        fake_pkl = base64.b64encode(b"fake pkl data from URL handler").decode('utf-8')

        result = {
            "status": "success",
            "message": "URL handler test successful",
            "xlsx_base64": fake_xlsx,
            "pkl_base64": fake_pkl,
            "statistics": {
                "handler": "minimal_with_url",
                "input_type": "url" if "video_url" in job_input else "base64",
                "frames_processed": 10
            }
        }

        print("=== HANDLER COMPLETED ===")
        return result

    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print("Starting RunPod serverless with URL support...")
    runpod.serverless.start({"handler": handler})
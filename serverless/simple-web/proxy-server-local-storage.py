"""
Flask proxy server with LOCAL temporary storage for large files
No external dependencies - stores files locally and serves them
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import base64
import os
import time
import uuid
from pathlib import Path
import threading

app = Flask(__name__)
CORS(app)

# RunPod configuration
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

# Local storage configuration
TEMP_STORAGE_DIR = Path("temp_videos")
TEMP_STORAGE_DIR.mkdir(exist_ok=True)
FILE_EXPIRY_SECONDS = 3600  # 1 hour
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Track uploaded files
uploaded_files = {}

def cleanup_old_files():
    """Remove expired files from temp storage"""
    while True:
        try:
            current_time = time.time()
            for file_id, info in list(uploaded_files.items()):
                if current_time - info['timestamp'] > FILE_EXPIRY_SECONDS:
                    file_path = info['path']
                    if file_path.exists():
                        file_path.unlink()
                    del uploaded_files[file_id]
                    print(f"Cleaned up expired file: {file_id}")
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(60)  # Check every minute

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

@app.route('/health', methods=['GET'])
def health():
    """Check RunPod endpoint health"""
    try:
        response = requests.get(
            f'{RUNPOD_BASE_URL}/health',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        )

        print(f"RunPod health status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Workers: {data.get('workers', {})}")
            return jsonify(data), 200
        else:
            print(f"Health check failed: {response.text}")
            return jsonify({
                'error': f'RunPod returned {response.status_code}',
                'details': response.text
            }), response.status_code

    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/temp/<file_id>', methods=['GET'])
def serve_temp_file(file_id):
    """Serve temporary file for RunPod to download"""
    if file_id in uploaded_files:
        file_info = uploaded_files[file_id]
        file_path = file_info['path']
        if file_path.exists():
            return send_file(file_path, mimetype='video/mp4')

    return jsonify({'error': 'File not found'}), 404

def store_video_locally(video_data, filename):
    """Store video in local temp storage and return URL"""
    try:
        # Generate unique ID
        file_id = str(uuid.uuid4())

        # Save file
        file_path = TEMP_STORAGE_DIR / f"{file_id}_{filename}"
        file_path.write_bytes(video_data)

        # Track file
        uploaded_files[file_id] = {
            'path': file_path,
            'filename': filename,
            'timestamp': time.time(),
            'size': len(video_data)
        }

        # Generate local URL
        # Use ngrok or public IP in production
        local_url = f"http://localhost:5000/temp/{file_id}"

        print(f"Stored {len(video_data)/(1024*1024):.2f}MB locally as {file_id}")
        print(f"Local URL: {local_url}")

        return local_url

    except Exception as e:
        print(f"Storage error: {e}")
        return None

@app.route('/submit', methods=['POST'])
def submit_job():
    """Submit job to RunPod with smart file handling"""
    try:
        data = request.json
        print(f"Received data with keys: {list(data.keys())}")

        # Extract input data if it's wrapped
        if 'input' in data and isinstance(data['input'], dict):
            actual_data = data['input']
            print(f"Unwrapped input, actual keys: {list(actual_data.keys())}")
        else:
            actual_data = data

        # Check if we have video data
        if 'video_base64' in actual_data:
            video_base64 = actual_data['video_base64']
            video_name = actual_data.get('video_name', 'input.mp4')

            # Decode to check size
            video_data = base64.b64decode(video_base64)
            file_size_mb = len(video_data) / (1024 * 1024)

            print(f"Video size: {file_size_mb:.2f}MB")

            # Strategy based on file size
            if file_size_mb <= 7:
                # Small file - use base64 directly
                print("Using direct base64 upload (file < 7MB)")
                runpod_input = {
                    'video_base64': video_base64,
                    'video_name': video_name,
                    'quality': actual_data.get('quality', 'medium'),
                    'user_email': actual_data.get('user_email', '')
                }

            else:
                # Large file - store locally and provide URL
                print("File too large for base64, storing locally...")

                # Store video locally
                file_url = store_video_locally(video_data, video_name)

                if not file_url:
                    return jsonify({
                        'error': 'Failed to store large file locally'
                    }), 500

                # For now, since RunPod can't access localhost,
                # we need to use a workaround
                print("WARNING: Large files need public URL access")
                print("Options:")
                print("1. Use ngrok to expose local server")
                print("2. Deploy proxy to cloud")
                print("3. Use smaller videos for testing")

                # Still try to send (will fail from RunPod but shows the flow)
                runpod_input = {
                    'video_url': file_url,
                    'video_name': video_name,
                    'quality': actual_data.get('quality', 'medium'),
                    'user_email': actual_data.get('user_email', '')
                }
                print(f"Using URL: {file_url}")
        else:
            # No video data - pass through as is
            runpod_input = actual_data

        # Submit to RunPod
        print("Submitting to RunPod...")
        print(f"Input keys: {list(runpod_input.keys())}")

        response = requests.post(
            f'{RUNPOD_BASE_URL}/run',
            headers={
                'Authorization': f'Bearer {RUNPOD_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'input': runpod_input}
        )

        print(f"RunPod response: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Job submitted: {result.get('id')}")
            return jsonify(result), 200
        else:
            print(f"Submit failed: {response.text}")
            return jsonify({
                'error': f'Submit failed: {response.text}'
            }), response.status_code

    except Exception as e:
        print(f"Submit error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job status from RunPod"""
    try:
        response = requests.get(
            f'{RUNPOD_BASE_URL}/status/{job_id}',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Job {job_id} status: {data.get('status')}")
            return jsonify(data), 200
        else:
            print(f"Status check failed: {response.text}")
            return jsonify({
                'error': f'Status check failed: {response.text}'
            }), response.status_code

    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Ergonomic Analysis Proxy Server")
    print("WITH LOCAL STORAGE (Testing)")
    print("=" * 50)
    print("Configuration:")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print(f"  Temp storage: {TEMP_STORAGE_DIR}")
    print(f"  File expiry: {FILE_EXPIRY_SECONDS}s")
    print("")
    print("Features:")
    print("- Files < 7MB: Direct base64 upload")
    print("- Files > 7MB: Local storage (needs ngrok for RunPod)")
    print("- Max file size: 500MB")
    print("")
    print("⚠️  WARNING: Large files need public URL!")
    print("Run: ngrok http 5000")
    print("Then update the URL in the code")
    print("")
    print("Testing RunPod connection...")

    # Test connection on startup
    try:
        test_response = requests.get(
            f'{RUNPOD_BASE_URL}/health',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
            timeout=5
        )
        if test_response.status_code == 200:
            data = test_response.json()
            print(f"✅ RunPod connected! Workers ready: {data.get('workers', {}).get('ready', 0)}")
        else:
            print(f"⚠️ RunPod returned: {test_response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to RunPod: {e}")

    print("=" * 50)
    print("")
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("")

    app.run(host='localhost', port=5000, debug=True)
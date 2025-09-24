"""
Flask proxy server with URL upload support for large files
FIXED VERSION - combines working config with URL support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# RunPod configuration (hardcoded, not from environment)
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

# File size limits
MAX_BASE64_SIZE = 7 * 1024 * 1024  # 7MB before base64 encoding
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

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

def upload_to_fileio(file_data, filename):
    """Upload file to file.io (free temporary storage)"""
    try:
        print(f"Uploading {len(file_data)/1024/1024:.2f}MB to file.io...")

        response = requests.post(
            'https://file.io',
            files={'file': (filename, file_data)},
            data={'expires': '1d'}  # Expire after 1 day
        )

        if response.status_code == 200:
            result = response.json()
            if result['success']:
                file_url = result['link']
                print(f"Upload successful: {file_url}")
                return file_url
            else:
                print(f"Upload failed: {result}")
                return None
        else:
            print(f"Upload error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Upload exception: {e}")
        return None

def upload_to_transfersh(file_data, filename):
    """Backup: Upload to transfer.sh"""
    try:
        print(f"Uploading {len(file_data)/1024/1024:.2f}MB to transfer.sh...")

        response = requests.put(
            f'https://transfer.sh/{filename}',
            data=file_data,
            headers={'Max-Days': '1'}
        )

        if response.status_code == 200:
            file_url = response.text.strip()
            print(f"Upload successful: {file_url}")
            return file_url
        else:
            print(f"Upload error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Upload exception: {e}")
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
                # Large file - upload to temporary storage
                print("File too large for base64, uploading to temporary storage...")

                # Try file.io first
                file_url = upload_to_fileio(video_data, video_name)

                # Fallback to transfer.sh if file.io fails
                if not file_url:
                    print("file.io failed, trying transfer.sh...")
                    file_url = upload_to_transfersh(video_data, video_name)

                if not file_url:
                    return jsonify({
                        'error': 'Failed to upload large file to temporary storage'
                    }), 500

                # Send URL instead of base64
                runpod_input = {
                    'video_url': file_url,
                    'video_name': video_name,
                    'quality': actual_data.get('quality', 'medium'),
                    'user_email': actual_data.get('user_email', '')
                }
                print(f"Using URL upload: {file_url}")
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
    print("WITH LARGE FILE SUPPORT (FIXED)")
    print("=" * 50)
    print("Configuration:")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print(f"  API URL: {RUNPOD_BASE_URL}")
    print("")
    print("Features:")
    print("- Files < 7MB: Direct base64 upload")
    print("- Files > 7MB: Upload to file.io/transfer.sh")
    print("- Max file size: 100MB")
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
            print(f"Response: {test_response.text}")
    except Exception as e:
        print(f"❌ Cannot connect to RunPod: {e}")

    print("=" * 50)
    print("")
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("")

    app.run(host='localhost', port=5000, debug=True)
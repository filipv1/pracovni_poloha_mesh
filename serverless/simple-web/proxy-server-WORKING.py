"""
Flask proxy server with WORKING file upload for ANY SIZE
Uses tmpfiles.org for reliable temporary storage
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# RunPod configuration
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

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
            return jsonify({
                'error': f'RunPod returned {response.status_code}',
                'details': response.text
            }), response.status_code

    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({'error': str(e)}), 500

def upload_to_tmpfiles(file_data, filename):
    """Upload to tmpfiles.org - ACTUALLY WORKS"""
    try:
        print(f"Uploading {len(file_data)/(1024*1024):.2f}MB to tmpfiles.org...")

        # tmpfiles.org API
        response = requests.post(
            'https://tmpfiles.org/api/v1/upload',
            files={'file': (filename, file_data, 'video/mp4')}
        )

        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                # Convert URL from viewing format to direct download
                # https://tmpfiles.org/12345/video.mp4 -> https://tmpfiles.org/dl/12345/video.mp4
                view_url = result['data']['url']
                download_url = view_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                print(f"Upload successful: {download_url}")
                return download_url

        print(f"Upload failed: {response.text}")
        return None

    except Exception as e:
        print(f"Upload exception: {e}")
        return None

def upload_to_0x0(file_data, filename):
    """Backup: Upload to 0x0.st"""
    try:
        print(f"Uploading {len(file_data)/(1024*1024):.2f}MB to 0x0.st...")

        response = requests.post(
            'https://0x0.st',
            files={'file': (filename, file_data, 'video/mp4')}
        )

        if response.status_code == 200:
            file_url = response.text.strip()
            print(f"Upload successful: {file_url}")
            return file_url

        print(f"Upload failed: {response.status_code}")
        return None

    except Exception as e:
        print(f"Upload exception: {e}")
        return None

def upload_to_litterbox(file_data, filename):
    """Backup: Upload to litterbox.catbox.moe (1 hour expiry)"""
    try:
        print(f"Uploading {len(file_data)/(1024*1024):.2f}MB to litterbox...")

        response = requests.post(
            'https://litterbox.catbox.moe/resources/internals/api.php',
            data={
                'reqtype': 'fileupload',
                'time': '1h'  # 1 hour expiry
            },
            files={'fileToUpload': (filename, file_data, 'video/mp4')}
        )

        if response.status_code == 200:
            file_url = response.text.strip()
            if file_url.startswith('https://'):
                print(f"Upload successful: {file_url}")
                return file_url

        print(f"Upload failed: {response.text}")
        return None

    except Exception as e:
        print(f"Upload exception: {e}")
        return None

@app.route('/submit', methods=['POST'])
def submit_job():
    """Submit job to RunPod with WORKING file handling"""
    try:
        data = request.json
        print(f"Received data with keys: {list(data.keys())}")

        # Extract input data if wrapped
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

                # Try multiple services
                file_url = None

                # Try tmpfiles.org first (most reliable)
                file_url = upload_to_tmpfiles(video_data, video_name)

                # Fallback to 0x0.st
                if not file_url:
                    print("tmpfiles failed, trying 0x0.st...")
                    file_url = upload_to_0x0(video_data, video_name)

                # Fallback to litterbox
                if not file_url:
                    print("0x0.st failed, trying litterbox...")
                    file_url = upload_to_litterbox(video_data, video_name)

                if not file_url:
                    return jsonify({
                        'error': 'All upload services failed. Try again.'
                    }), 500

                # Send URL to RunPod
                runpod_input = {
                    'video_url': file_url,
                    'video_name': video_name,
                    'quality': actual_data.get('quality', 'medium'),
                    'user_email': actual_data.get('user_email', '')
                }
                print(f"Using URL upload: {file_url}")
        else:
            # No video data
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
            return jsonify({
                'error': f'Status check failed: {response.text}'
            }), response.status_code

    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("PRODUCTION PROXY SERVER - ANY FILE SIZE")
    print("=" * 50)
    print("Configuration:")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print("")
    print("Features:")
    print("- Files < 7MB: Direct base64")
    print("- Files > 7MB: Upload to tmpfiles.org/0x0.st/litterbox")
    print("- ANY SIZE WORKS!")
    print("")
    print("Testing RunPod connection...")

    # Test connection
    try:
        test_response = requests.get(
            f'{RUNPOD_BASE_URL}/health',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
            timeout=5
        )
        if test_response.status_code == 200:
            data = test_response.json()
            print(f"[OK] RunPod connected! Workers ready: {data.get('workers', {}).get('ready', 0)}")
        else:
            print(f"[WARNING] RunPod returned: {test_response.status_code}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to RunPod: {e}")

    print("=" * 50)
    print("")
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("")

    app.run(host='localhost', port=5000, debug=True)
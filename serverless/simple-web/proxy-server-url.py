"""
Flask proxy server with URL upload support for large files
Handles files > 10MB by uploading to temporary storage
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
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY', 'YOUR_RUNPOD_API_KEY_HERE')
ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID', 'dfcn3rqntfybuk')
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

# File size limits
MAX_BASE64_SIZE = 7 * 1024 * 1024  # 7MB before base64 encoding (~9.3MB after)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

@app.route('/health', methods=['GET'])
def health():
    """Check RunPod endpoint health"""
    try:
        response = requests.get(
            f'{RUNPOD_BASE_URL}/health',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        )

        # Log the response for debugging
        print(f"RunPod health status: {response.status_code}")

        if response.status_code == 503:
            print("RunPod is throttled or unavailable")
            return jsonify({
                'error': 'RunPod endpoint is throttled. Please wait a few minutes.',
                'status_code': 503,
                'raw_response': response.text
            }), 503

        try:
            data = response.json()
        except:
            data = {'raw': response.text}

        return jsonify(data), response.status_code

    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({'error': str(e)}), 500

def upload_to_fileio(file_data, filename):
    """Upload file to file.io (free temporary storage, expires in 14 days)"""
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
    """Backup: Upload to transfer.sh (alternative service)"""
    try:
        print(f"Uploading {len(file_data)/1024/1024:.2f}MB to transfer.sh...")

        response = requests.put(
            f'https://transfer.sh/{filename}',
            data=file_data,
            headers={'Max-Days': '1'}  # Keep for 1 day
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

        # Check if we have video data
        if 'video_base64' in data:
            video_base64 = data['video_base64']
            video_name = data.get('video_name', 'input.mp4')

            # Decode to check size
            video_data = base64.b64decode(video_base64)
            file_size_mb = len(video_data) / (1024 * 1024)

            print(f"Video size: {file_size_mb:.2f}MB")

            # Strategy based on file size
            if file_size_mb <= 7:
                # Small file - use base64 directly
                print("Using direct base64 upload (file < 7MB)")
                runpod_data = data

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
                runpod_data = {
                    'video_url': file_url,
                    'video_name': video_name,
                    'quality': data.get('quality', 'medium'),
                    'user_email': data.get('user_email', '')
                }
                print(f"Using URL upload: {file_url}")
        else:
            # No video data
            runpod_data = data

        # Submit to RunPod
        print("Submitting to RunPod...")
        response = requests.post(
            f'{RUNPOD_BASE_URL}/run',
            headers={
                'Authorization': f'Bearer {RUNPOD_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'input': runpod_data}
        )

        result = response.json()

        if response.status_code == 200:
            print(f"Job submitted: {result.get('id')}")
        else:
            print(f"Submit error: {result}")

        return jsonify(result), response.status_code

    except Exception as e:
        print(f"Submit exception: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job status from RunPod"""
    try:
        response = requests.get(
            f'{RUNPOD_BASE_URL}/status/{job_id}',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Ergonomic Analysis Proxy Server")
    print("WITH LARGE FILE SUPPORT")
    print("=" * 50)
    print("Features:")
    print("- Files < 7MB: Direct base64 upload")
    print("- Files > 7MB: Upload to file.io/transfer.sh")
    print("- Max file size: 100MB")
    print("")
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("=" * 50)

    app.run(host='localhost', port=5000, debug=True)
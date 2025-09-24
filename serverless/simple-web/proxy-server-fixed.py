"""
Fixed Flask proxy server for RunPod
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os

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
            print(f"Health check failed: {response.text}")
            return jsonify({
                'error': f'RunPod returned {response.status_code}',
                'details': response.text
            }), response.status_code

    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit_job():
    """Submit job to RunPod"""
    try:
        data = request.json
        print(f"Submitting job with keys: {list(data.keys())}")

        # Check file size
        if 'video_base64' in data:
            video_size = len(data['video_base64']) * 3/4 / 1024 / 1024  # Approx MB
            print(f"Video size (base64): ~{video_size:.2f}MB")

            if video_size > 9:  # Leave some margin for headers
                return jsonify({
                    'error': 'File too large for base64. Please use a smaller file or implement URL upload.'
                }), 400

        # Submit to RunPod
        response = requests.post(
            f'{RUNPOD_BASE_URL}/run',
            headers={
                'Authorization': f'Bearer {RUNPOD_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'input': data}
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
    print("RunPod Proxy Server (FIXED)")
    print("=" * 50)
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("=" * 50)
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
    except:
        print("❌ Cannot connect to RunPod")

    print("=" * 50)

    app.run(host='localhost', port=5000, debug=True)
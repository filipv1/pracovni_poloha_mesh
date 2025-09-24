"""
Simple proxy server to bypass CORS for RunPod API calls
Run this locally to enable the web interface
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
RUNPOD_ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}'

@app.route('/')
def index():
    return "Proxy server is running! Open index.html in your browser."

@app.route('/health', methods=['GET'])
def health_check():
    """Check RunPod endpoint health"""
    try:
        response = requests.get(
            f'{RUNPOD_BASE_URL}/health',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit_job():
    """Submit job to RunPod"""
    try:
        data = request.json

        # Forward to RunPod
        response = requests.post(
            f'{RUNPOD_BASE_URL}/run',
            headers={
                'Authorization': f'Bearer {RUNPOD_API_KEY}',
                'Content-Type': 'application/json'
            },
            json=data
        )

        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check job status"""
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
    print("=" * 50)
    print("Server running at: http://localhost:5000")
    print("Open index-with-proxy.html in your browser")
    print("=" * 50)

    app.run(host='localhost', port=5000, debug=True)
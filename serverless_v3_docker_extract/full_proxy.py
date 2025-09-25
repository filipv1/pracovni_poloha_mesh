#!/usr/bin/env python3
"""
Full proxy server - handles both RunPod API and R2 uploads
Solves all CORS issues!
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app, origins='*', methods=['GET', 'POST', 'PUT', 'OPTIONS'], allow_headers='*')

# RunPod configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/d1mtcfjymab45g/runsync"
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY', '')

@app.route('/runpod', methods=['POST', 'OPTIONS'])
def proxy_runpod():
    """Proxy requests to RunPod API"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        print(f"[RunPod] Proxying: {json.dumps(data, indent=2)}")

        headers = {
            'Authorization': f'Bearer {RUNPOD_API_KEY}',
            'Content-Type': 'application/json'
        }

        response = requests.post(RUNPOD_ENDPOINT, json=data, headers=headers, timeout=30)
        result = response.json()

        print(f"[RunPod] Response status: {result.get('status')}")
        if 'output' in result:
            print(f"[RunPod] Response output: {json.dumps(result['output'], indent=2)}")
        else:
            print(f"[RunPod] Full response: {json.dumps(result, indent=2)}")
        return jsonify(result)

    except Exception as e:
        print(f"[RunPod] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST', 'OPTIONS'])
def proxy_upload():
    """Proxy file upload to R2 presigned URL"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Get presigned URL from request
        upload_url = request.args.get('url')
        if not upload_url:
            return jsonify({"error": "Missing upload URL"}), 400

        # Get file data
        file_data = request.data

        print(f"[Upload] Proxying {len(file_data)} bytes to R2")

        # Upload to R2
        response = requests.put(upload_url, data=file_data, headers={'Content-Type': 'video/mp4'})

        print(f"[Upload] R2 response: {response.status_code}")

        if response.status_code == 200:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": f"R2 returned {response.status_code}"}), 500

    except Exception as e:
        print(f"[Upload] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "V3 Full Proxy"})

if __name__ == '__main__':
    print("=" * 60)
    print("V3 FULL PROXY SERVER")
    print("=" * 60)
    print("RunPod proxy: http://localhost:5001/runpod")
    print("Upload proxy: http://localhost:5001/upload?url=<presigned_url>")
    print("=" * 60)
    print("NOTE: Install Flask first: pip install flask flask-cors")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)
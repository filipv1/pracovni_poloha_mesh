#!/usr/bin/env python3
"""
CORS Proxy Server for RunPod API
Solves "Failed to fetch" error when running frontend locally
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# RunPod configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/d1mtcfjymab45g/runsync"
RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY_HERE"

@app.route('/runpod', methods=['POST'])
def proxy_runpod():
    """Proxy requests to RunPod API"""
    try:
        # Get data from frontend
        data = request.get_json()

        # Prepare headers for RunPod
        headers = {
            'Authorization': f'Bearer {RUNPOD_API_KEY}',
            'Content-Type': 'application/json'
        }

        print(f"Proxying request: {json.dumps(data, indent=2)}")

        # Forward to RunPod
        response = requests.post(
            RUNPOD_ENDPOINT,
            json=data,
            headers=headers,
            timeout=30
        )

        # Return response to frontend
        result = response.json()
        print(f"RunPod response: {json.dumps(result, indent=2)[:500]}")

        return jsonify(result)

    except Exception as e:
        print(f"Proxy error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "RunPod Proxy"})

if __name__ == '__main__':
    print("=" * 60)
    print("RUNPOD PROXY SERVER")
    print("=" * 60)
    print("Proxy running at: http://localhost:5000")
    print("Frontend should call: http://localhost:5000/runpod")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
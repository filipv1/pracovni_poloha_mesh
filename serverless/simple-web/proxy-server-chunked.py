"""
Flask proxy server with CHUNKED upload for large files
Splits large videos into multiple small requests
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import json
import time
from pathlib import Path

app = Flask(__name__)
CORS(app)

# RunPod configuration
RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY_HERE'
ENDPOINT_ID = 'dfcn3rqntfybuk'
RUNPOD_BASE_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'

# Chunked upload configuration
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks
MAX_CHUNKS = 100  # Max 500MB total

# Storage for chunks
video_chunks = {}

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

@app.route('/submit', methods=['POST'])
def submit_job():
    """Submit job to RunPod - handles both small and chunked uploads"""
    try:
        data = request.json
        print(f"Received data with keys: {list(data.keys())}")

        # Extract input data if it's wrapped
        if 'input' in data and isinstance(data['input'], dict):
            actual_data = data['input']
            print(f"Unwrapped input, actual keys: {list(actual_data.keys())}")
        else:
            actual_data = data

        # Check if this is a chunked upload
        if 'chunk_id' in actual_data:
            # Handle chunk
            chunk_id = actual_data['chunk_id']
            chunk_index = actual_data['chunk_index']
            total_chunks = actual_data['total_chunks']
            chunk_data = actual_data['chunk_data']

            print(f"Received chunk {chunk_index + 1}/{total_chunks} for {chunk_id}")

            # Store chunk
            if chunk_id not in video_chunks:
                video_chunks[chunk_id] = {
                    'chunks': {},
                    'total': total_chunks,
                    'metadata': actual_data
                }

            video_chunks[chunk_id]['chunks'][chunk_index] = chunk_data

            # Check if all chunks received
            if len(video_chunks[chunk_id]['chunks']) == total_chunks:
                print(f"All chunks received for {chunk_id}, assembling...")

                # Assemble video
                full_video_base64 = ''
                for i in range(total_chunks):
                    full_video_base64 += video_chunks[chunk_id]['chunks'][i]

                # Clean up chunks
                metadata = video_chunks[chunk_id]['metadata']
                del video_chunks[chunk_id]

                # Submit to RunPod
                runpod_input = {
                    'video_base64': full_video_base64,
                    'video_name': metadata.get('video_name', 'input.mp4'),
                    'quality': metadata.get('quality', 'medium'),
                    'user_email': metadata.get('user_email', '')
                }

                print(f"Submitting assembled video to RunPod...")

            else:
                # Waiting for more chunks
                received = len(video_chunks[chunk_id]['chunks'])
                return jsonify({
                    'status': 'chunk_received',
                    'chunk_id': chunk_id,
                    'received': received,
                    'total': total_chunks
                }), 200

        elif 'video_base64' in actual_data:
            # Regular upload (small file)
            video_base64 = actual_data['video_base64']
            video_name = actual_data.get('video_name', 'input.mp4')

            # Check size
            video_data = base64.b64decode(video_base64)
            file_size_mb = len(video_data) / (1024 * 1024)

            print(f"Video size: {file_size_mb:.2f}MB")

            if file_size_mb <= 9:  # Leave some margin
                print("Using direct base64 upload")
                runpod_input = {
                    'video_base64': video_base64,
                    'video_name': video_name,
                    'quality': actual_data.get('quality', 'medium'),
                    'user_email': actual_data.get('user_email', '')
                }
            else:
                return jsonify({
                    'error': 'File too large. Please use chunked upload from frontend.'
                }), 400
        else:
            # No video data
            runpod_input = actual_data

        # Submit to RunPod
        print(f"Submitting to RunPod with keys: {list(runpod_input.keys())}")

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
    print("Ergonomic Analysis Proxy Server")
    print("WITH CHUNKED UPLOAD SUPPORT")
    print("=" * 50)
    print("Configuration:")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print(f"  Chunk size: {CHUNK_SIZE / (1024*1024):.1f}MB")
    print(f"  Max chunks: {MAX_CHUNKS}")
    print("")
    print("Features:")
    print("- Files < 9MB: Direct base64 upload")
    print("- Files > 9MB: Chunked upload (5MB chunks)")
    print("- Max file size: 500MB")
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
    print("⚠️  Frontend needs update for chunked upload!")
    print("")

    app.run(host='localhost', port=5000, debug=True)
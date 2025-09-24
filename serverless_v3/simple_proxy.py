#!/usr/bin/env python3
"""
Simple CORS proxy for RunPod API - fixes "Failed to fetch" error
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import requests

RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/d1mtcfjymab45g/runsync"
RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY_HERE"

class ProxyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Proxy POST requests to RunPod"""
        if self.path == '/runpod':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                # Parse request from frontend
                data = json.loads(post_data)
                print(f"Proxying request: {json.dumps(data, indent=2)}")

                # Forward to RunPod
                headers = {
                    'Authorization': f'Bearer {RUNPOD_API_KEY}',
                    'Content-Type': 'application/json'
                }

                response = requests.post(RUNPOD_ENDPOINT, json=data, headers=headers, timeout=30)
                result = response.json()

                print(f"RunPod response status: {result.get('status')}")

                # Send response back to frontend
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            except Exception as e:
                print(f"Proxy error: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    print("=" * 60)
    print("RUNPOD CORS PROXY")
    print("=" * 60)
    print("Proxy running at: http://localhost:8001/runpod")
    print("Update frontend to use this URL instead of RunPod directly")
    print("=" * 60)

    server = HTTPServer(('localhost', 8001), ProxyHandler)
    server.serve_forever()
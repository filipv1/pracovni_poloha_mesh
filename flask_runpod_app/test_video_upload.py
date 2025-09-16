#!/usr/bin/env python
"""
Test video upload functionality
"""

import requests
import os
import time
import json
from pathlib import Path

def test_upload():
    """Test the upload functionality with a small test video"""
    
    # Create a small test video file (just a dummy file for testing)
    test_video = "test_video.mp4"
    
    # Create a minimal MP4 file (just header for testing)
    # This is a minimal valid MP4 header
    mp4_header = b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00\x69\x73\x6f\x6d\x69\x73\x6f\x32\x6d\x70\x34\x31'
    
    with open(test_video, 'wb') as f:
        f.write(mp4_header)
        # Add some dummy data to make it look like a video
        f.write(b'\x00' * 1024)  # 1KB of data
    
    print(f"[INFO] Created test video: {test_video}")
    
    # Login first
    session = requests.Session()
    
    login_data = {
        'username': 'admin',
        'password': 'admin123'
    }
    
    login_response = session.post(
        'http://localhost:5000/login',
        data=login_data,
        allow_redirects=False
    )
    
    if login_response.status_code == 302:
        print("[OK] Login successful")
    else:
        print(f"[ERROR] Login failed: {login_response.status_code}")
        return
    
    # Upload the video
    with open(test_video, 'rb') as f:
        files = {'video': (test_video, f, 'video/mp4')}
        
        print("[INFO] Uploading video...")
        upload_response = session.post(
            'http://localhost:5000/api/upload',
            files=files
        )
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        print(f"[OK] Upload successful!")
        print(f"     Job ID: {result.get('job_id')}")
        print(f"     Status: {result.get('status')}")
        
        # Check progress
        job_id = result.get('job_id')
        if job_id:
            print(f"\n[INFO] Checking job progress...")
            time.sleep(1)
            
            progress_response = session.get(f'http://localhost:5000/api/jobs/{job_id}')
            if progress_response.status_code == 200:
                job_data = progress_response.json()
                print(f"[OK] Job Status: {job_data.get('status')}")
                print(f"     Progress: {job_data.get('progress', 0)}%")
                
                # Check history
                print(f"\n[INFO] Checking history...")
                history_response = session.get('http://localhost:5000/api/jobs')
                if history_response.status_code == 200:
                    jobs = history_response.json()
                    print(f"[OK] Found {len(jobs)} job(s) in history")
                    for job in jobs[:1]:  # Show first job
                        print(f"     - Job {job.get('id')}: {job.get('status')} ({job.get('progress', 0)}%)")
    else:
        print(f"[ERROR] Upload failed: {upload_response.status_code}")
        print(f"       Response: {upload_response.text}")
    
    # Cleanup
    if os.path.exists(test_video):
        os.remove(test_video)
        print(f"\n[INFO] Cleaned up test video")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    test_upload()
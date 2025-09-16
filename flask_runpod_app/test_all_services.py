#!/usr/bin/env python
"""
Test all configured services
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("Testing All Services")
print("=" * 60)

# Test 1: RunPod
print("\n[1/3] Testing RunPod...")
try:
    import requests
    api_key = os.environ.get('RUNPOD_API_KEY')
    pod_id = os.environ.get('RUNPOD_POD_ID')
    
    if api_key and pod_id:
        print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
        print(f"  Pod ID: {pod_id}")
        
        # Try to get pod status
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"https://api.runpod.io/v2/pod/{pod_id}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            print("  [OK] RunPod configured (API responds)")
        else:
            print(f"  [WARNING] RunPod API returned: {response.status_code}")
            print("  Note: Pod might not exist yet or API key might be incorrect")
    else:
        print("  [ERROR] Missing RUNPOD_API_KEY or RUNPOD_POD_ID")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 2: Cloudflare R2
print("\n[2/3] Testing Cloudflare R2...")
try:
    import boto3
    from botocore.config import Config
    
    account_id = os.environ.get('R2_ACCOUNT_ID')
    access_key = os.environ.get('R2_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    bucket = os.environ.get('R2_BUCKET_NAME')
    
    if all([account_id, access_key, secret_key, bucket]):
        print(f"  Account ID: {account_id[:8]}...")
        print(f"  Access Key: {access_key[:10]}...")
        print(f"  Bucket: {bucket}")
        
        # Try to connect to R2
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4', retries={'max_attempts': 1}),
            region_name='auto'
        )
        
        # Try to list buckets or check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"  [OK] R2 Storage configured - bucket '{bucket}' accessible")
        except Exception as e:
            if 'NoSuchBucket' in str(e):
                print(f"  [INFO] Bucket '{bucket}' doesn't exist, will be created on first use")
            else:
                print(f"  [WARNING] R2 connection issue: {str(e)[:50]}...")
    else:
        print("  [ERROR] Missing R2 configuration")
except ImportError:
    print("  [ERROR] boto3 not installed - run: pip install boto3")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 3: Email (Gmail SMTP)
print("\n[3/3] Testing Email (Gmail)...")
try:
    import smtplib
    import ssl
    
    smtp_server = os.environ.get('SMTP_SERVER')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    username = os.environ.get('SMTP_USERNAME')
    password = os.environ.get('SMTP_PASSWORD')
    
    if all([smtp_server, username, password]):
        print(f"  Server: {smtp_server}:{smtp_port}")
        print(f"  Username: {username}")
        print(f"  Password: {'*' * 12}{password[-4:]}")
        
        # Try to connect to SMTP
        context = ssl.create_default_context()
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(username, password)
                print("  [OK] Email configured - Gmail login successful")
        except smtplib.SMTPAuthenticationError:
            print("  [ERROR] Gmail authentication failed")
            print("  Check: 1) App password is correct")
            print("        2) 2FA is enabled on Gmail account")
        except Exception as e:
            print(f"  [WARNING] SMTP connection issue: {str(e)[:50]}...")
    else:
        print("  [ERROR] Missing email configuration")
except Exception as e:
    print(f"  [ERROR] {e}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("-" * 60)

# Check what's working
services = []

if os.environ.get('RUNPOD_API_KEY') and os.environ.get('RUNPOD_POD_ID'):
    services.append("RunPod (configured)")

if all([os.environ.get('R2_ACCOUNT_ID'), 
        os.environ.get('R2_ACCESS_KEY_ID'), 
        os.environ.get('R2_SECRET_ACCESS_KEY')]):
    services.append("Cloudflare R2 (configured)")

if all([os.environ.get('SMTP_USERNAME'), 
        os.environ.get('SMTP_PASSWORD')]):
    services.append("Email (configured)")

if services:
    print("[OK] Configured services:")
    for service in services:
        print(f"  - {service}")
else:
    print("[WARNING] No services configured")

print("\n[INFO] The app will work in local mode without these services")
print("[INFO] To start the app: python app.py")
print("=" * 60)
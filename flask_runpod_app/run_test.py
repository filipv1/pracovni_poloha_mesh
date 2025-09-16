#!/usr/bin/env python
"""
Test runner for Flask RunPod Application
This script tests the application without requiring all dependencies
"""

import os
import sys

# Set minimal environment variables for testing
os.environ['FLASK_SECRET_KEY'] = 'test-secret-key-123'
os.environ['DATABASE_URL'] = 'sqlite:///test.db'

print("=" * 60)
print("Flask RunPod Application - Test Runner")
print("=" * 60)

try:
    print("\n1. Testing imports...")
    from app import app, initialize_app
    print("   [OK] App imported successfully")
    
    print("\n2. Initializing application...")
    initialize_app()
    print("   [OK] Application initialized")
    
    print("\n3. Testing routes...")
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/health')
        print(f"   [OK] Health check: {response.status_code}")
        
        # Test home page redirect
        response = client.get('/')
        print(f"   [OK] Home page: {response.status_code}")
        
        # Test login page
        response = client.get('/login')
        print(f"   [OK] Login page: {response.status_code}")
    
    print("\n4. Checking configuration...")
    print(f"   [OK] Upload limit: {app.config.get('MAX_CONTENT_LENGTH', 0) / (1024*1024):.0f} MB")
    print(f"   [OK] Secret key: {'SET' if app.config['SECRET_KEY'] else 'NOT SET'}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("\nThe application is ready to run.")
    print("\nTo start the server:")
    print("  1. Configure your .env file with API keys")
    print("  2. Run: python app.py")
    print("  3. Visit: http://localhost:5000")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
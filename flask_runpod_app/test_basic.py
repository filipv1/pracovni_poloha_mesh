#!/usr/bin/env python
"""
Basic functionality test for Flask RunPod application
"""

import os
import sys
import tempfile

def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    
    try:
        from app import app
        print("✓ Flask app imports")
        
        from models import db, User, Job, File, Log, UsageStats
        print("✓ Database models import")
        
        from auth import init_auth, authenticate_user, load_user
        print("✓ Authentication imports")
        
        from config import Config
        print("✓ Configuration imports")
        
        # Core modules
        from core.runpod_client import RunPodClient
        print("✓ RunPod client imports")
        
        from core.storage_client import R2StorageClient
        print("✓ Storage client imports")
        
        from core.job_processor import JobProcessor
        print("✓ Job processor imports")
        
        from core.email_service import EmailService
        print("✓ Email service imports")
        
        from core.progress_tracker import ProgressTracker
        print("✓ Progress tracker imports")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_flask_app():
    """Test Flask app initialization"""
    print("\nTesting Flask app initialization...")
    
    try:
        from app import app
        
        # Test configuration
        assert app.config['SECRET_KEY'] is not None
        print("✓ Secret key configured")
        
        # Test routes exist
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        
        required_routes = ['/', '/login', '/upload', '/history', '/health']
        for route in required_routes:
            assert route in rules, f"Route {route} not found"
            print(f"✓ Route {route} exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app error: {e}")
        return False


def test_database():
    """Test database initialization"""
    print("\nTesting database...")
    
    try:
        from app import app, db
        from models import User
        
        with app.app_context():
            # Create tables
            db.create_all()
            print("✓ Database tables created")
            
            # Test user creation
            test_user = User(
                username='test_user',
                email='test@example.com'
            )
            test_user.set_password('test123')
            
            assert test_user.check_password('test123')
            print("✓ User password hashing works")
            
            # Don't actually save to avoid conflicts
            print("✓ Database models work")
            
        return True
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False


def test_templates():
    """Test templates exist"""
    print("\nTesting templates...")
    
    templates = [
        'templates/base.html',
        'templates/login.html',
        'templates/upload.html',
        'templates/progress.html',
        'templates/history.html',
        'templates/error.html',
        'templates/admin_dashboard.html',
        'templates/admin_logs.html'
    ]
    
    all_exist = True
    for template in templates:
        if os.path.exists(template):
            print(f"✓ {template} exists")
        else:
            print(f"✗ {template} missing")
            all_exist = False
    
    return all_exist


def test_static_files():
    """Test static files exist"""
    print("\nTesting static files...")
    
    static_files = [
        'static/js/upload.js',
        'static/js/progress.js',
        'static/css/style.css'
    ]
    
    all_exist = True
    for static_file in static_files:
        if os.path.exists(static_file):
            print(f"✓ {static_file} exists")
        else:
            print(f"✗ {static_file} missing")
            all_exist = False
    
    return all_exist


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        # Test defaults
        assert Config.MAX_CONTENT_LENGTH > 0
        print(f"✓ Max upload size: {Config.MAX_CONTENT_LENGTH / (1024*1024)} MB")
        
        assert Config.FILE_RETENTION_DAYS > 0
        print(f"✓ File retention: {Config.FILE_RETENTION_DAYS} days")
        
        assert Config.JOB_RETRY_LIMIT > 0
        print(f"✓ Job retry limit: {Config.JOB_RETRY_LIMIT}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Flask RunPod Application - Basic Tests")
    print("=" * 60)
    
    results = {
        'imports': test_imports(),
        'flask': test_flask_app(),
        'database': test_database(),
        'templates': test_templates(),
        'static': test_static_files(),
        'config': test_configuration()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("-" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.ljust(15)}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("-" * 60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ All tests passed! Application is ready.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure")
        print("2. Run: flask init-db")
        print("3. Run: python app.py")
        print("4. Visit: http://localhost:5000")
    else:
        print("\n❌ Some tests failed. Please review errors above.")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
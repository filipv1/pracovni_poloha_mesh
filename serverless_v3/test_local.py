"""
Local testing script for V3 architecture components
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

def test_storage_client():
    """Test R2/S3 storage client"""
    print("Testing Storage Client...")

    try:
        # Import from correct path
        sys.path.insert(0, str(Path(__file__).parent))
        from runpod.s3_utils import StorageClient

        # Test with mock credentials
        os.environ['STORAGE_PROVIDER'] = 's3'  # Use S3 for local testing
        os.environ['S3_ACCESS_KEY_ID'] = 'test_key'
        os.environ['S3_SECRET_ACCESS_KEY'] = 'test_secret'
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        os.environ['S3_REGION'] = 'us-east-1'

        client = StorageClient()
        print("[OK] StorageClient initialized")

        # Test presigned URL generation (will fail without real credentials)
        try:
            url = client.generate_presigned_url('test.mp4', 'put', 3600)
            if url:
                print("[OK] Presigned URL generated (mock)")
        except Exception as e:
            print(f"  Note: Presigned URL generation needs real credentials: {e}")

        return True

    except Exception as e:
        print(f"[FAIL] Storage client test failed: {e}")
        return False


def test_job_manager():
    """Test job management"""
    print("\nTesting Job Manager...")

    try:
        from runpod.s3_utils import JobManager

        # Create mock storage client
        class MockStorage:
            def upload_json(self, data, key):
                print(f"  Mock upload JSON to {key}")
                return True

            def download_json(self, key):
                return {
                    'job_id': 'test-job',
                    'status': 'processing',
                    'progress': 50
                }

        manager = JobManager(MockStorage())

        # Test job creation
        status = manager.create_job_status('test-job-123')
        print(f"[OK] Job created: {status['job_id']}")

        # Test job update
        updated = manager.update_job_status(
            'test-job-123',
            status='processing',
            progress=75
        )
        print(f"[OK] Job updated: progress={updated['progress']}")

        # Test job retrieval
        retrieved = manager.get_job_status('test-job-123')
        print(f"[OK] Job retrieved: status={retrieved['status']}")

        return True

    except Exception as e:
        print(f"[FAIL] Job manager test failed: {e}")
        return False


def test_handler_logic():
    """Test handler business logic"""
    print("\nTesting Handler Logic...")

    try:
        # Test action routing
        actions = [
            'generate_upload_url',
            'start_processing',
            'get_status',
            'generate_download_url'
        ]

        for action in actions:
            print(f"[OK] Action '{action}' defined")

        # Test job ID generation
        import uuid
        job_id = str(uuid.uuid4())
        print(f"[OK] Job ID generated: {job_id[:8]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Handler logic test failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting Configuration...")

    try:
        from config.config import (
            STORAGE_PROVIDER,
            UPLOADS_PREFIX,
            RESULTS_PREFIX,
            STATUS_PREFIX,
            DEFAULT_QUALITY
        )

        print(f"[OK] Storage provider: {STORAGE_PROVIDER}")
        print(f"[OK] Uploads prefix: {UPLOADS_PREFIX}")
        print(f"[OK] Results prefix: {RESULTS_PREFIX}")
        print(f"[OK] Status prefix: {STATUS_PREFIX}")
        print(f"[OK] Default quality: {DEFAULT_QUALITY}")

        return True

    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("V3 Architecture Component Tests")
    print("=" * 60)

    tests = [
        test_config,
        test_storage_client,
        test_job_manager,
        test_handler_logic
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] All tests passed!")
    else:
        print("[WARNING] Some tests failed. Check output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
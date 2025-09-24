"""
Test handlers with mock RunPod for local testing
Works without installing runpod package
"""

import sys
import os
import base64

# Try to import runpod, use mock if not available
try:
    import runpod
    print("[INFO] Using real runpod module")
    USING_MOCK = False
except ImportError:
    print("[INFO] RunPod not installed, using mock for testing")
    from mock_runpod import runpod
    USING_MOCK = True
    # Inject mock into sys.modules so handlers can import it
    sys.modules['runpod'] = runpod

def test_handler(handler_name, handler_module):
    """Test a handler module"""
    print(f"\n{'='*50}")
    print(f"Testing {handler_name}")
    print('='*50)

    try:
        # Import handler
        handler_func = handler_module.handler

        # Create test job
        test_job = {
            "input": {
                "video_base64": base64.b64encode(b"test video data").decode('utf-8'),
                "video_name": "test.mp4",
                "quality": "medium",
                "user_email": "test@example.com"
            }
        }

        print(f"[OK] {handler_name} imported successfully")
        print("Running handler...")

        # Execute handler
        result = handler_func(test_job)

        # Check result
        print(f"Result status: {result.get('status')}")

        if result.get('status') == 'success':
            print(f"[PASS] {handler_name} executed successfully")

            # Check for expected fields
            if 'xlsx_base64' in result:
                print("  - xlsx_base64: present")
            if 'pkl_base64' in result:
                print("  - pkl_base64: present")
            if 'statistics' in result:
                print(f"  - statistics: {result.get('statistics')}")

            return True
        else:
            print(f"[FAIL] Handler returned error: {result.get('error')}")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("HANDLER TESTING WITH MOCK RUNPOD")
    print("="*50)

    if USING_MOCK:
        print("\n[WARNING] Using mock RunPod module")
        print("This tests handler logic but not actual RunPod integration")
        print("To test with real RunPod: pip install runpod")

    # Test results
    results = []

    # Test minimal handler
    try:
        import handler_minimal
        results.append(("Minimal Handler", test_handler("handler_minimal", handler_minimal)))
    except ImportError as e:
        print(f"[ERROR] Cannot import handler_minimal: {e}")
        results.append(("Minimal Handler", False))

    # Test fixed handler
    try:
        import handler_fixed
        results.append(("Fixed Handler", test_handler("handler_fixed", handler_fixed)))
    except ImportError as e:
        print(f"[ERROR] Cannot import handler_fixed: {e}")
        results.append(("Fixed Handler", False))

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: [{status}]")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[SUCCESS] All handler tests passed!")
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t test .")
        print("2. Test in Docker: docker run test python handler_minimal.py")
        print("3. Deploy to RunPod: docker push ...")
    else:
        print("\n[WARNING] Some tests failed")
        print("Check the errors above before deploying")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
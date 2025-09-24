#!/usr/bin/env python3
"""
Test frontend HTML functionality
"""

from pathlib import Path
import re

def test_frontend():
    """Test frontend HTML for proper download handling"""
    print("=" * 50)
    print("TESTING FRONTEND HTML")
    print("=" * 50)

    frontend_path = Path(__file__).parent / "frontend" / "index-with-proxy.html"

    if not frontend_path.exists():
        print(f"[FAIL] Frontend not found: {frontend_path}")
        return False

    print(f"[OK] Frontend found: {frontend_path}")

    with open(frontend_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for required functionality
    checks = {
        'Multiple download buttons': 'downloadButtons',
        'PKL download': 'pkl_url',
        'CSV download': 'csv_url',
        'Excel download': 'excel_url',
        'Video downloads': 'results.videos',
        'createDownloadButton function': 'function createDownloadButton',
        'Progress updates': 'Step 4/4',
        'Job status polling': 'startPolling',
    }

    all_good = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name} not found")
            all_good = False

    # Check for removed single download button
    if 'const downloadButton = document.getElementById' not in content:
        print(f"  [OK] Old single download button removed")
    else:
        print(f"  [WARNING] Old single download button still present")

    return all_good

def test_proxy_config():
    """Test proxy configuration in HTML"""
    print("\n" + "=" * 50)
    print("TESTING PROXY CONFIGURATION")
    print("=" * 50)

    frontend_path = Path(__file__).parent / "frontend" / "index-with-proxy.html"

    with open(frontend_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check proxy settings
    if "PROXY_URL = 'http://localhost:5001'" in content:
        print("[OK] Proxy URL configured correctly")
    else:
        print("[FAIL] Proxy URL not configured")
        return False

    if "USE_PROXY = true" in content:
        print("[OK] Proxy enabled")
    else:
        print("[WARNING] Proxy not enabled")

    # Check API calls go through proxy
    if "`${PROXY_URL}/runpod`" in content:
        print("[OK] RunPod calls use proxy")
    else:
        print("[FAIL] RunPod calls don't use proxy")
        return False

    if "`${PROXY_URL}/upload" in content:
        print("[OK] Upload calls use proxy")
    else:
        print("[FAIL] Upload calls don't use proxy")
        return False

    return True

def test_progress_messages():
    """Test if all progress steps are shown"""
    print("\n" + "=" * 50)
    print("TESTING PROGRESS MESSAGES")
    print("=" * 50)

    frontend_path = Path(__file__).parent / "frontend" / "index-with-proxy.html"

    with open(frontend_path, 'r', encoding='utf-8') as f:
        content = f.read()

    steps = [
        'Upload video to cloud storage',
        'Process with SMPL-X pipeline',
        'Download results'
    ]

    all_good = True
    for step in steps:
        if step in content:
            print(f"  [OK] Step: {step}")
        else:
            print(f"  [FAIL] Missing step: {step}")
            all_good = False

    return all_good

def main():
    """Run all frontend tests"""
    print("\n" + "=" * 50)
    print("FRONTEND TEST SUITE")
    print("=" * 50)

    results = {
        'frontend_functionality': test_frontend(),
        'proxy_config': test_proxy_config(),
        'progress_messages': test_progress_messages()
    }

    print("\n" + "=" * 50)
    print("FRONTEND TEST SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{test_name:25s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL FRONTEND TESTS PASSED!")
    else:
        print("SOME FRONTEND TESTS FAILED!")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
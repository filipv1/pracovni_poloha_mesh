"""
Simple test to verify basic functionality without full pipeline
"""

import sys
import os

print("=== Simple Test ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Test imports
try:
    import numpy as np
    print("✓ NumPy imported")

    import cv2
    print("✓ OpenCV imported")

    import mediapipe
    print("✓ MediaPipe imported")

    import runpod
    print("✓ RunPod imported")

    # Test if main files exist
    files_to_check = [
        'handler.py',
        'run_production_simple.py',
        'production_3d_pipeline_clean.py'
    ]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} NOT FOUND")

    print("\n=== Basic test passed! ===")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
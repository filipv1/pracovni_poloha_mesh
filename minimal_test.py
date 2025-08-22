#!/usr/bin/env python3
"""
Minimal test to isolate the hanging issue
"""

import os
import sys
import numpy as np
import cv2
import torch
import mediapipe as mp
from pathlib import Path
import time

print("MINIMAL TEST STARTING")
print("=" * 30)

# Test 1: Basic imports
print("Testing imports...")
try:
    import smplx
    print("OK SMPL-X imported successfully")
except Exception as e:
    print(f"ERROR SMPL-X import failed: {e}")

# Test 2: MediaPipe
print("Testing MediaPipe...")
try:
    mp_pose = mp.solutions.pose.Pose()
    print("OK MediaPipe Pose initialized")
except Exception as e:
    print(f"ERROR MediaPipe failed: {e}")

# Test 3: Basic video reading
print("Testing video reading...")
try:
    if Path("test.mp4").exists():
        cap = cv2.VideoCapture("test.mp4")
        ret, frame = cap.read()
        if ret:
            print(f"OK Video frame read: {frame.shape}")
        cap.release()
    else:
        print("ERROR test.mp4 not found")
except Exception as e:
    print(f"ERROR Video reading failed: {e}")

# Test 4: SMPL-X model loading (this might be where it hangs)
print("Testing SMPL-X model loading...")
try:
    device = 'cpu'  # Force CPU to avoid GPU issues
    print(f"Using device: {device}")
    
    smplx_path = "models/smplx"
    if Path(smplx_path).exists():
        print(f"OK SMPL-X path exists: {smplx_path}")
        
        # This is likely where it hangs - let's test step by step
        print("Loading SMPL-X model...")
        model = smplx.SMPLX(
            model_path=smplx_path,
            gender='neutral',
            use_face_contour=False,
            use_hands=False,
            num_betas=10,
            num_expression_coeffs=0,
            create_global_orient=True,
            create_body_pose=True,
            create_transl=True
        )
        model = model.to(device)
        print("OK SMPL-X model loaded successfully!")
    else:
        print(f"ERROR SMPL-X path not found: {smplx_path}")
        
except Exception as e:
    print(f"ERROR SMPL-X model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("MINIMAL TEST COMPLETED")
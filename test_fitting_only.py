#!/usr/bin/env python3
"""
Test just SMPL-X fitting without rendering
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
from pathlib import Path

print("TESTING SMPL-X FITTING ONLY")
print("=" * 40)

try:
    # Import the pipeline
    from run_production_simple import MasterPipeline, PreciseMediaPipeConverter
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize processor
    processor = PreciseMediaPipeConverter()
    
    # Initialize pipeline
    pipeline = MasterPipeline(
        smplx_path="models/smplx",
        device='cpu',
        gender='neutral'
    )
    
    print("OK All components initialized")
    
    # Read test video frame
    cap = cv2.VideoCapture("test.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR Could not read video frame")
        sys.exit(1)
    
    print(f"OK Read frame: {frame.shape}")
    
    # Process frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        print("ERROR No pose landmarks detected")
        sys.exit(1)
        
    print("OK MediaPipe detected pose landmarks")
    
    # Convert to 3D landmarks
    conversion_result = processor.convert_landmarks_to_smplx(results.pose_landmarks)
    if conversion_result is None:
        print("ERROR Could not convert to 3D landmarks")
        sys.exit(1)
    
    # Check if tuple (joint positions, confidence)
    if isinstance(conversion_result, tuple):
        landmarks_3d, confidence = conversion_result
        print(f"OK Converted to 3D landmarks: {landmarks_3d.shape}, confidence: {confidence}")
    else:
        landmarks_3d = conversion_result
        print(f"OK Converted to 3D landmarks: {landmarks_3d.shape}")
    
    # Fit SMPL-X mesh (this is where it might hang)
    print("Fitting SMPL-X mesh...")
    mesh_data = pipeline.mesh_fitter.fit_mesh_to_landmarks(landmarks_3d)
    
    if mesh_data is None:
        print("ERROR SMPL-X fitting failed")
        sys.exit(1)
    
    print(f"OK SMPL-X mesh fitted: {mesh_data['vertices'].shape[0]} vertices")
    print("SUCCESS: All components working correctly!")
    
except Exception as e:
    print(f"ERROR Test failed: {e}")
    import traceback
    traceback.print_exc()

print("TEST COMPLETED")
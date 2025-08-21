#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK MEDIAPIPE TEST SCRIPT
===========================

Quick validation of MediaPipe pose detection functionality
using the existing pracovni_poloha2 pipeline.

Usage:
    python quick_mediapipe_test.py

This will process first 10 frames of your input video and show results.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add pracovni_poloha2 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pracovni_poloha2', 'src'))

def test_mediapipe_quick():
    """Quick test of MediaPipe pose detection"""
    
    print("QUICK MEDIAPIPE TEST")
    print("=" * 40)
    
    input_video = "input_video.mp4"
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        print("Please place your test video as 'input_video.mp4'")
        return False
    
    try:
        # Import MediaPipe components
        from pose_detector import PoseDetector
        
        print("✓ MediaPipe components imported successfully")
        
        # Initialize pose detector
        detector = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ PoseDetector initialized")
        
        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("❌ Cannot open input video")
            return False
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Video opened: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Test first 10 frames
        test_frames = min(10, total_frames)
        successful_detections = 0
        landmark_data = []
        
        print(f"\nTesting first {test_frames} frames:")
        
        for frame_num in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect poses
            pose_results = detector.detect_pose(frame)
            
            if pose_results.pose_world_landmarks is not None:
                landmarks_3d = detector.extract_3d_landmarks(pose_results.pose_world_landmarks)
                if landmarks_3d is not None:
                    successful_detections += 1
                    landmark_data.append(landmarks_3d)
                    print(f"  Frame {frame_num+1:2d}: + Pose detected ({len(landmarks_3d)} landmarks)")
                    
                    # Show sample landmark (nose tip)
                    if len(landmarks_3d) >= 1:
                        nose = landmarks_3d[0]  # First landmark is usually nose
                        print(f"             Nose position: ({nose[0]:.3f}, {nose[1]:.3f}, {nose[2]:.3f})")
                else:
                    print(f"  Frame {frame_num+1:2d}: - Failed to extract 3D landmarks")
            else:
                print(f"  Frame {frame_num+1:2d}: X No pose detected")
        
        cap.release()
        # MediaPipe detector doesn't need explicit close
        
        # Results summary
        success_rate = (successful_detections / test_frames) * 100
        
        print(f"\nRESULTS:")
        print(f"  Successful detections: {successful_detections}/{test_frames}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if success_rate >= 70:
            print("  ✓ EXCELLENT: MediaPipe is working very well")
        elif success_rate >= 50:
            print("  ⚠ GOOD: MediaPipe is working, but consider better lighting/positioning")
        else:
            print("  ❌ POOR: MediaPipe detection is struggling with this video")
        
        # Analyze landmark stability if we have multiple frames
        if len(landmark_data) >= 3:
            print(f"\nLANDMARK STABILITY ANALYSIS:")
            
            # Calculate movement variance for key landmarks
            nose_positions = [landmarks[0] for landmarks in landmark_data]
            nose_variance = np.var(nose_positions, axis=0)
            avg_variance = np.mean(nose_variance)
            
            print(f"  Average position variance: {avg_variance:.6f}")
            if avg_variance < 0.01:
                print("  ✓ STABLE: Low jitter in landmark detection")
            elif avg_variance < 0.05:
                print("  ⚠ MODERATE: Some jitter in landmark detection")
            else:
                print("  ❌ UNSTABLE: High jitter in landmark detection")
        
        return success_rate >= 50
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure MediaPipe is installed: pip install mediapipe")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trunk_analysis():
    """Quick test of trunk angle analysis"""
    
    print("\n" + "=" * 40)
    print("TRUNK ANALYSIS TEST")
    print("=" * 40)
    
    try:
        from trunk_analyzer import TrunkAngleCalculator
        
        # Create sample landmarks (33 3D points)
        # This simulates a person standing straight
        sample_landmarks = np.zeros((33, 3))
        
        # Set some key landmarks for a standing pose
        sample_landmarks[11] = [0.0, 0.5, 0.0]  # Left shoulder
        sample_landmarks[12] = [0.0, -0.5, 0.0]  # Right shoulder  
        sample_landmarks[23] = [0.0, 0.3, -0.5]  # Left hip
        sample_landmarks[24] = [0.0, -0.3, -0.5]  # Right hip
        
        calculator = TrunkAngleCalculator()
        angle = calculator.calculate_trunk_angle(sample_landmarks)
        
        if angle is not None:
            print(f"✓ Trunk angle calculation successful: {angle:.1f}°")
            return True
        else:
            print("❌ Trunk angle calculation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Trunk analysis test failed: {e}")
        return False

def main():
    """Main test function"""
    print("MediaPipe Pipeline Quick Test")
    print("This will test MediaPipe functionality with your input video.\n")
    
    # Run tests
    mp_success = test_mediapipe_quick()
    trunk_success = test_trunk_analysis()
    
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    if mp_success and trunk_success:
        print("🎉 ALL TESTS PASSED!")
        print("MediaPipe functionality is working correctly.")
        print("\nYou can now run the full pipeline:")
        print("  python production_3d_pipeline_clean.py")
    elif mp_success:
        print("⚠️  PARTIAL SUCCESS")
        print("MediaPipe detection works, but trunk analysis has issues.")
    else:
        print("❌ TESTS FAILED")
        print("MediaPipe detection is not working properly.")
        print("\nTroubleshooting:")
        print("1. Ensure your video has clear human poses")
        print("2. Check lighting and video quality")
        print("3. Try with a different test video")
        print("4. Verify MediaPipe installation: pip install mediapipe")

if __name__ == "__main__":
    main()
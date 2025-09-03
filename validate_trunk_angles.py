#!/usr/bin/env python3
"""
Validate trunk angle accuracy by comparing:
1. Raw MediaPipe landmarks (ground truth)
2. arm_meshes.pkl (old individual processing)  
3. fast_meshes.pkl (new batch processing)
"""

import cv2
import sys
import os
sys.path.append('.')  # Add current directory to path
import numpy as np
import pickle

def setup_mediapipe():
    """Setup MediaPipe with proper imports"""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return pose, mp_pose
    except ImportError:
        print("ERROR: MediaPipe not available. Please install: pip install mediapipe")
        return None, None

def calculate_mediapipe_trunk_angle(landmarks):
    """Calculate trunk angle from MediaPipe pose landmarks"""
    if not landmarks or len(landmarks.landmark) < 33:
        return None
        
    # MediaPipe landmark indices
    # 11 = LEFT_SHOULDER, 12 = RIGHT_SHOULDER 
    # 23 = LEFT_HIP, 24 = RIGHT_HIP
    
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12] 
    left_hip = landmarks.landmark[23]
    right_hip = landmarks.landmark[24]
    
    # Get 3D world coordinates
    shoulder_center = np.array([
        (left_shoulder.x + right_shoulder.x) / 2,
        (left_shoulder.y + right_shoulder.y) / 2, 
        (left_shoulder.z + right_shoulder.z) / 2
    ])
    
    hip_center = np.array([
        (left_hip.x + right_hip.x) / 2,
        (left_hip.y + right_hip.y) / 2,
        (left_hip.z + right_hip.z) / 2  
    ])
    
    # Calculate trunk vector (from hips to shoulders)
    trunk_vector = shoulder_center - hip_center
    trunk_length = np.linalg.norm(trunk_vector)
    
    if trunk_length > 0:
        trunk_unit = trunk_vector / trunk_length
        # MediaPipe world coordinates: Y points up (same as SMPL-X)
        vertical = np.array([0, 1, 0])  # Y-up
        
        cos_angle = np.dot(trunk_unit, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        trunk_angle = np.degrees(np.arccos(cos_angle))
        return trunk_angle
    
    return None

def calculate_smplx_trunk_angle(joints):
    """Calculate trunk angle from SMPL-X joints (same as create_combined_angles_csv.py)"""
    from arm_angle_calculator import SMPL_X_JOINT_INDICES
    
    spine1_idx = SMPL_X_JOINT_INDICES['spine1']
    neck_idx = SMPL_X_JOINT_INDICES['neck']
    
    lumbar_joint = joints[spine1_idx]
    cervical_joint = joints[neck_idx]
    trunk_vector = cervical_joint - lumbar_joint
    trunk_length = np.linalg.norm(trunk_vector)
    
    if trunk_length > 0:
        spine_unit = trunk_vector / trunk_length
        vertical = np.array([0, 1, 0])  # Y-up coordinate system (SMPL-X)
        cos_angle = np.dot(spine_unit, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        trunk_angle_deg = np.degrees(np.arccos(cos_angle))
        return trunk_angle_deg
    
    return None

def main():
    """Main validation function"""
    
    print("TRUNK ANGLE ACCURACY VALIDATION")
    print("=" * 50)
    
    # Setup MediaPipe
    pose, mp_pose = setup_mediapipe()
    if not pose:
        return 1
        
    # Load PKL data
    print("Loading PKL files...")
    with open('arm_meshes.pkl', 'rb') as f:
        arm_data = pickle.load(f)
    with open('fast_meshes.pkl', 'rb') as f:
        fast_data = pickle.load(f)
    
    # Open video
    video_path = 'test.mp4'
    if not os.path.exists(video_path):
        print(f"ERROR: Video file '{video_path}' not found!")
        return 1
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file!")
        return 1
    
    # Focus on problematic frames (119-122 where differences are largest)
    problematic_frames = [119, 120, 121, 122]
    
    print(f"\nAnalyzing problematic frames: {problematic_frames}")
    print(f"Frame | MediaPipe | ARM_OLD | FAST_NEW | MP-ARM | MP-FAST")
    print("-" * 65)
    
    results = []
    
    for frame_idx in problematic_frames:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue
            
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = pose.process(rgb_frame)
        
        # Calculate MediaPipe trunk angle (ground truth)
        mp_trunk_angle = None
        if mp_results.pose_world_landmarks:
            mp_trunk_angle = calculate_mediapipe_trunk_angle(mp_results.pose_world_landmarks)
        
        # Get SMPL-X angles from PKL files
        arm_trunk_angle = None
        fast_trunk_angle = None
        
        if frame_idx < len(arm_data):
            arm_trunk_angle = calculate_smplx_trunk_angle(arm_data[frame_idx]['joints'])
            
        if frame_idx < len(fast_data):
            fast_trunk_angle = calculate_smplx_trunk_angle(fast_data[frame_idx]['joints'])
        
        # Calculate differences from ground truth
        mp_arm_diff = abs(mp_trunk_angle - arm_trunk_angle) if mp_trunk_angle and arm_trunk_angle else None
        mp_fast_diff = abs(mp_trunk_angle - fast_trunk_angle) if mp_trunk_angle and fast_trunk_angle else None
        
        # Display results
        mp_str = f"{mp_trunk_angle:5.1f}°" if mp_trunk_angle else "  N/A"
        arm_str = f"{arm_trunk_angle:5.1f}°" if arm_trunk_angle else "  N/A"
        fast_str = f"{fast_trunk_angle:5.1f}°" if fast_trunk_angle else "  N/A"
        mp_arm_str = f"{mp_arm_diff:5.1f}°" if mp_arm_diff else "  N/A"
        mp_fast_str = f"{mp_fast_diff:5.1f}°" if mp_fast_diff else "  N/A"
        
        print(f"{frame_idx:5d} | {mp_str} | {arm_str} | {fast_str} | {mp_arm_str} | {mp_fast_str}")
        
        results.append({
            'frame': frame_idx,
            'mediapipe': mp_trunk_angle,
            'arm_old': arm_trunk_angle, 
            'fast_new': fast_trunk_angle,
            'mp_arm_diff': mp_arm_diff,
            'mp_fast_diff': mp_fast_diff
        })
    
    cap.release()
    pose.close()
    
    # Analysis summary
    print(f"\nACCURACY ANALYSIS:")
    print("=" * 50)
    
    valid_results = [r for r in results if r['mp_arm_diff'] and r['mp_fast_diff']]
    if valid_results:
        arm_errors = [r['mp_arm_diff'] for r in valid_results]
        fast_errors = [r['mp_fast_diff'] for r in valid_results]
        
        avg_arm_error = np.mean(arm_errors)
        avg_fast_error = np.mean(fast_errors)
        
        print(f"Average error from MediaPipe ground truth:")
        print(f"  arm_meshes.pkl (old):  {avg_arm_error:.1f}°")
        print(f"  fast_meshes.pkl (new): {avg_fast_error:.1f}°")
        print()
        
        if avg_arm_error < avg_fast_error:
            print(f"🏆 WINNER: arm_meshes.pkl (old version) is MORE ACCURATE")
            print(f"   Old version is {avg_fast_error - avg_arm_error:.1f}° closer to MediaPipe ground truth")
        else:
            print(f"🏆 WINNER: fast_meshes.pkl (new version) is MORE ACCURATE") 
            print(f"   New version is {avg_arm_error - avg_fast_error:.1f}° closer to MediaPipe ground truth")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
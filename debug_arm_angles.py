#!/usr/bin/env python3
"""
Debug script to analyze arm angle calculation issues
Investigates why arm angles are showing negative values when they should be positive
"""

import numpy as np
import pickle
from pathlib import Path
from arm_angle_calculator import calculate_arm_angle_to_trunk_robust, SMPL_X_JOINT_INDICES

def debug_vector_components(joints, arm_side='left', frame_idx=0):
    """
    Debug function to examine raw vector components and atan2 calculations
    """
    
    print(f"\n=== FRAME {frame_idx} - {arm_side.upper()} ARM DEBUG ===")
    
    # Get anatomical points (same as in the original function)
    lumbar = joints[SMPL_X_JOINT_INDICES['spine1']]      # L3/L4
    cervical = joints[SMPL_X_JOINT_INDICES['neck']]       # C7/T1
    left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
    right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
    
    if arm_side == 'left':
        shoulder = left_shoulder
        elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        side_sign = 1
    else:
        shoulder = right_shoulder
        elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        side_sign = -1
    
    print(f"Raw positions:")
    print(f"  Lumbar (spine1): {lumbar}")
    print(f"  Cervical (neck): {cervical}")
    print(f"  Shoulder: {shoulder}")
    print(f"  Elbow: {elbow}")
    
    # Calculate vectors
    trunk_vector = cervical - lumbar
    arm_vector = elbow - shoulder  
    shoulder_width_vector = right_shoulder - left_shoulder
    
    print(f"\nRaw vectors:")
    print(f"  Trunk vector: {trunk_vector}")
    print(f"  Arm vector: {arm_vector}")
    print(f"  Shoulder width vector: {shoulder_width_vector}")
    
    # Calculate coordinate system
    trunk_length = np.linalg.norm(trunk_vector)
    arm_length = np.linalg.norm(arm_vector)
    shoulder_width = np.linalg.norm(shoulder_width_vector)
    
    print(f"\nVector lengths:")
    print(f"  Trunk length: {trunk_length:.4f}")
    print(f"  Arm length: {arm_length:.4f}")
    print(f"  Shoulder width: {shoulder_width:.4f}")
    
    # Anatomical coordinate system (same as original)
    trunk_up = trunk_vector / trunk_length
    shoulder_right = shoulder_width_vector / shoulder_width
    body_forward_unnorm = np.cross(shoulder_right, trunk_up)
    body_forward_length = np.linalg.norm(body_forward_unnorm)
    
    if body_forward_length < 1e-8:
        body_forward = np.array([1, 0, 0])  # fallback
    else:
        body_forward = body_forward_unnorm / body_forward_length
    
    shoulder_right = np.cross(trunk_up, body_forward) 
    
    print(f"\nCoordinate system:")
    print(f"  Trunk up: {trunk_up}")
    print(f"  Body forward: {body_forward}")
    print(f"  Shoulder right: {shoulder_right}")
    
    # Project arm vector onto coordinate axes
    arm_norm = arm_vector / arm_length
    arm_forward_comp = np.dot(arm_norm, body_forward)
    arm_up_comp = np.dot(arm_norm, trunk_up)
    arm_right_comp = np.dot(arm_norm, shoulder_right)
    
    print(f"\nArm vector components:")
    print(f"  Normalized arm vector: {arm_norm}")
    print(f"  Forward component: {arm_forward_comp:.4f}")
    print(f"  Up component: {arm_up_comp:.4f}")
    print(f"  Right component: {arm_right_comp:.4f}")
    
    # Current atan2 calculation (from line 119)
    current_angle_rad = np.arctan2(-arm_forward_comp, -arm_up_comp)
    current_angle_deg = np.degrees(current_angle_rad)
    
    print(f"\nCURRENT atan2 calculation:")
    print(f"  atan2(-arm_forward_comp, -arm_up_comp)")
    print(f"  atan2({-arm_forward_comp:.4f}, {-arm_up_comp:.4f})")
    print(f"  = {current_angle_rad:.4f} rad = {current_angle_deg:.1f}°")
    
    # Let's try different atan2 formulations
    print(f"\nALTERNATIVE atan2 formulations:")
    
    # Option 1: Direct mapping
    alt1_rad = np.arctan2(arm_forward_comp, -arm_up_comp)
    alt1_deg = np.degrees(alt1_rad)
    print(f"  Option 1 - atan2(arm_forward_comp, -arm_up_comp)")
    print(f"    atan2({arm_forward_comp:.4f}, {-arm_up_comp:.4f}) = {alt1_deg:.1f}°")
    
    # Option 2: Standard trigonometry
    alt2_rad = np.arctan2(arm_forward_comp, arm_up_comp)
    alt2_deg = np.degrees(alt2_rad)
    print(f"  Option 2 - atan2(arm_forward_comp, arm_up_comp)")
    print(f"    atan2({arm_forward_comp:.4f}, {arm_up_comp:.4f}) = {alt2_deg:.1f}°")
    
    # Option 3: Adjusted for hanging down = 0
    alt3_rad = np.arctan2(arm_forward_comp, -arm_up_comp)
    alt3_deg = np.degrees(alt3_rad) 
    print(f"  Option 3 - atan2(arm_forward_comp, -arm_up_comp)")
    print(f"    atan2({arm_forward_comp:.4f}, {-arm_up_comp:.4f}) = {alt3_deg:.1f}°")
    
    # Expected behavior analysis
    print(f"\nEXPECTED BEHAVIOR ANALYSIS:")
    if abs(arm_up_comp) > abs(arm_forward_comp):
        if arm_up_comp > 0:
            print(f"  Arm pointing UP relative to trunk (arm_up_comp = {arm_up_comp:.4f})")
            print(f"  Expected: Negative angle (arm raised above horizontal)")
        else:
            print(f"  Arm pointing DOWN relative to trunk (arm_up_comp = {arm_up_comp:.4f})")
            print(f"  Expected: Near 0° (arm hanging)")
    else:
        if arm_forward_comp > 0:
            print(f"  Arm pointing FORWARD (arm_forward_comp = {arm_forward_comp:.4f})")
            print(f"  Expected: Positive angle (~+90° for full forward)")
        else:
            print(f"  Arm pointing BACKWARD (arm_forward_comp = {arm_forward_comp:.4f})")
            print(f"  Expected: Negative angle (~-90° for full backward)")
    
    print("-" * 60)
    
    return {
        'arm_forward_comp': arm_forward_comp,
        'arm_up_comp': arm_up_comp,
        'current_angle': current_angle_deg,
        'alt1_angle': alt1_deg,
        'alt2_angle': alt2_deg,
        'alt3_angle': alt3_deg
    }


def analyze_first_few_frames(pkl_file, num_frames=5):
    """
    Load and analyze the first few frames to understand the angle calculation issue
    """
    
    print(f"DEBUGGING ARM ANGLE CALCULATION")
    print(f"Loading from: {pkl_file}")
    print("=" * 80)
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"Loaded {len(meshes)} frames")
    
    debug_results = []
    
    for frame_idx in range(min(num_frames, len(meshes))):
        mesh_data = meshes[frame_idx]
        joints = mesh_data['joints']  # (117, 3)
        
        # Debug both arms for this frame
        left_debug = debug_vector_components(joints, 'left', frame_idx)
        right_debug = debug_vector_components(joints, 'right', frame_idx)
        
        debug_results.append({
            'frame': frame_idx,
            'left': left_debug,
            'right': right_debug
        })
        
        # Also get the official result for comparison
        official_result = calculate_arm_angle_to_trunk_robust(joints, 'left')
        if official_result:
            print(f"OFFICIAL RESULT for LEFT ARM:")
            print(f"  Sagittal angle: {official_result['sagittal_angle']:.1f}°")
            print(f"  Confidence: {official_result['confidence']:.3f}")
        
        print("\n" + "="*80 + "\n")
    
    # Summary analysis
    print("\nSUMMARY ANALYSIS:")
    print("=" * 50)
    
    for i, result in enumerate(debug_results):
        left = result['left']
        right = result['right']
        
        print(f"Frame {i}:")
        print(f"  Left arm  - Forward: {left['arm_forward_comp']:+.3f}, Up: {left['arm_up_comp']:+.3f}")
        print(f"            Current: {left['current_angle']:+6.1f}°, Option1: {left['alt1_angle']:+6.1f}°")
        print(f"  Right arm - Forward: {right['arm_forward_comp']:+.3f}, Up: {right['arm_up_comp']:+.3f}")
        print(f"            Current: {right['current_angle']:+6.1f}°, Option1: {right['alt1_angle']:+6.1f}°")
        print()
    
    return debug_results


def recommend_fix():
    """
    Provide recommendations for fixing the angle calculation
    """
    
    print("\nRECOMMENDED FIX:")
    print("=" * 50)
    print("""
PROBLEM ANALYSIS:
The current formula uses: atan2(-arm_forward_comp, -arm_up_comp)
This double negative is causing issues with the expected behavior.

EXPECTED BEHAVIOR:
- 0° when arms hang down along trunk (arm_up_comp < 0, arm_forward_comp ~= 0)
- +90° when arms point forward (arm_up_comp ~= 0, arm_forward_comp > 0)
- -90° when arms point backward (arm_up_comp ~= 0, arm_forward_comp < 0)

RECOMMENDED FORMULA:
Change line 119 from:
    sagittal_angle_rad = np.arctan2(-arm_forward_comp, -arm_up_comp)
    
To:
    sagittal_angle_rad = np.arctan2(arm_forward_comp, -arm_up_comp)

EXPLANATION:
- We keep the negative on arm_up_comp because we want hanging down (negative up) to be 0°
- We remove the negative on arm_forward_comp so forward motion gives positive angles
- This aligns with anatomical convention where forward flexion is positive
    """)


if __name__ == "__main__":
    pkl_file = "arm_meshes.pkl"
    
    if Path(pkl_file).exists():
        debug_results = analyze_first_few_frames(pkl_file, num_frames=3)
        recommend_fix()
    else:
        print(f"PKL file not found: {pkl_file}")
        print("Available PKL files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
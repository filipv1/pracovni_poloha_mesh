#!/usr/bin/env python3
"""
Debug script to check if left/right arms are correctly assigned
"""

import pickle
import numpy as np
from arm_angle_calculator import SMPL_X_JOINT_INDICES

def debug_arm_sides(pkl_file, num_frames=5):
    """Check if left/right arms are correctly assigned"""
    
    print("DEBUGGING ARM SIDE ASSIGNMENT")
    print("=" * 50)
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    for frame_idx in range(min(num_frames, len(meshes))):
        joints = meshes[frame_idx]['joints']
        
        # Get shoulder and elbow positions
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Calculate shoulder width and midpoint
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        shoulder_width_vector = right_shoulder - left_shoulder
        shoulder_width = np.linalg.norm(shoulder_width_vector)
        
        print(f"\nFrame {frame_idx}:")
        print(f"  Shoulder midpoint: [{shoulder_midpoint[0]:.3f}, {shoulder_midpoint[1]:.3f}, {shoulder_midpoint[2]:.3f}]")
        print(f"  Shoulder width: {shoulder_width:.3f}m")
        
        print(f"  LEFT  shoulder:  [{left_shoulder[0]:.3f}, {left_shoulder[1]:.3f}, {left_shoulder[2]:.3f}]")
        print(f"  RIGHT shoulder:  [{right_shoulder[0]:.3f}, {right_shoulder[1]:.3f}, {right_shoulder[2]:.3f}]")
        
        print(f"  LEFT  elbow:     [{left_elbow[0]:.3f}, {left_elbow[1]:.3f}, {left_elbow[2]:.3f}]")
        print(f"  RIGHT elbow:     [{right_elbow[0]:.3f}, {right_elbow[1]:.3f}, {right_elbow[2]:.3f}]")
        
        # Check X coordinates to verify left/right assignment
        # In a typical SMPL-X coordinate system:
        # - Person facing camera (negative Z direction)
        # - Left side should have LOWER X coordinate
        # - Right side should have HIGHER X coordinate
        
        left_is_actually_left = left_shoulder[0] < right_shoulder[0]
        
        print(f"  X-coordinate check:")
        print(f"    Left shoulder X:  {left_shoulder[0]:.3f}")
        print(f"    Right shoulder X: {right_shoulder[0]:.3f}")
        print(f"    Assignment correct: {'YES' if left_is_actually_left else 'NO - SWAPPED!'}")
        
        # Calculate arm vectors
        left_arm_vec = left_elbow - left_shoulder
        right_arm_vec = right_elbow - right_shoulder
        
        print(f"  Arm vectors:")
        print(f"    Left arm:  [{left_arm_vec[0]:.3f}, {left_arm_vec[1]:.3f}, {left_arm_vec[2]:.3f}]")
        print(f"    Right arm: [{right_arm_vec[0]:.3f}, {right_arm_vec[1]:.3f}, {right_arm_vec[2]:.3f}]")
        
    print(f"\n" + "=" * 50)
    print("INTERPRETATION:")
    print("- In SMPL-X coordinate system, person typically faces camera (negative Z)")
    print("- Left side should have LOWER X coordinate")
    print("- Right side should have HIGHER X coordinate")
    print("- If assignment is wrong, we need to swap joint indices!")

if __name__ == "__main__":
    debug_arm_sides("arm_meshes.pkl", num_frames=3)
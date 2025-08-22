#!/usr/bin/env python3
"""
Prozkoumání kloubů v mesh datech - zejména páteř a krk
"""

import pickle
import numpy as np

def explore_joints(pkl_file):
    """Prozkoumej klouby v PKL souboru"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    if not meshes:
        print("No meshes found!")
        return
    
    # První frame
    joints = meshes[0]['joints']
    print(f"JOINTS ANALYSIS: {len(joints)} total joints")
    print("=" * 50)
    
    # SMPL-X joint mapování (aproximace)
    joint_names = {
        0: "pelvis",
        1: "left_hip", 2: "right_hip", 3: "spine1",
        4: "left_knee", 5: "right_knee", 6: "spine2", 
        7: "left_ankle", 8: "right_ankle", 9: "spine3",
        10: "left_foot", 11: "right_foot", 12: "neck",
        13: "left_collar", 14: "right_collar", 15: "head",
        16: "left_shoulder", 17: "right_shoulder",
        18: "left_elbow", 19: "right_elbow",
        20: "left_wrist", 21: "right_wrist"
    }
    
    print("KEY JOINTS:")
    print("-" * 30)
    for i in range(min(25, len(joints))):
        name = joint_names.get(i, f"joint_{i}")
        pos = joints[i]
        print(f"  {i:2d}: {name:15} = [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
    
    # Páteř analýza
    spine_joints = [3, 6, 9, 12]  # spine1, spine2, spine3, neck
    print(f"\nSPINE CHAIN:")
    print("-" * 30)
    for i, joint_idx in enumerate(spine_joints):
        pos = joints[joint_idx]
        name = joint_names[joint_idx]
        print(f"  {name:10} [{joint_idx:2d}]: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
    
    # Porovnání mezi framy
    print(f"\nSPINE MOVEMENT ACROSS FRAMES:")
    print("-" * 30)
    
    for frame_idx in [0, len(meshes)//2, -1]:  # První, střední, poslední
        if frame_idx < 0:
            frame_idx = len(meshes) - 1
            
        spine_pos = meshes[frame_idx]['joints'][12]  # neck joint
        print(f"  Frame {frame_idx:2d}: neck = [{spine_pos[0]:6.3f}, {spine_pos[1]:6.3f}, {spine_pos[2]:6.3f}]")
    
    # Celkové info
    print(f"\nDATA SUMMARY:")
    print("-" * 30)
    print(f"  Total frames: {len(meshes)}")
    print(f"  Joints per frame: {len(joints)}")
    print(f"  Vertices per frame: {len(meshes[0]['vertices'])}")
    print(f"  Available for biomechanics: ✅ YES")

if __name__ == "__main__":
    explore_joints("simple_results/test_meshes.pkl")
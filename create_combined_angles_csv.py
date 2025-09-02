#!/usr/bin/env python3
"""
Combined Angles CSV Creator
Creates single CSV with: frame, trunk_angle, neck_angle, left_arm_angle, right_arm_angle
Uses existing calculators to generate clean, simple output
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing calculators
from neck_angle_calculator_like_arm import calculate_neck_angle_to_trunk_like_arm
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES

def create_combined_angles_csv(pkl_file, output_csv="combined_angles.csv"):
    """
    Create simple CSV with all angles from PKL file
    
    Output columns:
    - frame: Frame number (0-based)
    - trunk_angle: Trunk angle in degrees
    - neck_angle: Neck angle in degrees  
    - left_arm_angle: Left arm angle in degrees
    - right_arm_angle: Right arm angle in degrees
    """
    
    print("CREATING COMBINED ANGLES CSV")
    print("=" * 50)
    
    # Load PKL data
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        return None
    
    print(f"Loading PKL file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"Loaded {len(meshes)} frames")
    
    # No separate trunk calculator needed - use direct calculation like export_arm_analysis_with_angles.py
    
    # Process all frames
    results = []
    print(f"\nProcessing {len(meshes)} frames...")
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']
        vertices = mesh_data['vertices']
        
        result = {
            'frame': frame_idx,
            'trunk_angle': 0.0,
            'neck_angle': 0.0,
            'left_arm_angle': 0.0,
            'right_arm_angle': 0.0
        }
        
        try:
            # 1. Calculate trunk angle - use same approach as export_arm_analysis_with_angles.py
            lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]
            cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]
            trunk_vector = cervical_joint - lumbar_joint
            trunk_length = np.linalg.norm(trunk_vector)
            
            if trunk_length > 0:
                spine_unit = trunk_vector / trunk_length
                vertical = np.array([0, 1, 0])  # Y-up coordinate system (SMPL-X)
                cos_angle = np.dot(spine_unit, vertical)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                trunk_angle_deg = np.degrees(np.arccos(cos_angle))
                result['trunk_angle'] = trunk_angle_deg
        except Exception as e:
            print(f"  Warning: Trunk angle failed for frame {frame_idx}: {e}")
        
        try:
            # 2. Calculate neck angle
            neck_result = calculate_neck_angle_to_trunk_like_arm(joints, vertices)
            if neck_result:
                result['neck_angle'] = neck_result['sagittal_angle']
        except Exception as e:
            print(f"  Warning: Neck angle failed for frame {frame_idx}: {e}")
        
        try:
            # 3. Calculate arm angles
            arm_result = calculate_bilateral_arm_angles(joints)
            if arm_result:
                result['left_arm_angle'] = arm_result['left_arm']['sagittal_angle']
                result['right_arm_angle'] = arm_result['right_arm']['sagittal_angle']
        except Exception as e:
            print(f"  Warning: Arm angles failed for frame {frame_idx}: {e}")
        
        results.append(result)
        
        # Progress output
        if frame_idx % 50 == 0 or frame_idx < 5:
            print(f"Frame {frame_idx:3d}: T={result['trunk_angle']:6.1f}° N={result['neck_angle']:6.1f}° L={result['left_arm_angle']:6.1f}° R={result['right_arm_angle']:6.1f}°")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"\nCOMBINED ANGLES CSV CREATED!")
    print(f"File: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Show statistics
    print(f"\nANGLE STATISTICS:")
    print(f"Trunk angle:     {df['trunk_angle'].mean():6.1f}° ± {df['trunk_angle'].std():5.1f}° (range: {df['trunk_angle'].min():6.1f}° to {df['trunk_angle'].max():6.1f}°)")
    print(f"Neck angle:      {df['neck_angle'].mean():6.1f}° ± {df['neck_angle'].std():5.1f}° (range: {df['neck_angle'].min():6.1f}° to {df['neck_angle'].max():6.1f}°)")
    print(f"Left arm angle:  {df['left_arm_angle'].mean():6.1f}° ± {df['left_arm_angle'].std():5.1f}° (range: {df['left_arm_angle'].min():6.1f}° to {df['left_arm_angle'].max():6.1f}°)")
    print(f"Right arm angle: {df['right_arm_angle'].mean():6.1f}° ± {df['right_arm_angle'].std():5.1f}° (range: {df['right_arm_angle'].min():6.1f}° to {df['right_arm_angle'].max():6.1f}°)")
    
    # Show first few rows
    print(f"\nFIRST 5 ROWS:")
    print(df.head().to_string(index=False))
    
    print(f"\nSUCCESS: Combined angles CSV ready for analysis!")
    return output_path

def main():
    """Main execution function"""
    
    # Default PKL file
    pkl_file = "fil_vid_meshes.pkl"
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Available PKL files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
        return
    
    # Create combined CSV
    output_file = create_combined_angles_csv(pkl_file, "combined_angles.csv")
    
    if output_file:
        print(f"\nREADY FOR ANALYSIS:")
        print(f"Import into Excel, Python pandas, or any analysis tool")
        print(f"File: {output_file}")

if __name__ == "__main__":
    main()
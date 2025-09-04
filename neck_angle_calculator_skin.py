#!/usr/bin/env python3
"""
Neck Angle Calculator using SKIN VERTICES
Uses vertex 2151 (cervical/neck on skin) to vertex 9002 (head on skin)
Calculates neck angle relative to trunk using same logic as arm calculator
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional

# SKIN VERTEX IDs
CERVICAL_SKIN_VERTEX = 2151  # Neck area on skin (same as trunk end point)
HEAD_VERTEX_ID = 9002         # Head vertex on skin (already used)
LUMBAR_SKIN_VERTEX = 5614    # Lower back for trunk reference

# Import joint indices for shoulder reference
from arm_angle_calculator import SMPL_X_JOINT_INDICES

def calculate_neck_angle_skin(vertices, joints):
    """
    Calculate neck angle using SKIN vertices
    Uses same logic as arm calculator for consistency
    
    Args:
        vertices: SMPL-X mesh vertices
        joints: SMPL-X joint positions (for shoulder reference)
        
    Returns:
        dict with sagittal_angle, confidence, components
    """
    
    # Get skin points
    cervical_skin = vertices[CERVICAL_SKIN_VERTEX]  # Base of neck on skin
    head_skin = vertices[HEAD_VERTEX_ID]             # Head on skin
    lumbar_skin = vertices[LUMBAR_SKIN_VERTEX]      # Lower back for trunk
    
    # Get shoulder positions for coordinate system
    left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
    right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
    
    # === BASIC VECTORS ===
    # Trunk vector from skin vertices
    trunk_vector = cervical_skin - lumbar_skin
    # Neck vector from skin vertices  
    neck_vector = head_skin - cervical_skin
    # Shoulder width
    shoulder_width_vector = left_shoulder - right_shoulder
    
    # === VALIDATE LENGTHS ===
    trunk_length = np.linalg.norm(trunk_vector)
    neck_length = np.linalg.norm(neck_vector)
    shoulder_width = np.linalg.norm(shoulder_width_vector)
    
    MIN_TRUNK_LENGTH = 0.05   # 5cm
    MIN_NECK_LENGTH = 0.03    # 3cm  
    MIN_SHOULDER_WIDTH = 0.08 # 8cm
    
    confidence = 1.0
    
    if trunk_length < MIN_TRUNK_LENGTH:
        return None
    if neck_length < MIN_NECK_LENGTH:
        return None
    if shoulder_width < MIN_SHOULDER_WIDTH:
        confidence *= 0.5
    
    # === ANATOMICAL COORDINATE SYSTEM (same as arm calculator) ===
    # Z-axis: "up" along trunk
    trunk_up = trunk_vector / trunk_length
    
    # Y-axis: "right" (left → right shoulder)
    shoulder_right = shoulder_width_vector / shoulder_width
    
    # X-axis: "forward" (cross product)
    body_forward_unnorm = np.cross(shoulder_right, trunk_up)
    body_forward_length = np.linalg.norm(body_forward_unnorm)
    
    if body_forward_length < 1e-8:
        confidence *= 0.3
        body_forward = np.array([1, 0, 0])
    else:
        body_forward = body_forward_unnorm / body_forward_length
    
    # Recalculate right axis for orthogonality
    shoulder_right = np.cross(trunk_up, body_forward)
    
    # === PROJECT NECK VECTOR ===
    neck_unit = neck_vector / neck_length
    
    # Project onto sagittal plane (forward-up plane)
    forward_component = np.dot(neck_unit, body_forward)
    up_component = np.dot(neck_unit, trunk_up)
    
    # Sagittal angle (in forward-up plane)
    sagittal_angle_rad = np.arctan2(forward_component, up_component)
    sagittal_angle_deg = np.degrees(sagittal_angle_rad)
    
    # Project onto frontal plane (right-up plane)
    right_component = np.dot(neck_unit, shoulder_right)
    frontal_angle_rad = np.arctan2(right_component, up_component)
    frontal_angle_deg = np.degrees(frontal_angle_rad)
    
    return {
        'sagittal_angle': sagittal_angle_deg,
        'frontal_angle': frontal_angle_deg,
        'confidence': confidence,
        'neck_length': neck_length,
        'forward_component': forward_component,
        'up_component': up_component,
        'right_component': right_component,
        'cervical_pos': cervical_skin.tolist(),
        'head_pos': head_skin.tolist()
    }

class NeckAngleCalculatorSkin:
    """Calculate neck angles using skin surface vertices"""
    
    def __init__(self):
        print("NeckAngleCalculatorSkin initialized")
        print(f"Using SKIN SURFACE vertices:")
        print(f"  Cervical (neck base): Vertex {CERVICAL_SKIN_VERTEX}")
        print(f"  Head: Vertex {HEAD_VERTEX_ID}")
        print(f"  Lumbar (trunk reference): Vertex {LUMBAR_SKIN_VERTEX}")
        print("All measurements from actual skin surface!")
    
    def calculate_angles_from_pkl(self, pkl_path: str) -> pd.DataFrame:
        """Calculate neck angles for all frames"""
        
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"\nLoading PKL file: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            meshes = pickle.load(f)
        
        print(f"Loaded {len(meshes)} frames")
        
        # Verify vertices
        if meshes:
            first_frame = meshes[0]
            num_vertices = len(first_frame['vertices'])
            print(f"Mesh has {num_vertices} vertices")
            
            for vertex_id, name in [(CERVICAL_SKIN_VERTEX, "Cervical"), 
                                   (HEAD_VERTEX_ID, "Head"),
                                   (LUMBAR_SKIN_VERTEX, "Lumbar")]:
                if vertex_id >= num_vertices:
                    print(f"WARNING: {name} vertex {vertex_id} out of range!")
        
        print(f"\nCalculating neck angles using skin vertices...")
        
        results = []
        
        for frame_idx, mesh_data in enumerate(meshes):
            vertices = mesh_data['vertices']
            joints = mesh_data['joints']
            
            angle_data = calculate_neck_angle_skin(vertices, joints)
            
            if angle_data:
                results.append({
                    'frame': frame_idx,
                    'neck_sagittal_angle': angle_data['sagittal_angle'],
                    'neck_frontal_angle': angle_data['frontal_angle'],
                    'neck_length': angle_data['neck_length'],
                    'confidence': angle_data['confidence'],
                    'forward_component': angle_data['forward_component'],
                    'cervical_x': angle_data['cervical_pos'][0],
                    'cervical_y': angle_data['cervical_pos'][1],
                    'cervical_z': angle_data['cervical_pos'][2],
                    'head_x': angle_data['head_pos'][0],
                    'head_y': angle_data['head_pos'][1],
                    'head_z': angle_data['head_pos'][2]
                })
            else:
                results.append({
                    'frame': frame_idx,
                    'neck_sagittal_angle': 0,
                    'neck_frontal_angle': 0,
                    'neck_length': 0,
                    'confidence': 0
                })
            
            if frame_idx % 50 == 0 or frame_idx < 5:
                if angle_data:
                    print(f"Frame {frame_idx:3d}: Sagittal = {angle_data['sagittal_angle']:6.1f}°, "
                          f"Frontal = {angle_data['frontal_angle']:6.1f}°")
        
        df = pd.DataFrame(results)
        
        print(f"\nProcessed {len(df)} frames")
        print(f"Average neck sagittal angle: {df['neck_sagittal_angle'].mean():.1f}°")
        print(f"Range: {df['neck_sagittal_angle'].min():.1f}° to {df['neck_sagittal_angle'].max():.1f}°")
        
        return df
    
    def export_statistics(self, df: pd.DataFrame, output_file: str = "neck_angle_skin_statistics.txt"):
        """Export detailed statistics"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NECK ANGLE STATISTICS (SKIN-BASED)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Using skin vertices:\n")
            f.write(f"  Cervical (neck base): {CERVICAL_SKIN_VERTEX}\n")
            f.write(f"  Head: {HEAD_VERTEX_ID}\n")
            f.write(f"  Lumbar (trunk ref): {LUMBAR_SKIN_VERTEX}\n\n")
            f.write(f"Total frames analyzed: {len(df)}\n\n")
            
            f.write("SAGITTAL ANGLE STATISTICS (forward/back):\n")
            f.write(f"  Mean: {df['neck_sagittal_angle'].mean():.2f}°\n")
            f.write(f"  Std Dev: {df['neck_sagittal_angle'].std():.2f}°\n")
            f.write(f"  Min: {df['neck_sagittal_angle'].min():.2f}°\n")
            f.write(f"  Max: {df['neck_sagittal_angle'].max():.2f}°\n")
            f.write(f"  Median: {df['neck_sagittal_angle'].median():.2f}°\n\n")
            
            f.write("FRONTAL ANGLE STATISTICS (side tilt):\n")
            f.write(f"  Mean: {df['neck_frontal_angle'].mean():.2f}°\n")
            f.write(f"  Std Dev: {df['neck_frontal_angle'].std():.2f}°\n")
            f.write(f"  Min: {df['neck_frontal_angle'].min():.2f}°\n")
            f.write(f"  Max: {df['neck_frontal_angle'].max():.2f}°\n\n")
            
            # Posture classification
            f.write("NECK POSTURE DISTRIBUTION:\n")
            neutral = len(df[(df['neck_sagittal_angle'] >= -15) & (df['neck_sagittal_angle'] <= 15)])
            forward = len(df[df['neck_sagittal_angle'] > 15])
            backward = len(df[df['neck_sagittal_angle'] < -15])
            
            total = len(df)
            f.write(f"  Neutral (-15° to 15°): {neutral} frames ({100*neutral/total:.1f}%)\n")
            f.write(f"  Forward (>15°): {forward} frames ({100*forward/total:.1f}%)\n")
            f.write(f"  Backward (<-15°): {backward} frames ({100*backward/total:.1f}%)\n")
        
        print(f"\nStatistics saved to: {output_file}")
    
    def save_to_csv(self, df: pd.DataFrame, output_file: str = "neck_angles_skin.csv"):
        """Save angles to CSV"""
        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")

def main():
    """Main execution"""
    import sys
    
    pkl_file = "arm_meshes.pkl"
    
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Usage: python neck_angle_calculator_skin.py [pkl_file]")
        return
    
    print("\nSKIN-BASED NECK ANGLE CALCULATOR")
    print("=" * 50)
    
    calculator = NeckAngleCalculatorSkin()
    
    # Calculate angles
    df = calculator.calculate_angles_from_pkl(pkl_file)
    
    # Export results
    calculator.export_statistics(df, "neck_angle_skin_statistics.txt")
    calculator.save_to_csv(df, "neck_angles_skin.csv")
    
    print("\n" + "=" * 50)
    print("SKIN-BASED NECK ANALYSIS COMPLETE!")
    print("Files created:")
    print("  - neck_angle_skin_statistics.txt")
    print("  - neck_angles_skin.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()
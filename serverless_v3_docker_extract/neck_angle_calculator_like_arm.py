#!/usr/bin/env python3
"""
Neck Angle Calculator using EXACT SAME LOGIC as arm_angle_calculator.py
This should fix the frame instability by using proven stable algorithm
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# HARDCODED HEAD VERTEX
HEAD_VERTEX_ID = 9002

# SMPL-X joint indices (same as arm calculator)
SMPL_X_JOINT_INDICES = {
    'pelvis': 0,
    'spine1': 3,          # Lumbar (L3/L4) 
    'spine2': 6,          # Mid spine
    'spine3': 9,          # Upper spine
    'neck': 12,           # Cervical (C7/T1)
    'head': 15,
    'left_shoulder': 17,  # Left shoulder joint
    'right_shoulder': 16, # Right shoulder joint
}

def calculate_neck_angle_to_trunk_like_arm(joints, vertices):
    """
    Calculate neck angle using EXACT SAME LOGIC as arm calculator
    This should be stable like arm angles
    
    Args:
        joints: SMPL-X joint positions (117, 3)
        vertices: SMPL-X mesh vertices
        
    Returns:
        dict with sagittal_angle, confidence, components (same as arm calculator)
    """
    
    # === ANATOMICKÉ BODY === (same as arm calculator)
    lumbar = joints[SMPL_X_JOINT_INDICES['spine1']]      # L3/L4
    cervical = joints[SMPL_X_JOINT_INDICES['neck']]       # C7/T1
    left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
    right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
    
    # NECK specific: use head vertex instead of elbow
    head_vertex = vertices[HEAD_VERTEX_ID]
    
    # === ZÁKLADNÍ VEKTORY === (same as arm calculator)
    trunk_vector = cervical - lumbar
    neck_vector = head_vertex - cervical  # This replaces arm_vector = elbow - shoulder
    shoulder_width_vector = left_shoulder - right_shoulder  # FIXED: left->right direction (same as arm calc)
    
    # === VALIDACE DÉLKY VEKTORŮ === (same as arm calculator)
    trunk_length = np.linalg.norm(trunk_vector)
    neck_length = np.linalg.norm(neck_vector)  # replaces arm_length
    shoulder_width = np.linalg.norm(shoulder_width_vector)
    
    # Minimální prahové hodnoty (same as arm calculator)
    MIN_TRUNK_LENGTH = 0.05   # 5cm
    MIN_NECK_LENGTH = 0.03    # 3cm (replaces MIN_ARM_LENGTH)
    MIN_SHOULDER_WIDTH = 0.08 # 8cm
    
    confidence = 1.0
    
    if trunk_length < MIN_TRUNK_LENGTH:
        return None  # Nevalidní trunk
    if neck_length < MIN_NECK_LENGTH:
        return None  # Neck příliš krátký
    if shoulder_width < MIN_SHOULDER_WIDTH:
        confidence *= 0.5  # Snížená spolehlivost
    
    # === ANATOMICKÝ KOORDINÁTNÍ SYSTÉM === (EXACT SAME as arm calculator)
    # Z-axis: "nahoru" podél trupu
    trunk_up = trunk_vector / trunk_length
    
    # Y-axis: "doprava" (levé → pravé rameno)
    shoulder_right = shoulder_width_vector / shoulder_width
    
    # X-axis: "dopředu" (cross produkt)
    body_forward_unnorm = np.cross(shoulder_right, trunk_up)
    body_forward_length = np.linalg.norm(body_forward_unnorm)
    
    if body_forward_length < 1e-8:
        confidence *= 0.3  # Ramena rovnoběžná s trupem
        # Fallback: použij jiný směr
        body_forward = np.array([1, 0, 0])  # defaultní forward
    else:
        body_forward = body_forward_unnorm / body_forward_length
    
    # Rekalkulace right axis pro ortogonalitu
    shoulder_right = np.cross(trunk_up, body_forward) 
    
    # === TRANSFORMACE NECK VEKTORU === (same logic as arm calculator)
    neck_norm = neck_vector / neck_length
    
    # Projekce na anatomické osy (same as arm calculator)
    neck_forward_comp = np.dot(neck_norm, body_forward)    # dopředu/dozadu
    neck_up_comp = np.dot(neck_norm, trunk_up)            # nahoru/dolů  
    neck_right_comp = np.dot(neck_norm, shoulder_right)    # doprava/doleva
    
    # === SAGITÁLNÍ ÚHEL === (EXACT SAME LOGIC as arm calculator)
    # We want: 0° = podél trupu (up), +90° = forward, -90° = backward
    # neck_up_comp: positive when neck points up along trunk
    # neck_forward_comp: positive when neck points forward
    
    # Use atan2 to get angle from "up along trunk" direction (like arm calculator)
    # When neck aligned with trunk: neck_up_comp = 1, neck_forward_comp = 0 -> angle = 0°
    # When neck forward: neck_up_comp = 0, neck_forward_comp = 1 -> angle = 90°
    # When neck backward: neck_up_comp = 0, neck_forward_comp = -1 -> angle = -90°
    
    sagittal_angle_rad = np.arctan2(neck_forward_comp, neck_up_comp)  # SAME as arm calculator logic
    sagittal_angle_deg = np.degrees(sagittal_angle_rad)
    
    # === CONFIDENCE SCORING === (same as arm calculator)
    # Penalizace za extrémní úhly nebo nestabilitu
    if abs(sagittal_angle_deg) > 135:  # Velmi extrémní pozice
        confidence *= 0.7
    
    # Penalizace za degenerovaný koordinátní systém
    if body_forward_length < 1e-8:
        confidence *= 0.3
        
    # Length-based confidence (same as arm calculator)
    length_factor = min(neck_length / 0.15, trunk_length / 0.3, 1.0)
    confidence *= length_factor
    
    return {
        'sagittal_angle': sagittal_angle_deg,  # Main result (same name as arm calculator)
        'sagittal_angle_rad': sagittal_angle_rad,
        'confidence': max(0.0, min(1.0, confidence)),
        
        # Components (same as arm calculator)
        'neck_forward_comp': neck_forward_comp,
        'neck_up_comp': neck_up_comp,
        'neck_right_comp': neck_right_comp,
        
        # Vectors and lengths
        'trunk_vector': trunk_vector,
        'neck_vector': neck_vector,
        'trunk_length': trunk_length,
        'neck_length': neck_length,
        'shoulder_width': shoulder_width,
        
        # Coordinate system
        'body_forward': body_forward,
        'trunk_up': trunk_up,
        'shoulder_right': shoulder_right,
        'coordinate_system_quality': body_forward_length,
        
        # Classification (based on angle)
        'is_flexion': sagittal_angle_deg > 15.0,
        'is_extension': sagittal_angle_deg < -15.0,
    }

class NeckAngleCalculatorLikeArm:
    """Neck angle calculator using exact same logic as arm calculator"""
    
    def __init__(self):
        """Initialize calculator using arm calculator approach"""
        
        self.lumbar_joint = SMPL_X_JOINT_INDICES['spine1']
        self.cervical_joint = SMPL_X_JOINT_INDICES['neck']
        self.head_vertex_id = HEAD_VERTEX_ID
        
        print("NeckAngleCalculatorLikeArm initialized")
        print("Uses EXACT SAME LOGIC as arm_angle_calculator.py")
        print("This should fix frame instability")
    
    def load_pkl_data(self, pkl_path: str) -> List[Dict]:
        """Load mesh data from PKL file"""
        
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"\nLoading PKL file: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            meshes = pickle.load(f)
        
        print(f"Loaded {len(meshes)} frames")
        return meshes
    
    def process_all_frames(self, meshes: List[Dict]) -> pd.DataFrame:
        """Process all frames using arm calculator logic"""
        
        results = []
        
        print(f"\nProcessing {len(meshes)} frames with arm calculator logic...")
        
        for frame_idx, mesh_data in enumerate(meshes):
            joints = mesh_data['joints']
            vertices = mesh_data['vertices']
            
            # Calculate neck angle using arm calculator logic
            neck_result = calculate_neck_angle_to_trunk_like_arm(joints, vertices)
            
            if neck_result is None:
                # Invalid frame
                result = {
                    'frame': frame_idx,
                    'time_sec': frame_idx / 30.0,
                    'neck_angle': 0.0,
                    'confidence': 0.0,
                    'is_flexion': False,
                    'is_extension': False,
                    'error': 'Invalid measurements'
                }
            else:
                # Store result
                result = {
                    'frame': frame_idx,
                    'time_sec': frame_idx / 30.0,
                    'neck_angle': neck_result['sagittal_angle'],
                    'confidence': neck_result['confidence'],
                    'trunk_length': neck_result['trunk_length'],
                    'neck_length': neck_result['neck_length'],
                    'shoulder_width': neck_result['shoulder_width'],
                    'is_flexion': neck_result['is_flexion'],
                    'is_extension': neck_result['is_extension'],
                    'neck_forward_component': neck_result['neck_forward_comp'],
                    'neck_up_component': neck_result['neck_up_comp'],
                    'coordinate_system_quality': neck_result['coordinate_system_quality'],
                    'error': ''
                }
            
            results.append(result)
            
            # Progress output
            if frame_idx % 50 == 0 or frame_idx < 5:
                if neck_result is None:
                    print(f"Frame {frame_idx:3d}: ERROR - invalid measurements")
                else:
                    classification = "flexion" if neck_result['is_flexion'] else ("extension" if neck_result['is_extension'] else "neutral")
                    print(f"Frame {frame_idx:3d}: Neck={neck_result['sagittal_angle']:6.1f}° ({classification}, conf={neck_result['confidence']:.2f})")
        
        return pd.DataFrame(results)
    
    def export_results(self, df: pd.DataFrame, output_path: str):
        """Export results to CSV and JSON"""
        
        output_path = Path(output_path)
        
        # CSV export
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to CSV: {csv_path}")
        
        # Summary statistics
        summary = {
            'algorithm': 'neck_like_arm_calculator',
            'total_frames': len(df),
            'duration_sec': df['time_sec'].max(),
            'neck_angle_stats': {
                'mean': float(df['neck_angle'].mean()),
                'std': float(df['neck_angle'].std()),
                'min': float(df['neck_angle'].min()),
                'max': float(df['neck_angle'].max()),
                'median': float(df['neck_angle'].median())
            },
            'confidence_stats': {
                'mean': float(df['confidence'].mean()),
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max())
            },
            'movement_classification': {
                'flexion_frames': int(df['is_flexion'].sum()),
                'extension_frames': int(df['is_extension'].sum()),
                'neutral_frames': int((~df['is_flexion'] & ~df['is_extension']).sum())
            },
            'error_frames': int(df['error'].apply(lambda x: len(x) > 0).sum())
        }
        
        # JSON export
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary exported to JSON: {json_path}")
        
        # Print summary
        print(f"\nNECK ANGLE ANALYSIS (ARM CALCULATOR LOGIC):")
        print(f"=" * 60)
        print(f"Algorithm: Same as arm_angle_calculator.py")
        print(f"Total frames: {summary['total_frames']}")
        print(f"Duration: {summary['duration_sec']:.1f} seconds")
        print(f"Error frames: {summary['error_frames']}")
        
        print(f"\nNeck angle statistics:")
        print(f"  Mean: {summary['neck_angle_stats']['mean']:.1f}° ± {summary['neck_angle_stats']['std']:.1f}°")
        print(f"  Range: {summary['neck_angle_stats']['min']:.1f}° to {summary['neck_angle_stats']['max']:.1f}°")
        print(f"  Median: {summary['neck_angle_stats']['median']:.1f}°")
        
        print(f"\nMovement classification:")
        print(f"  Flexion frames: {summary['movement_classification']['flexion_frames']}")
        print(f"  Extension frames: {summary['movement_classification']['extension_frames']}")
        print(f"  Neutral frames: {summary['movement_classification']['neutral_frames']}")
        
        print(f"\nMean confidence: {summary['confidence_stats']['mean']:.3f}")
        
        # Test problematic frames
        frame_184_angle = df[df['frame'] == 184]['neck_angle'].iloc[0]
        frame_185_angle = df[df['frame'] == 185]['neck_angle'].iloc[0]
        frame_183_angle = df[df['frame'] == 183]['neck_angle'].iloc[0]
        
        print(f"\nPROBLEMATIC FRAMES TEST:")
        print(f"Frame 183: {frame_183_angle:.1f}°")
        print(f"Frame 184: {frame_184_angle:.1f}°")
        print(f"Frame 185: {frame_185_angle:.1f}°")
        print(f"Diff 183-184: {abs(frame_184_angle - frame_183_angle):.1f}°")
        print(f"Diff 184-185: {abs(frame_185_angle - frame_184_angle):.1f}°")
        
        if abs(frame_184_angle - frame_183_angle) < 5.0 and abs(frame_185_angle - frame_184_angle) < 5.0:
            print("SUCCESS: Frame stability achieved using arm calculator logic!")
        else:
            print("WARNING: Still unstable - may need further investigation")
        
        return summary

def main():
    """Main execution function using arm calculator logic"""
    
    print("NECK ANGLE CALCULATOR USING ARM CALCULATOR LOGIC")
    print("=" * 70)
    print("Uses EXACT SAME ALGORITHM as arm_angle_calculator.py:")
    print("+ Same coordinate system calculation")
    print("+ Same atan2() angle calculation (not arccos)")
    print("+ Same confidence scoring")
    print("+ Should be stable like arm angles")
    print()
    
    # Initialize calculator
    calculator = NeckAngleCalculatorLikeArm()
    
    # Process data
    pkl_file = "arm_meshes.pkl"
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        return
    
    # Load and process
    meshes = calculator.load_pkl_data(pkl_file)
    results_df = calculator.process_all_frames(meshes)
    
    # Export results
    output_base = "neck_angle_analysis_LIKE_ARM"
    summary = calculator.export_results(results_df, output_base)
    
    print(f"\nNECK ANGLE ANALYSIS (ARM LOGIC) COMPLETE!")
    print(f"Used hardcoded head vertex: {HEAD_VERTEX_ID}")
    print(f"Algorithm: Exact copy of arm_angle_calculator.py logic")

if __name__ == "__main__":
    main()
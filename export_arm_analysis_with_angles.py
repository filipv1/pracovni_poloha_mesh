#!/usr/bin/env python3
"""
Export arm analysis with calculated angles to visualization
Combines arm vectors, trunk vectors, and angle calculations
Creates enhanced side-by-side visualization with angle data
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES

def create_enhanced_arm_analysis_export(pkl_file, output_dir):
    """
    Create enhanced arm analysis with angle calculations
    Exports OBJ files with arm vectors + angle statistics
    """
    
    print(f"CREATING ENHANCED ARM ANALYSIS WITH ANGLES")
    print(f"Input: {pkl_file}")
    print("=" * 60)
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Import required functions from existing visualizer
    from visualize_arm_vectors_with_trunk import (
        create_arrow_mesh, 
        create_angle_arc, 
        calculate_trunk_angle_to_gravity
    )
    
    angle_data_list = []
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']  # Shape: (117, 3)
        
        # Calculate all angles
        bilateral_result = calculate_bilateral_arm_angles(joints)
        
        if not bilateral_result['left_arm'] or not bilateral_result['right_arm']:
            print(f"Frame {frame_idx:3d}: SKIPPED - Invalid angle calculation")
            continue
        
        # Extract joint positions
        lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]
        cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Calculate vectors
        trunk_vector = cervical_joint - lumbar_joint
        left_arm_vector = left_elbow - left_shoulder
        right_arm_vector = right_elbow - right_shoulder
        trunk_length = np.linalg.norm(trunk_vector)
        
        # Get angle data
        left_arm_data = bilateral_result['left_arm']
        right_arm_data = bilateral_result['right_arm']
        
        # Calculate trunk angle to gravity using CORRECT implementation
        # Use same logic as trunk_angle_calculator.py
        spine_unit = trunk_vector / trunk_length
        vertical = np.array([0, 1, 0])  # Y-up coordinate system (SMPL-X)
        cos_angle = np.dot(spine_unit, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        trunk_angle_deg = np.degrees(np.arccos(cos_angle))
        
        # Store angle data for statistics
        angle_data_list.append({
            'frame': frame_idx,
            'trunk_angle': trunk_angle_deg,
            'left_sagittal': left_arm_data['sagittal_angle'],
            'left_frontal': left_arm_data['frontal_angle'],
            'left_confidence': left_arm_data['confidence'],
            'right_sagittal': right_arm_data['sagittal_angle'],
            'right_frontal': right_arm_data['frontal_angle'],
            'right_confidence': right_arm_data['confidence'],
            'trunk_length': trunk_length,
            'left_arm_length': np.linalg.norm(left_arm_vector),
            'right_arm_length': np.linalg.norm(right_arm_vector),
        })
        
        # Create combined mesh with all vectors and angles
        all_vertices = []
        all_faces = []
        current_vertex_offset = 0
        
        # 1. Trunk vector arrow (RED)
        trunk_vertices, trunk_faces = create_arrow_mesh(lumbar_joint, cervical_joint, arrow_radius=0.012)
        all_vertices.extend(trunk_vertices)
        all_faces.extend(trunk_faces + current_vertex_offset)
        current_vertex_offset += len(trunk_vertices)
        
        # 2. Gravitational reference vector (BLUE)
        gravity_end = lumbar_joint + np.array([0, 0, -trunk_length])
        gravity_vertices, gravity_faces = create_arrow_mesh(lumbar_joint, gravity_end, arrow_radius=0.008)
        all_vertices.extend(gravity_vertices)
        all_faces.extend(gravity_faces + current_vertex_offset)
        current_vertex_offset += len(gravity_vertices)
        
        # 3. Trunk angle arc (GREEN)
        trunk_arc_vertices, trunk_arc_faces = create_angle_arc(lumbar_joint, trunk_vector, np.array([0, 0, -trunk_length]), radius=0.08)
        if len(trunk_arc_vertices) > 0:
            all_vertices.extend(trunk_arc_vertices)
            all_faces.extend(trunk_arc_faces + current_vertex_offset)
            current_vertex_offset += len(trunk_arc_vertices)
        
        # 4. Left arm vector (YELLOW) 
        left_arm_vertices, left_arm_faces = create_arrow_mesh(left_shoulder, left_elbow, arrow_radius=0.010)
        all_vertices.extend(left_arm_vertices)
        all_faces.extend(left_arm_faces + current_vertex_offset)
        current_vertex_offset += len(left_arm_vertices)
        
        # 5. Right arm vector (ORANGE)
        right_arm_vertices, right_arm_faces = create_arrow_mesh(right_shoulder, right_elbow, arrow_radius=0.010)
        all_vertices.extend(right_arm_vertices)
        all_faces.extend(right_arm_faces + current_vertex_offset)
        current_vertex_offset += len(right_arm_vertices)
        
        # 6. Left arm angle arc relative to trunk (CYAN) - smaller radius
        left_trunk_down = -trunk_vector / np.linalg.norm(trunk_vector)  # Down along trunk
        left_arm_arc_vertices, left_arm_arc_faces = create_angle_arc(
            left_shoulder, left_trunk_down * np.linalg.norm(left_arm_vector), left_arm_vector, radius=0.05
        )
        if len(left_arm_arc_vertices) > 0:
            all_vertices.extend(left_arm_arc_vertices)
            all_faces.extend(left_arm_arc_faces + current_vertex_offset)
            current_vertex_offset += len(left_arm_arc_vertices)
        
        # 7. Right arm angle arc relative to trunk (MAGENTA) - smaller radius  
        right_arm_arc_vertices, right_arm_arc_faces = create_angle_arc(
            right_shoulder, left_trunk_down * np.linalg.norm(right_arm_vector), right_arm_vector, radius=0.05
        )
        if len(right_arm_arc_vertices) > 0:
            all_vertices.extend(right_arm_arc_vertices)
            all_faces.extend(right_arm_arc_faces + current_vertex_offset)
        
        # Export OBJ file with enhanced header
        obj_file = output_dir / f"enhanced_arm_analysis_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w') as f:
            # Enhanced header with all angle data
            f.write(f"# Enhanced Arm + Trunk Analysis Frame {frame_idx}\n")
            f.write(f"# TRUNK DATA:\n")
            f.write(f"#   Lumbar: [{lumbar_joint[0]:.3f}, {lumbar_joint[1]:.3f}, {lumbar_joint[2]:.3f}]\n")
            f.write(f"#   Cervical: [{cervical_joint[0]:.3f}, {cervical_joint[1]:.3f}, {cervical_joint[2]:.3f}]\n")
            f.write(f"#   Trunk Angle to Gravity: {trunk_angle_deg:.1f}°\n")
            f.write(f"#   Trunk Length: {trunk_length:.3f}m\n")
            f.write(f"# LEFT ARM DATA:\n")
            f.write(f"#   Shoulder: [{left_shoulder[0]:.3f}, {left_shoulder[1]:.3f}, {left_shoulder[2]:.3f}]\n")
            f.write(f"#   Elbow: [{left_elbow[0]:.3f}, {left_elbow[1]:.3f}, {left_elbow[2]:.3f}]\n")
            f.write(f"#   Sagittal Angle: {left_arm_data['sagittal_angle']:.1f}° (0°=down, +90°=forward, -90°=back)\n")
            f.write(f"#   Frontal Angle: {left_arm_data['frontal_angle']:.1f}° (0°=down, +90°=out)\n")
            f.write(f"#   Confidence: {left_arm_data['confidence']:.3f}\n")
            f.write(f"# RIGHT ARM DATA:\n")
            f.write(f"#   Shoulder: [{right_shoulder[0]:.3f}, {right_shoulder[1]:.3f}, {right_shoulder[2]:.3f}]\n")
            f.write(f"#   Elbow: [{right_elbow[0]:.3f}, {right_elbow[1]:.3f}, {right_elbow[2]:.3f}]\n")
            f.write(f"#   Sagittal Angle: {right_arm_data['sagittal_angle']:.1f}° (0°=down, +90°=forward, -90°=back)\n")
            f.write(f"#   Frontal Angle: {right_arm_data['frontal_angle']:.1f}° (0°=down, +90°=out)\n")
            f.write(f"#   Confidence: {right_arm_data['confidence']:.3f}\n")
            f.write(f"# COLOR CODING: RED=trunk, BLUE=gravity, GREEN=trunk_angle, YELLOW=left_arm, ORANGE=right_arm, CYAN=left_angle, MAGENTA=right_angle\n\n")
            
            # Write all vertices
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces with group labels
            vertex_groups = [
                (trunk_faces, "TrunkVector"),
                (gravity_faces, "GravityReference"),
                (trunk_arc_faces if len(trunk_arc_vertices) > 0 else [], "TrunkAngleArc"),
                (left_arm_faces, "LeftArmVector"),
                (right_arm_faces, "RightArmVector"),
                (left_arm_arc_faces if len(left_arm_arc_vertices) > 0 else [], "LeftArmAngleArc"),
                (right_arm_arc_faces if len(right_arm_arc_vertices) > 0 else [], "RightArmAngleArc"),
            ]
            
            face_offset = 1  # OBJ files are 1-indexed
            for faces, group_name in vertex_groups:
                if len(faces) > 0:
                    f.write(f"g {group_name}\n")
                    for face in faces:
                        f.write(f"f {face[0]+face_offset} {face[1]+face_offset} {face[2]+face_offset}\n")
                    face_offset += len([v for v_group, g in zip([trunk_vertices, gravity_vertices, trunk_arc_vertices, left_arm_vertices, right_arm_vertices, left_arm_arc_vertices, right_arm_arc_vertices], [g[1] for g in vertex_groups]) if g == group_name for v in v_group])
        
        # Progress output
        if frame_idx % 1 == 0:
            print(f"Frame {frame_idx:3d}: Trunk={trunk_angle_deg:5.1f}°, "
                  f"L_arm={left_arm_data['sagittal_angle']:6.1f}°, "
                  f"R_arm={right_arm_data['sagittal_angle']:6.1f}°")
    
    # Export comprehensive angle statistics
    if angle_data_list:
        df = pd.DataFrame(angle_data_list)
        
        # CSV export
        csv_file = output_dir / "enhanced_arm_analysis_complete.csv"
        df.to_csv(csv_file, index=False)
        
        # Enhanced statistics
        stats_file = output_dir / "enhanced_arm_analysis_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED ARM + TRUNK ANALYSIS STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"DATA SUMMARY:\n")
            f.write(f"  Processed frames: {len(df)}\n")
            f.write(f"  Total duration: {len(df)} frames\n\n")
            
            # Trunk statistics
            f.write(f"TRUNK ANGLE STATISTICS (relative to gravity):\n")
            f.write(f"  Mean: {df['trunk_angle'].mean():.1f}°\n")
            f.write(f"  Std Dev: {df['trunk_angle'].std():.1f}°\n")
            f.write(f"  Range: {df['trunk_angle'].min():.1f}° to {df['trunk_angle'].max():.1f}°\n\n")
            
            # Left arm statistics  
            f.write(f"LEFT ARM SAGITTAL ANGLE STATISTICS (relative to trunk):\n")
            f.write(f"  Mean: {df['left_sagittal'].mean():.1f}°\n")
            f.write(f"  Std Dev: {df['left_sagittal'].std():.1f}°\n")
            f.write(f"  Range: {df['left_sagittal'].min():.1f}° to {df['left_sagittal'].max():.1f}°\n")
            f.write(f"  Mean Confidence: {df['left_confidence'].mean():.3f}\n\n")
            
            # Right arm statistics
            f.write(f"RIGHT ARM SAGITTAL ANGLE STATISTICS (relative to trunk):\n")
            f.write(f"  Mean: {df['right_sagittal'].mean():.1f}°\n")
            f.write(f"  Std Dev: {df['right_sagittal'].std():.1f}°\n")
            f.write(f"  Range: {df['right_sagittal'].min():.1f}° to {df['right_sagittal'].max():.1f}°\n")
            f.write(f"  Mean Confidence: {df['right_confidence'].mean():.3f}\n\n")
            
            # Movement analysis
            f.write(f"MOVEMENT ANALYSIS:\n")
            f.write(f"  Left arm forward flexion (>45°): {(df['left_sagittal'] > 45).sum()} frames\n")
            f.write(f"  Left arm backward extension (<-45°): {(df['left_sagittal'] < -45).sum()} frames\n")
            f.write(f"  Right arm forward flexion (>45°): {(df['right_sagittal'] > 45).sum()} frames\n")
            f.write(f"  Right arm backward extension (<-45°): {(df['right_sagittal'] < -45).sum()} frames\n\n")
            
            f.write(f"INTERPRETATION:\n")
            f.write(f"  Sagittal angles: 0°=arms hanging, +90°=forward, -90°=backward\n")
            f.write(f"  All angles calculated relative to trunk orientation\n")
            f.write(f"  Confidence >0.8 indicates high reliability\n")
        
        print(f"\nENHANCED ANALYSIS COMPLETE!")
        print(f"  Exported {len(angle_data_list)} enhanced frames")
        print(f"  Files saved to: {output_dir}")
        print(f"  CSV data: {csv_file.name}")
        print(f"  Statistics: {stats_file.name}")
        print(f"\nARMLEAL ANGLE SUMMARY:")
        print(f"  Left arm mean: {df['left_sagittal'].mean():.1f}° ± {df['left_sagittal'].std():.1f}°")
        print(f"  Right arm mean: {df['right_sagittal'].mean():.1f}° ± {df['right_sagittal'].std():.1f}°")
    
    return output_dir

if __name__ == "__main__":
    # Process with existing PKL file
    pkl_file = "arm_meshes.pkl"
    output_dir = "enhanced_arm_analysis_export"
    
    if Path(pkl_file).exists():
        result_dir = create_enhanced_arm_analysis_export(pkl_file, output_dir)
        print(f"\nReady for Blender visualization!")
        print(f"Use existing side_by_side_arm_and_trunk_sequence.py with: {result_dir}")
    else:
        print(f"PKL file not found: {pkl_file}")
        available_files = list(Path(".").glob("*.pkl"))
        if available_files:
            print("Available PKL files:")
            for pkl in available_files:
                print(f"  - {pkl}")
        else:
            print("No PKL files found in current directory")
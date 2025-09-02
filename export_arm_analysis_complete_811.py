#!/usr/bin/env python3
"""
Export complete arm analysis with 811 frames
Uses interpolation for frames with invalid angle calculations
Maintains high precision while ensuring complete dataset
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES
from scipy.interpolate import interp1d

def interpolate_missing_angles(angle_data_list, total_frames):
    """
    Interpolate missing angle data for complete 811-frame sequence
    """
    
    print(f"INTERPOLATING MISSING DATA")
    print(f"Valid frames: {len(angle_data_list)}")
    print(f"Target frames: {total_frames}")
    print("=" * 40)
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(angle_data_list)
    
    # Create complete frame index
    complete_frames = list(range(total_frames))
    
    # Get existing valid frame indices
    valid_frames = df['frame'].tolist()
    missing_frames = [f for f in complete_frames if f not in valid_frames]
    
    print(f"Missing frames: {len(missing_frames)}")
    print(f"Will interpolate: trunk_angle, left_arm_angle, right_arm_angle")
    
    # Prepare interpolation data
    interpolation_columns = ['trunk_angle', 'left_arm_angle', 'right_arm_angle', 
                           'left_arm_confidence', 'right_arm_confidence']
    
    interpolated_data = []
    
    for frame_idx in complete_frames:
        if frame_idx in valid_frames:
            # Use existing valid data
            frame_data = df[df['frame'] == frame_idx].iloc[0].to_dict()
            frame_data['interpolated'] = False
        else:
            # Interpolate missing data
            frame_data = {
                'frame': frame_idx,
                'interpolated': True
            }
            
            # For each column, interpolate from valid data
            for col in interpolation_columns:
                if len(valid_frames) >= 2:  # Need at least 2 points for interpolation
                    # Get valid values for this column
                    valid_values = df[col].values
                    
                    # Create interpolation function
                    if len(valid_frames) == 2:
                        # Linear interpolation between two points
                        interp_func = interp1d(valid_frames, valid_values, 
                                             kind='linear', fill_value='extrapolate')
                    else:
                        # Use cubic spline for smoother interpolation
                        interp_func = interp1d(valid_frames, valid_values, 
                                             kind='cubic', fill_value='extrapolate')
                    
                    # Interpolate value for this frame
                    frame_data[col] = float(interp_func(frame_idx))
                else:
                    # Fallback: use mean of valid data
                    frame_data[col] = df[col].mean() if len(df) > 0 else 0.0
            
            # Set confidence lower for interpolated frames
            frame_data['left_arm_confidence'] *= 0.5  # Mark as interpolated
            frame_data['right_arm_confidence'] *= 0.5
        
        interpolated_data.append(frame_data)
    
    print(f"INTERPOLATION COMPLETE")
    print(f"  Total frames: {len(interpolated_data)}")
    print(f"  Valid frames: {len([d for d in interpolated_data if not d['interpolated']])}")
    print(f"  Interpolated: {len([d for d in interpolated_data if d['interpolated']])}")
    
    return interpolated_data

def create_complete_arm_analysis_export(pkl_file, output_dir):
    """
    Create complete arm analysis with ALL 811 frames using interpolation
    """
    
    print(f"CREATING COMPLETE ARM ANALYSIS (811 FRAMES)")
    print(f"Input: {pkl_file}")
    print("=" * 60)
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    total_frames = len(meshes)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Import required functions
    from visualize_arm_vectors_with_trunk import (
        create_arrow_mesh, 
        create_angle_arc, 
        calculate_trunk_angle_to_gravity
    )
    
    # First pass: collect all valid angle data
    print("PASS 1: Calculating valid angles...")
    valid_angle_data = []
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']
        
        # Calculate angles with strict validation
        bilateral_result = calculate_bilateral_arm_angles(joints)
        
        if bilateral_result['left_arm'] and bilateral_result['right_arm']:
            # Valid calculation
            trunk_angle_deg, _ = calculate_trunk_angle_to_gravity(
                joints[SMPL_X_JOINT_INDICES['neck']] - joints[SMPL_X_JOINT_INDICES['spine1']]
            )
            trunk_angle = trunk_angle_deg
            
            angle_data = {
                'frame': frame_idx,
                'trunk_angle': trunk_angle,
                'left_arm_angle': bilateral_result['left_arm']['sagittal_angle'],
                'right_arm_angle': bilateral_result['right_arm']['sagittal_angle'],
                'left_arm_confidence': bilateral_result['left_arm']['confidence'],
                'right_arm_confidence': bilateral_result['right_arm']['confidence']
            }
            valid_angle_data.append(angle_data)
        
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}: {len(valid_angle_data)} valid so far")
    
    print(f"Valid angle calculations: {len(valid_angle_data)}/{total_frames}")
    
    # Second pass: interpolate missing data
    complete_angle_data = interpolate_missing_angles(valid_angle_data, total_frames)
    
    # Third pass: create OBJ exports for all frames
    print(f"PASS 2: Creating OBJ exports for all {total_frames} frames...")
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']
        angle_data = complete_angle_data[frame_idx]
        
        # Get interpolated or real angle data
        trunk_angle = float(angle_data['trunk_angle'])
        left_arm_angle = float(angle_data['left_arm_angle'])
        right_arm_angle = float(angle_data['right_arm_angle'])
        is_interpolated = angle_data['interpolated']
        
        # Extract joint positions for visualization
        lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]
        cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Create combined mesh with all vectors
        all_vertices = []
        all_faces = []
        current_vertex_offset = 0
        
        # Calculate trunk vector for stable reference
        trunk_vector = cervical_joint - lumbar_joint
        
        # 1. Trunk vector (RED)
        trunk_vertices, trunk_faces = create_arrow_mesh(
            lumbar_joint, cervical_joint, arrow_radius=0.012
        )
        all_vertices.extend(trunk_vertices)
        all_faces.extend([[f[0]+current_vertex_offset, f[1]+current_vertex_offset, f[2]+current_vertex_offset] 
                         for f in trunk_faces])
        current_vertex_offset += len(trunk_vertices)
        
        # 2. Left arm vector (GREEN)
        left_arm_vertices, left_arm_faces = create_arrow_mesh(
            left_shoulder, left_elbow, arrow_radius=0.010
        )
        all_vertices.extend(left_arm_vertices)
        all_faces.extend([[f[0]+current_vertex_offset, f[1]+current_vertex_offset, f[2]+current_vertex_offset] 
                         for f in left_arm_faces])
        current_vertex_offset += len(left_arm_vertices)
        
        # 3. Right arm vector (BLUE)
        right_arm_vertices, right_arm_faces = create_arrow_mesh(
            right_shoulder, right_elbow, arrow_radius=0.010
        )
        all_vertices.extend(right_arm_vertices)
        all_faces.extend([[f[0]+current_vertex_offset, f[1]+current_vertex_offset, f[2]+current_vertex_offset] 
                         for f in right_arm_faces])
        current_vertex_offset += len(right_arm_vertices)
        
        # Export OBJ file
        obj_file = output_dir / f"complete_arm_analysis_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w', encoding='utf-8') as f:
            # Header with complete angle data
            f.write(f"# Complete Arm Analysis Frame {frame_idx}\n")
            f.write(f"# Data Quality: {'INTERPOLATED' if is_interpolated else 'MEASURED'}\n")
            f.write(f"# TRUNK ANGLE: {trunk_angle:.1f}° (relative to gravity)\n")
            f.write(f"# LEFT ARM: {left_arm_angle:.1f}° (relative to trunk)\n")
            f.write(f"# RIGHT ARM: {right_arm_angle:.1f}° (relative to trunk)\n")
            f.write(f"# LEFT CONFIDENCE: {angle_data['left_arm_confidence']:.3f}\n")
            f.write(f"# RIGHT CONFIDENCE: {angle_data['right_arm_confidence']:.3f}\n")
            f.write(f"# COLOR CODING: RED=trunk, GREEN=left arm, BLUE=right arm\n\n")
            
            # Write vertices
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces with group labels
            vertex_groups = [
                (trunk_faces, "TrunkVector"),
                (left_arm_faces, "LeftArm"),
                (right_arm_faces, "RightArm"),
            ]
            
            face_offset = 1
            for faces, group_name in vertex_groups:
                if len(faces) > 0:
                    f.write(f"g {group_name}\n")
                    for face in faces:
                        f.write(f"f {face[0]+face_offset} {face[1]+face_offset} {face[2]+face_offset}\n")
                    face_offset += len([v for v_group, g in zip([trunk_vertices, left_arm_vertices, right_arm_vertices], 
                                                              [g[1] for g in vertex_groups]) if g == group_name for v in v_group])
        
        # Progress output
        if frame_idx % 50 == 0:
            status = "INTERPOLATED" if is_interpolated else "MEASURED"
            print(f"Frame {frame_idx:3d}: {status} - Trunk={trunk_angle:.1f}°, L={left_arm_angle:.1f}°, R={right_arm_angle:.1f}°")
    
    # Create statistics file
    stats_file = output_dir / "complete_arm_analysis_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("COMPLETE ARM ANALYSIS STATISTICS (811 FRAMES)\n")
        f.write("=" * 60 + "\n\n")
        
        # Calculate statistics
        trunk_angles = [d['trunk_angle'] for d in complete_angle_data]
        left_angles = [d['left_arm_angle'] for d in complete_angle_data]
        right_angles = [d['right_arm_angle'] for d in complete_angle_data]
        
        measured_count = len([d for d in complete_angle_data if not d['interpolated']])
        interpolated_count = len([d for d in complete_angle_data if d['interpolated']])
        
        f.write(f"DATA SUMMARY:\n")
        f.write(f"  Total frames: {len(complete_angle_data)}\n")
        f.write(f"  Measured frames: {measured_count}\n") 
        f.write(f"  Interpolated frames: {interpolated_count}\n")
        f.write(f"  Data completeness: 100% (with interpolation)\n\n")
        
        f.write(f"TRUNK ANGLE STATISTICS:\n")
        f.write(f"  Mean: {np.mean(trunk_angles):.1f}°\n")
        f.write(f"  Std Dev: {np.std(trunk_angles):.1f}°\n") 
        f.write(f"  Range: {np.min(trunk_angles):.1f}° to {np.max(trunk_angles):.1f}°\n\n")
        
        f.write(f"LEFT ARM STATISTICS:\n")
        f.write(f"  Mean: {np.mean(left_angles):.1f}°\n")
        f.write(f"  Std Dev: {np.std(left_angles):.1f}°\n")
        f.write(f"  Range: {np.min(left_angles):.1f}° to {np.max(left_angles):.1f}°\n\n")
        
        f.write(f"RIGHT ARM STATISTICS:\n")
        f.write(f"  Mean: {np.mean(right_angles):.1f}°\n")
        f.write(f"  Std Dev: {np.std(right_angles):.1f}°\n")
        f.write(f"  Range: {np.min(right_angles):.1f}° to {np.max(right_angles):.1f}°\n\n")
        
        f.write(f"QUALITY NOTES:\n")
        f.write(f"  - Measured data uses strict validation for maximum precision\n")
        f.write(f"  - Interpolated data fills gaps using cubic spline interpolation\n")
        f.write(f"  - Interpolated frames have reduced confidence values (×0.5)\n")
        f.write(f"  - All frames marked in OBJ headers as MEASURED or INTERPOLATED\n")
    
    print(f"\nCOMPLETE ARM ANALYSIS EXPORT FINISHED!")
    print(f"  Total frames: {len(complete_angle_data)}")
    print(f"  Measured: {measured_count} ({measured_count/len(complete_angle_data)*100:.1f}%)")
    print(f"  Interpolated: {interpolated_count} ({interpolated_count/len(complete_angle_data)*100:.1f}%)")
    print(f"  Files saved to: {output_dir}")
    print(f"  Statistics: {stats_file}")
    
    return output_dir

if __name__ == "__main__":
    pkl_file = "fil_vid_meshes.pkl"
    output_dir = "complete_arm_analysis_export"
    
    if Path(pkl_file).exists():
        result_dir = create_complete_arm_analysis_export(pkl_file, output_dir)
        print(f"\nREADY FOR COMPREHENSIVE VISUALIZATION!")
        print(f"Now update comprehensive script to use: complete_arm_analysis_export/")
    else:
        print(f"PKL file not found: {pkl_file}")
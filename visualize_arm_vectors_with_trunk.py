#!/usr/bin/env python3
"""
Visualize arm vectors with trunk vector from PKL mesh data
Calculates arm vectors from shoulder to elbow joints for both arms
Export for Blender side-by-side visualization with trunk analysis
"""

import pickle
import numpy as np
import os
from pathlib import Path
import math

# SMPL-X joint indices for trunk and arm analysis
SMPL_X_JOINT_INDICES = {
    # Trunk joints (existing)
    'pelvis': 0,          # Root/pelvis joint
    'spine1': 3,          # Lower spine (lumbar region)  
    'spine2': 6,          # Mid spine
    'spine3': 9,          # Upper spine
    'neck': 12,           # Neck base (cervical region)
    'head': 15,           # Head
    # Arm joints (new)
    'left_shoulder': 16,  # Left shoulder joint
    'right_shoulder': 17, # Right shoulder joint
    'left_elbow': 18,     # Left elbow joint
    'right_elbow': 19,    # Right elbow joint
}

def create_arrow_mesh(start_point, end_point, arrow_radius=0.01, shaft_segments=8, head_length_ratio=0.2):
    """Create arrow mesh geometry from start to end point"""
    
    # Calculate arrow direction and length
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    
    if length < 1e-6:  # Avoid zero-length arrows
        direction = np.array([0, 0, 1])
        length = 0.1
    else:
        direction = direction / length
    
    # Calculate shaft and head dimensions
    shaft_length = length * (1 - head_length_ratio)
    head_length = length * head_length_ratio
    head_radius = arrow_radius * 2.5
    
    vertices = []
    faces = []
    
    # Create coordinate system for the arrow
    if abs(direction[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, direction)
    
    # Shaft vertices (cylinder)
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = arrow_radius * (np.cos(angle) * right + np.sin(angle) * up)
        
        # Bottom of shaft (at start_point)
        vertices.append(start_point + offset)
        # Top of shaft
        shaft_end = start_point + direction * shaft_length
        vertices.append(shaft_end + offset)
    
    # Create shaft faces (cylinder)
    base_idx = 0
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        # Two triangles per segment
        bottom_curr = base_idx + i * 2
        bottom_next = base_idx + next_i * 2
        top_curr = bottom_curr + 1
        top_next = bottom_next + 1
        
        # Triangle 1
        faces.append([bottom_curr, top_curr, bottom_next])
        # Triangle 2
        faces.append([top_curr, top_next, bottom_next])
    
    # Arrow head vertices (cone)
    head_base_center = start_point + direction * shaft_length
    arrow_tip = end_point
    head_base_idx = len(vertices)
    
    # Add center point for head base
    vertices.append(head_base_center)
    center_idx = len(vertices) - 1
    
    # Head base circle
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = head_radius * (np.cos(angle) * right + np.sin(angle) * up)
        vertices.append(head_base_center + offset)
    
    # Add tip vertex
    vertices.append(arrow_tip)
    tip_idx = len(vertices) - 1
    
    # Create head faces
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        base_curr = head_base_idx + 1 + i
        base_next = head_base_idx + 1 + next_i
        
        # Base triangle (connecting to center)
        faces.append([center_idx, base_next, base_curr])
        
        # Side triangle (connecting to tip)
        faces.append([base_curr, base_next, tip_idx])
    
    return np.array(vertices), np.array(faces)

def create_angle_arc(center, start_vec, end_vec, radius=0.15, segments=16):
    """Create arc mesh to visualize angle between two vectors"""
    
    # Normalize vectors
    start_norm = start_vec / np.linalg.norm(start_vec)
    end_norm = end_vec / np.linalg.norm(end_vec)
    
    # Calculate angle
    dot_product = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    if angle < 0.01:  # Very small angle, skip arc
        return np.array([]), np.array([])
    
    # Create rotation axis (perpendicular to both vectors)
    cross_product = np.cross(start_norm, end_norm)
    if np.linalg.norm(cross_product) < 1e-6:
        return np.array([]), np.array([])
    
    rotation_axis = cross_product / np.linalg.norm(cross_product)
    
    vertices = [center]  # Center point
    faces = []
    
    # Create arc points
    for i in range(segments + 1):
        t = i / segments * angle
        
        # Rodrigues rotation formula
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        
        rotated_vec = (start_norm * cos_t + 
                      np.cross(rotation_axis, start_norm) * sin_t + 
                      rotation_axis * np.dot(rotation_axis, start_norm) * (1 - cos_t))
        
        vertices.append(center + rotated_vec * radius)
    
    # Create triangular faces for arc
    for i in range(segments):
        faces.append([0, i + 1, i + 2])  # Center to arc segments
    
    return np.array(vertices), np.array(faces)

def calculate_trunk_angle_to_gravity(trunk_vector):
    """Calculate angle between trunk vector and gravity (vertical down)"""
    gravity_vector = np.array([0, 0, -1])  # Pointing down (negative Z)
    
    # Normalize vectors
    trunk_norm = trunk_vector / np.linalg.norm(trunk_vector)
    
    # Calculate angle
    dot_product = np.clip(np.dot(trunk_norm, gravity_vector), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, angle_rad

def export_arm_vectors_with_trunk_to_obj(pkl_file, output_dir):
    """Export arm vectors with trunk vector sequence for side-by-side visualization"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"EXPORTUJI ARM VEKTORY + TRUNK Z {len(meshes)} SNIMKU")
    print("=" * 60)
    
    angle_data = []  # For trunk angle statistics
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']  # Shape: (117, 3)
        
        # Extract trunk joint positions
        lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]  # Lower spine
        cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]  # Neck base
        
        # Extract arm joint positions
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Calculate vectors
        trunk_vector = cervical_joint - lumbar_joint
        left_arm_vector = left_elbow - left_shoulder
        right_arm_vector = right_elbow - right_shoulder
        trunk_length = np.linalg.norm(trunk_vector)
        
        # Calculate trunk angle to gravity
        angle_deg, angle_rad = calculate_trunk_angle_to_gravity(trunk_vector)
        angle_data.append(angle_deg)
        
        # Create combined mesh with all vectors
        all_vertices = []
        all_faces = []
        current_vertex_offset = 0
        
        # 1. Trunk vector arrow (RED)
        trunk_vertices, trunk_faces = create_arrow_mesh(lumbar_joint, cervical_joint, arrow_radius=0.012)
        all_vertices.extend(trunk_vertices)
        all_faces.extend(trunk_faces + current_vertex_offset)
        current_vertex_offset += len(trunk_vertices)
        
        # 2. Gravitational reference vector (BLUE) - from lumbar joint downward
        gravity_end = lumbar_joint + np.array([0, 0, -trunk_length])  # Same length as trunk
        gravity_vertices, gravity_faces = create_arrow_mesh(lumbar_joint, gravity_end, arrow_radius=0.008)
        all_vertices.extend(gravity_vertices)
        all_faces.extend(gravity_faces + current_vertex_offset)
        current_vertex_offset += len(gravity_vertices)
        
        # 3. Angle arc visualization (GREEN)
        arc_vertices, arc_faces = create_angle_arc(lumbar_joint, trunk_vector, np.array([0, 0, -trunk_length]), radius=0.08)
        if len(arc_vertices) > 0:
            all_vertices.extend(arc_vertices)
            all_faces.extend(arc_faces + current_vertex_offset)
            current_vertex_offset += len(arc_vertices)
        
        # 4. Left arm vector (YELLOW)
        left_arm_vertices, left_arm_faces = create_arrow_mesh(left_shoulder, left_elbow, arrow_radius=0.010)
        all_vertices.extend(left_arm_vertices)
        all_faces.extend(left_arm_faces + current_vertex_offset)
        current_vertex_offset += len(left_arm_vertices)
        
        # 5. Right arm vector (ORANGE)
        right_arm_vertices, right_arm_faces = create_arrow_mesh(right_shoulder, right_elbow, arrow_radius=0.010)
        all_vertices.extend(right_arm_vertices)
        all_faces.extend(right_arm_faces + current_vertex_offset)
        
        # OBJ filename
        obj_file = output_dir / f"arm_analysis_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w') as f:
            # Write header with analysis data
            f.write(f"# Arm + Trunk Analysis Frame {frame_idx}\n")
            f.write(f"# Lumbar: [{lumbar_joint[0]:.3f}, {lumbar_joint[1]:.3f}, {lumbar_joint[2]:.3f}]\n")
            f.write(f"# Cervical: [{cervical_joint[0]:.3f}, {cervical_joint[1]:.3f}, {cervical_joint[2]:.3f}]\n")
            f.write(f"# Left Shoulder: [{left_shoulder[0]:.3f}, {left_shoulder[1]:.3f}, {left_shoulder[2]:.3f}]\n")
            f.write(f"# Left Elbow: [{left_elbow[0]:.3f}, {left_elbow[1]:.3f}, {left_elbow[2]:.3f}]\n")
            f.write(f"# Right Shoulder: [{right_shoulder[0]:.3f}, {right_shoulder[1]:.3f}, {right_shoulder[2]:.3f}]\n")
            f.write(f"# Right Elbow: [{right_elbow[0]:.3f}, {right_elbow[1]:.3f}, {right_elbow[2]:.3f}]\n")
            f.write(f"# Trunk Angle: {angle_deg:.1f}°\n")
            f.write(f"# RED=Trunk, BLUE=Gravity, GREEN=Angle, YELLOW=LeftArm, ORANGE=RightArm\n\n")
            
            # Write vertices and faces for each component
            
            # 1. Trunk vertices (RED)
            for v in trunk_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("g TrunkVector\n")
            for face in trunk_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            # 2. Gravity vertices (BLUE)
            for v in gravity_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("g GravityReference\n")
            for face in gravity_faces:
                f.write(f"f {face[0]+len(trunk_vertices)+1} {face[1]+len(trunk_vertices)+1} {face[2]+len(trunk_vertices)+1}\n")
            
            # 3. Arc vertices (GREEN) - if exists
            arc_offset = len(trunk_vertices) + len(gravity_vertices)
            if len(arc_vertices) > 0:
                for v in arc_vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write("g AngleArc\n")
                for face in arc_faces:
                    f.write(f"f {face[0]+arc_offset+1} {face[1]+arc_offset+1} {face[2]+arc_offset+1}\n")
                arm_offset = arc_offset + len(arc_vertices)
            else:
                arm_offset = arc_offset
            
            # 4. Left arm vertices (YELLOW)
            for v in left_arm_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("g LeftArmVector\n")
            for face in left_arm_faces:
                f.write(f"f {face[0]+arm_offset+1} {face[1]+arm_offset+1} {face[2]+arm_offset+1}\n")
            
            # 5. Right arm vertices (ORANGE)
            right_arm_offset = arm_offset + len(left_arm_vertices)
            for v in right_arm_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("g RightArmVector\n")
            for face in right_arm_faces:
                f.write(f"f {face[0]+right_arm_offset+1} {face[1]+right_arm_offset+1} {face[2]+right_arm_offset+1}\n")
        
        print(f"  Frame {frame_idx:3d}: Trunk={angle_deg:5.1f}°, Arms=L{np.linalg.norm(left_arm_vector):.3f}/R{np.linalg.norm(right_arm_vector):.3f}m")
    
    # Export trunk angle statistics
    stats_file = output_dir / "trunk_angle_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("ARM + TRUNK ANALYSIS STATISTICS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total frames: {len(angle_data)}\n")
        f.write(f"Min trunk angle: {min(angle_data):.1f}°\n")
        f.write(f"Max trunk angle: {max(angle_data):.1f}°\n")
        f.write(f"Avg trunk angle: {np.mean(angle_data):.1f}°\n")
        f.write(f"Std deviation: {np.std(angle_data):.1f}°\n\n")
        f.write("VECTOR COLOR CODING:\n")
        f.write("RED = Trunk vector (lumbar→cervical)\n")
        f.write("BLUE = Gravity reference (downward)\n") 
        f.write("GREEN = Trunk angle arc\n")
        f.write("YELLOW = Left arm vector (shoulder→elbow)\n")
        f.write("ORANGE = Right arm vector (shoulder→elbow)\n")
    
    print(f"\nEXPORT COMPLETE!")
    print(f"Files: {output_dir}")
    print(f"Frames: {len(meshes)}")
    print(f"Avg trunk angle: {np.mean(angle_data):.1f}°")
    
    return output_dir

if __name__ == "__main__":
    export_arm_vectors_with_trunk_to_obj("arm_meshes.pkl", "arm_analysis_export")
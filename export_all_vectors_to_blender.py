#!/usr/bin/env python3
"""
Export complete visualization: 3D mesh + all vectors (trunk, neck, arms) to OBJ files
Creates separate OBJ files for mesh and each vector for Blender import
"""

import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import calculators
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES
from neck_angle_calculator_like_arm import calculate_neck_angle_to_trunk_like_arm

# Hardcoded head vertex for neck vector
HEAD_VERTEX_ID = 9002

def create_arrow_mesh(start_point, end_point, arrow_radius=0.01, shaft_segments=8, head_length_ratio=0.2):
    """Create arrow mesh geometry from start to end point"""
    
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    
    if length < 1e-6:
        return np.array([]), np.array([])
    
    direction = direction / length
    
    # Calculate shaft and head dimensions
    shaft_length = length * (1 - head_length_ratio)
    head_length = length * head_length_ratio
    head_radius = arrow_radius * 2.5
    
    vertices = []
    faces = []
    
    # Create coordinate system
    if abs(direction[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    right = np.cross(direction, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    up = np.cross(right, direction)
    
    # Shaft vertices (cylinder)
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = arrow_radius * (np.cos(angle) * right + np.sin(angle) * up)
        
        # Bottom of shaft
        vertices.append(start_point + offset)
        # Top of shaft
        shaft_end = start_point + direction * shaft_length
        vertices.append(shaft_end + offset)
    
    # Shaft faces
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        bottom_curr = i * 2
        bottom_next = next_i * 2
        top_curr = bottom_curr + 1
        top_next = bottom_next + 1
        
        faces.append([bottom_curr, top_curr, bottom_next])
        faces.append([top_curr, top_next, bottom_next])
    
    # Arrow head (cone)
    head_base_center = start_point + direction * shaft_length
    arrow_tip = end_point
    head_base_idx = len(vertices)
    
    # Center point for head base
    vertices.append(head_base_center)
    center_idx = len(vertices) - 1
    
    # Head base circle
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = head_radius * (np.cos(angle) * right + np.sin(angle) * up)
        vertices.append(head_base_center + offset)
    
    # Tip vertex
    vertices.append(arrow_tip)
    tip_idx = len(vertices) - 1
    
    # Head faces
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        base_curr = head_base_idx + 1 + i
        base_next = head_base_idx + 1 + next_i
        
        faces.append([center_idx, base_next, base_curr])
        faces.append([base_curr, base_next, tip_idx])
    
    return np.array(vertices), np.array(faces)

def write_obj_file(filepath, vertices, faces):
    """Write vertices and faces to OBJ file"""
    with open(filepath, 'w') as f:
        f.write(f"# OBJ file generated from vector data\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def export_all_vectors_to_blender(pkl_file, output_dir="blender_export_all"):
    """
    Export complete visualization: mesh + all vectors to OBJ files
    
    Creates:
    - frame_XXXX.obj - 3D mesh
    - trunk_XXXX.obj - trunk vector (lumbar to cervical)
    - neck_XXXX.obj - neck vector (cervical to head)
    - left_arm_XXXX.obj - left arm vector
    - right_arm_XXXX.obj - right arm vector
    """
    
    print("EXPORTING ALL VECTORS TO BLENDER")
    print("=" * 60)
    
    # Load PKL data
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        return None
    
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # Handle both old and new PKL format
    if isinstance(pkl_data, dict) and 'mesh_sequence' in pkl_data:
        # New format with metadata
        meshes = pkl_data['mesh_sequence']
        metadata = pkl_data.get('metadata', {})
        print(f"  New PKL format detected with metadata")
        if 'fps' in metadata:
            print(f"  FPS from PKL: {metadata['fps']:.2f}")
        if 'video_filename' in metadata:
            print(f"  Original video: {metadata['video_filename']}")
    else:
        # Old format - just mesh sequence
        meshes = pkl_data
        print(f"  Old PKL format detected")
    
    print(f"Loaded {len(meshes)} frames")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Clean up old files
    for old_file in output_dir.glob("*.obj"):
        old_file.unlink()
    
    print(f"\nExporting to: {output_dir}")
    print("-" * 60)
    
    # Process each frame
    for frame_idx, mesh_data in enumerate(meshes):
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        joints = mesh_data['joints']
        
        # 1. Export mesh
        mesh_file = output_dir / f"frame_{frame_idx:04d}.obj"
        write_obj_file(mesh_file, vertices, faces)
        
        # Extract joint positions
        lumbar = joints[SMPL_X_JOINT_INDICES['spine1']]
        cervical = joints[SMPL_X_JOINT_INDICES['neck']]
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Head vertex for neck vector
        head_vertex = vertices[HEAD_VERTEX_ID]
        
        # 2. Export trunk vector (lumbar to cervical)
        trunk_arrow_verts, trunk_arrow_faces = create_arrow_mesh(
            lumbar, cervical, arrow_radius=0.015
        )
        if len(trunk_arrow_verts) > 0:
            trunk_file = output_dir / f"trunk_{frame_idx:04d}.obj"
            write_obj_file(trunk_file, trunk_arrow_verts, trunk_arrow_faces)
        
        # 3. Export neck vector (cervical to head)
        neck_arrow_verts, neck_arrow_faces = create_arrow_mesh(
            cervical, head_vertex, arrow_radius=0.012
        )
        if len(neck_arrow_verts) > 0:
            neck_file = output_dir / f"neck_{frame_idx:04d}.obj"
            write_obj_file(neck_file, neck_arrow_verts, neck_arrow_faces)
        
        # 4. Export left arm vector
        left_arm_arrow_verts, left_arm_arrow_faces = create_arrow_mesh(
            left_shoulder, left_elbow, arrow_radius=0.012
        )
        if len(left_arm_arrow_verts) > 0:
            left_arm_file = output_dir / f"left_arm_{frame_idx:04d}.obj"
            write_obj_file(left_arm_file, left_arm_arrow_verts, left_arm_arrow_faces)
        
        # 5. Export right arm vector
        right_arm_arrow_verts, right_arm_arrow_faces = create_arrow_mesh(
            right_shoulder, right_elbow, arrow_radius=0.012
        )
        if len(right_arm_arrow_verts) > 0:
            right_arm_file = output_dir / f"right_arm_{frame_idx:04d}.obj"
            write_obj_file(right_arm_file, right_arm_arrow_verts, right_arm_arrow_faces)
        
        # Calculate angles for info
        trunk_vector = cervical - lumbar
        trunk_length = np.linalg.norm(trunk_vector)
        trunk_angle = 0.0
        
        if trunk_length > 0:
            spine_unit = trunk_vector / trunk_length
            vertical = np.array([0, 1, 0])
            cos_angle = np.dot(spine_unit, vertical)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            trunk_angle = np.degrees(np.arccos(cos_angle))
        
        # Progress output
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"Frame {frame_idx:3d}: Mesh + 4 vectors exported (trunk angle: {trunk_angle:.1f}°)")
    
    print("-" * 60)
    print(f"EXPORT COMPLETE!")
    print(f"Output directory: {output_dir}")
    print(f"Files per frame: 5 (mesh + 4 vectors)")
    print(f"Total files: {len(meshes) * 5}")
    
    return output_dir

def main():
    """Main execution"""
    pkl_file = "arm_meshes.pkl"
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Available PKL files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
        return
    
    # Export all vectors
    output_dir = export_all_vectors_to_blender(pkl_file)
    
    if output_dir:
        print(f"\nNEXT STEP:")
        print(f"1. Open Blender")
        print(f"2. Run the visualization script: blender_visualize_all_vectors.py")
        print(f"3. Enjoy the complete visualization!")

if __name__ == "__main__":
    main()
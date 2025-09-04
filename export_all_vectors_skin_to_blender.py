#!/usr/bin/env python3
"""
Export complete visualization with SKIN-BASED trunk vector
Uses vertices 2151 (cervical) and 5614 (lumbar) on actual skin surface
Plus all other vectors (neck, arms) as before
"""

import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import calculators
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES
from neck_angle_calculator_like_arm import calculate_neck_angle_to_trunk_like_arm

# SKIN VERTEX IDs
CERVICAL_SKIN_VERTEX = 2151  # Neck area on skin
LUMBAR_SKIN_VERTEX = 5614    # Lower back on skin (alt: 4298)
HEAD_VERTEX_ID = 9002         # Head vertex (already on skin)

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

def export_all_vectors_skin_to_blender(pkl_file, output_dir="blender_export_skin", lumbar_vertex=5614):
    """
    Export complete visualization with SKIN-BASED trunk vector
    
    Creates:
    - frame_XXXX.obj - 3D mesh
    - trunk_skin_XXXX.obj - trunk vector FROM SKIN VERTICES (lumbar to cervical on skin)
    - neck_XXXX.obj - neck vector (cervical to head)
    - left_arm_XXXX.obj - left arm vector
    - right_arm_XXXX.obj - right arm vector
    """
    
    print("EXPORTING VECTORS WITH SKIN-BASED TRUNK")
    print("=" * 60)
    print(f"Using SKIN vertices for trunk:")
    print(f"  Lumbar (lower back): Vertex {lumbar_vertex}")
    print(f"  Cervical (neck): Vertex {CERVICAL_SKIN_VERTEX}")
    print("=" * 60)
    
    # Load PKL data
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        return None
    
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"Loaded {len(meshes)} frames")
    
    # Verify vertices exist
    if meshes:
        first_frame = meshes[0]
        num_vertices = len(first_frame['vertices'])
        print(f"Mesh has {num_vertices} vertices")
        
        if lumbar_vertex >= num_vertices:
            print(f"ERROR: Lumbar vertex {lumbar_vertex} out of range!")
            return None
        if CERVICAL_SKIN_VERTEX >= num_vertices:
            print(f"ERROR: Cervical vertex {CERVICAL_SKIN_VERTEX} out of range!")
            return None
    
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
        
        # Get SKIN points for trunk
        lumbar_skin = vertices[lumbar_vertex]
        cervical_skin = vertices[CERVICAL_SKIN_VERTEX]
        
        # Get joint positions for arms (still using joints for arms)
        left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
        left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
        right_elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        
        # Head vertex for neck vector
        head_vertex = vertices[HEAD_VERTEX_ID]
        
        # 2. Export SKIN-BASED trunk vector (lumbar to cervical ON SKIN)
        trunk_skin_arrow_verts, trunk_skin_arrow_faces = create_arrow_mesh(
            lumbar_skin, cervical_skin, arrow_radius=0.018  # Slightly thicker for visibility
        )
        if len(trunk_skin_arrow_verts) > 0:
            trunk_file = output_dir / f"trunk_skin_{frame_idx:04d}.obj"
            write_obj_file(trunk_file, trunk_skin_arrow_verts, trunk_skin_arrow_faces)
        
        # 3. Export SKIN-BASED neck vector (cervical skin to head skin - both on surface!)
        neck_arrow_verts, neck_arrow_faces = create_arrow_mesh(
            cervical_skin, head_vertex, arrow_radius=0.015  # Slightly thicker for visibility
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
        
        # Calculate skin-based trunk angle for info
        trunk_vector_skin = cervical_skin - lumbar_skin
        trunk_length = np.linalg.norm(trunk_vector_skin)
        trunk_angle_skin = 0.0
        
        if trunk_length > 0:
            spine_unit = trunk_vector_skin / trunk_length
            vertical = np.array([0, 1, 0])
            cos_angle = np.dot(spine_unit, vertical)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            trunk_angle_skin = np.degrees(np.arccos(cos_angle))
        
        # Progress output
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"Frame {frame_idx:3d}: Mesh + 4 vectors (skin trunk angle: {trunk_angle_skin:.1f}°)")
    
    print("-" * 60)
    print(f"EXPORT COMPLETE!")
    print(f"Output directory: {output_dir}")
    print(f"Files per frame: 5 (mesh + 4 vectors)")
    print(f"Total files: {len(meshes) * 5}")
    print(f"\nVECTORS FROM SKIN VERTICES:")
    print(f"  TRUNK: Lumbar vertex {lumbar_vertex} → Cervical vertex {CERVICAL_SKIN_VERTEX}")
    print(f"  NECK: Cervical vertex {CERVICAL_SKIN_VERTEX} → Head vertex {HEAD_VERTEX_ID}")
    print(f"  (Both trunk and neck are now on skin surface!)")
    
    return output_dir

def main():
    """Main execution with vertex choice"""
    pkl_file = "arm_meshes.pkl"
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Available PKL files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
        return
    
    print("\nSKIN-BASED VECTOR EXPORT")
    print("Choose lumbar vertex for trunk:")
    print("1. Vertex 5614 (default)")
    print("2. Vertex 4298 (alternative)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        output_dir = export_all_vectors_skin_to_blender(pkl_file, 
                                                        output_dir="blender_export_skin_4298",
                                                        lumbar_vertex=4298)
    else:
        output_dir = export_all_vectors_skin_to_blender(pkl_file, 
                                                        output_dir="blender_export_skin_5614",
                                                        lumbar_vertex=5614)
    
    if output_dir:
        print(f"\nNEXT STEP:")
        print(f"1. Open Blender")
        print(f"2. Use one of the visualization scripts with this directory:")
        print(f"   - Update base_dir in script to: {output_dir}")
        print(f"3. The trunk vector will now be ON THE SKIN SURFACE!")

if __name__ == "__main__":
    main()
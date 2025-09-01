#!/usr/bin/env python3
"""
Export trunk vector sequence to Blender-compatible format
Visualizes trunk posture as arrows between lumbar and cervical joints
"""

import pickle
import numpy as np
import os
from pathlib import Path

# SMPL-X joint indices for trunk analysis
SMPL_X_JOINT_INDICES = {
    'pelvis': 0,          # Root/pelvis joint
    'spine1': 3,          # Lower spine (lumbar region)  
    'spine2': 6,          # Mid spine
    'spine3': 9,          # Upper spine
    'neck': 12,           # Neck base (cervical region)
    'head': 15,           # Head
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

def export_trunk_vectors_to_obj(pkl_file, output_dir):
    """Export trunk vector sequence as OBJ files for Blender"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"EXPORTING TRUNK VECTORS FROM {len(meshes)} FRAMES TO BLENDER")
    print("=" * 60)
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']  # Shape: (117, 3)
        
        # Extract lumbar and cervical joint positions
        lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]  # Lower spine
        cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]  # Neck base
        
        # Create arrow mesh from lumbar to cervical
        vertices, faces = create_arrow_mesh(lumbar_joint, cervical_joint)
        
        # OBJ filename (frame padding for correct sorting)
        obj_file = output_dir / f"trunk_vector_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w') as f:
            # Write header
            f.write(f"# Trunk Vector Frame {frame_idx} - Lumbar to Cervical\n")
            f.write(f"# Lumbar: [{lumbar_joint[0]:.3f}, {lumbar_joint[1]:.3f}, {lumbar_joint[2]:.3f}]\n")
            f.write(f"# Cervical: [{cervical_joint[0]:.3f}, {cervical_joint[1]:.3f}, {cervical_joint[2]:.3f}]\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Calculate trunk vector length for analysis
        trunk_length = np.linalg.norm(cervical_joint - lumbar_joint)
        trunk_direction = (cervical_joint - lumbar_joint) / trunk_length if trunk_length > 0 else np.array([0,0,0])
        
        print(f"  Frame {frame_idx:3d}: Vector length={trunk_length:.3f}m, Direction=[{trunk_direction[0]:.2f},{trunk_direction[1]:.2f},{trunk_direction[2]:.2f}]")
    
    # Create Blender import script
    blender_script = output_dir / "import_trunk_vectors.py"
    with open(blender_script, 'w') as f:
        f.write(f'''
import bpy
import os
from pathlib import Path

def import_trunk_vector_sequence():
    """Import trunk vector sequence as keyframe animation in Blender"""
    
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"{output_dir.absolute()}")
    obj_files = sorted(base_dir.glob("trunk_vector_*.obj"))
    
    print(f"Found {{len(obj_files)}} trunk vector OBJ files")
    
    all_objects = []
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importing trunk vector frame {{frame_idx}}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"TrunkVector_{{frame_idx:04d}}"
            all_objects.append(obj)
            
            # Create material for trunk vector
            mat = bpy.data.materials.new(name=f"TrunkVector_Mat_{{frame_idx}}")
            mat.use_nodes = True
            
            # Set red color for trunk vector
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1.0)  # Red color
            bsdf.inputs[12].default_value = 0.1  # Roughness
            
            # Assign material
            obj.data.materials.append(mat)
            
            # Hide all objects except first
            if frame_idx > 0:
                obj.hide_viewport = True
                obj.hide_render = True
            
            print(f"  OK: {{obj.name}}")
        else:
            print(f"  ERROR: No object imported")
    
    print(f"\\nSETTING UP TRUNK VECTOR ANIMATION...")
    
    # Setup keyframes for visibility
    for frame_idx, obj in enumerate(all_objects):
        for other_frame, other_obj in enumerate(all_objects):
            if other_frame == frame_idx:
                # This frame should be visible
                other_obj.hide_viewport = False
                other_obj.hide_render = False
                other_obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                other_obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
            else:
                # Other frames should be hidden
                other_obj.hide_viewport = True
                other_obj.hide_render = True
                other_obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                other_obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(all_objects)
    bpy.context.scene.frame_set(1)
    
    print(f"TRUNK VECTOR ANIMATION READY!")
    print(f"Timeline: 1-{{len(all_objects)}} frames")
    print(f"Press SPACEBAR to play trunk posture animation")
    print(f"Red arrows show trunk orientation from lumbar to cervical spine")

if __name__ == "__main__":
    import_trunk_vector_sequence()
''')
    
    print(f"\nBLENDER INSTRUCTIONS:")
    print(f"  1. Open Blender")
    print(f"  2. Go to Scripting workspace")
    print(f"  3. Load script: {blender_script}")
    print(f"  4. Run script")
    print(f"  5. Press SPACE to play trunk vector timeline")
    print(f"  6. Red arrows show trunk posture changes over time")
    
    return output_dir

if __name__ == "__main__":
    export_trunk_vectors_to_obj("arm_meshes.pkl", "trunk_vector_export")
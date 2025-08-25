#!/usr/bin/env python3
"""
Export mesh sequence to Blender-compatible format
"""

import pickle
import numpy as np
import os
from pathlib import Path

def export_mesh_sequence_to_obj(pkl_file, output_dir):
    """Export mesh sequence as OBJ files for Blender"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"EXPORTING {len(meshes)} FRAMES TO BLENDER")
    print("=" * 50)
    
    for frame_idx, mesh_data in enumerate(meshes):
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # OBJ filename (frame padding for correct sorting)
        obj_file = output_dir / f"frame_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w') as f:
            # Write header
            f.write(f"# Frame {frame_idx} - Generated from SMPL-X mesh\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"  OK Frame {frame_idx:2d}: {obj_file}")
    
    # Create Blender import script
    blender_script = output_dir / "import_sequence.py"
    with open(blender_script, 'w') as f:
        f.write(f'''
import bpy
import os
from pathlib import Path

def import_mesh_sequence():
    """Import mesh sequence as keyframes in Blender"""
    
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"{output_dir.absolute()}")
    obj_files = sorted(base_dir.glob("frame_*.obj"))
    
    print(f"Found {{len(obj_files)}} OBJ files")
    
    for frame_idx, obj_file in enumerate(obj_files):
        # Set timeline frame
        bpy.context.scene.frame_set(frame_idx + 1)
        
        # Import OBJ
        bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported object
        obj = bpy.context.selected_objects[0]
        obj.name = f"HumanMesh_Frame_{{frame_idx:04d}}"
        
        # Hide all except current frame
        if frame_idx > 0:
            obj.hide_render = True
            obj.hide_viewport = True
        
        # Set keyframes for visibility
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
        
        if frame_idx > 0:
            obj.hide_viewport = True
            obj.hide_render = True
        if frame_idx < len(obj_files) - 1:
            bpy.context.scene.frame_set(frame_idx + 2)
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 2)
            obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 2)
        
        print(f"Imported frame {{frame_idx}}: {{obj.name}}")
    
    # Set timeline end
    bpy.context.scene.frame_end = len(obj_files)
    print(f"Timeline set to {{len(obj_files)}} frames")
    
    print("\\nIMPORT COMPLETE!")
    print("Press SPACE to play animation")
    print("Use mouse to rotate view")

if __name__ == "__main__":
    import_mesh_sequence()
''')
    
    print(f"\nBLENDER INSTRUCTIONS:")
    print(f"  1. Open Blender")
    print(f"  2. Go to Scripting workspace")
    print(f"  3. Load script: {blender_script}")
    print(f"  4. Run script")
    print(f"  5. Press SPACE to play timeline")
    print(f"  6. Mouse drag to rotate")
    
    return output_dir

if __name__ == "__main__":
    export_mesh_sequence_to_obj("frames2.pkl", "blender_export_2")
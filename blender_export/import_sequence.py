
import bpy
import os
from pathlib import Path

def import_mesh_sequence():
    """Import mesh sequence as keyframes in Blender"""
    
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\blender_export_2")
    obj_files = sorted(base_dir.glob("frame_*.obj"))
    
    print(f"Found {len(obj_files)} OBJ files")
    
    for frame_idx, obj_file in enumerate(obj_files):
        # Set timeline frame
        bpy.context.scene.frame_set(frame_idx + 1)
        
        # Import OBJ (compatible with Blender 4.x)
        try:
            # Try new Blender 4.x import
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except AttributeError:
            try:
                # Try old Blender 3.x import  
                bpy.ops.import_scene.obj(filepath=str(obj_file))
            except AttributeError:
                print(f"ERROR: Cannot import OBJ in this Blender version")
                return
        
        # Get imported object
        obj = bpy.context.selected_objects[0]
        obj.name = f"HumanMesh_Frame_{frame_idx:04d}"
        
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
        
        print(f"Imported frame {frame_idx}: {obj.name}")
    
    # Set timeline end
    bpy.context.scene.frame_end = len(obj_files)
    print(f"Timeline set to {len(obj_files)} frames")
    
    print("\nIMPORT COMPLETE!")
    print("Press SPACE to play animation")
    print("Use mouse to rotate view")

if __name__ == "__main__":
    import_mesh_sequence()

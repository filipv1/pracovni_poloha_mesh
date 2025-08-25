import bpy
from pathlib import Path

def import_working_sequence():
    """Working sequence import - step by step"""
    
    print("STARTING SEQUENCE IMPORT...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\blender_export_2")
    obj_files = sorted(base_dir.glob("frame_*.obj"))
    
    print(f"Found {len(obj_files)} OBJ files")
    
    all_objects = []
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importing frame {frame_idx}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"Frame_{frame_idx:04d}"
            all_objects.append(obj)
            
            # Hide all objects except first
            if frame_idx > 0:
                obj.hide_viewport = True
            
            print(f"  OK: {obj.name}")
        else:
            print(f"  ERROR: No object imported")
    
    print(f"\nSETTING UP ANIMATION...")
    
    # Setup keyframes for visibility
    for frame_idx, obj in enumerate(all_objects):
        # Set current frame
        bpy.context.scene.frame_set(1)
        
        for other_frame, other_obj in enumerate(all_objects):
            if other_frame == frame_idx:
                # This frame should be visible
                other_obj.hide_viewport = False
                other_obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
            else:
                # Other frames should be hidden
                other_obj.hide_viewport = True
                other_obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(all_objects)
    bpy.context.scene.frame_set(1)
    
    print(f"ANIMATION READY!")
    print(f"Timeline: 1-{len(all_objects)} frames")
    print(f"Press SPACEBAR to play")
    print(f"Use timeline slider to scrub through frames")

if __name__ == "__main__":
    import_working_sequence()
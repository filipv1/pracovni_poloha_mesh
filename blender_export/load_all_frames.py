import bpy
from pathlib import Path

def load_all_frames():
    """Load all frames as separate objects"""
    
    print("LOADING ALL FRAMES...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\blender_export_2")
    obj_files = sorted(base_dir.glob("frame_*.obj"))
    
    print(f"Found {len(obj_files)} OBJ files")
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Loading frame {frame_idx}...")
        
        # Import OBJ
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            try:
                bpy.ops.import_scene.obj(filepath=str(obj_file))
            except Exception as e:
                print(f"ERROR importing {obj_file}: {e}")
                continue
        
        # Rename imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"HumanFrame_{frame_idx:04d}"
            
            # Move each frame slightly to see them all
            obj.location.x = frame_idx * 2.0  # Spread them out
            
            print(f"  OK: {obj.name} at X={obj.location.x}")
    
    print(f"\nLOADED {len(obj_files)} FRAMES!")
    print("All frames are now visible side by side")
    print("You can manually hide/show them in Outliner")

if __name__ == "__main__":
    load_all_frames()
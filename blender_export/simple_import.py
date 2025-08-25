import bpy
from pathlib import Path

def simple_import():
    """Simple import - just load first frame"""
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import first frame
    obj_file = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\blender_export\frame_0000.obj")
    
    if obj_file.exists():
        try:
            # For Blender 4.x
            bpy.ops.wm.obj_import(filepath=str(obj_file))
            print("SUCCESS: Imported with Blender 4.x API")
        except:
            try:
                # For Blender 3.x
                bpy.ops.import_scene.obj(filepath=str(obj_file))
                print("SUCCESS: Imported with Blender 3.x API")
            except Exception as e:
                print(f"ERROR: {e}")
                
        # Get imported object and rename
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0] 
            obj.name = "HumanMesh"
            print(f"Imported: {obj.name}")
    else:
        print("ERROR: OBJ file not found")

if __name__ == "__main__":
    simple_import()
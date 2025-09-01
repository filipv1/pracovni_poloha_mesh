import bpy
from pathlib import Path

def import_trunk_vector_sequence():
    """Trunk vector sequence import - step by step animation"""
    
    print("STARTING TRUNK VECTOR SEQUENCE IMPORT...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\trunk_vector_export")
    obj_files = sorted(base_dir.glob("trunk_vector_*.obj"))
    
    print(f"Found {len(obj_files)} trunk vector OBJ files")
    
    all_objects = []
    
    # Create material for trunk vectors
    trunk_material = bpy.data.materials.new(name="TrunkVectorMaterial")
    trunk_material.use_nodes = True
    bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (1.0, 0.3, 0.2, 1.0)  # Orange-red color
    bsdf.inputs[12].default_value = 0.1  # Low roughness for shiny look
    bsdf.inputs[15].default_value = 1.0  # Full transmission for metallic look
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importing trunk vector frame {frame_idx}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"TrunkVector_{frame_idx:04d}"
            all_objects.append(obj)
            
            # Assign material
            if obj.data.materials:
                obj.data.materials[0] = trunk_material
            else:
                obj.data.materials.append(trunk_material)
            
            # Hide all objects except first
            if frame_idx > 0:
                obj.hide_viewport = True
                obj.hide_render = True
            
            print(f"  OK: {obj.name}")
        else:
            print(f"  ERROR: No trunk vector object imported")
    
    print(f"\nSETTING UP TRUNK VECTOR ANIMATION...")
    
    # Setup keyframes for visibility
    for frame_idx, obj in enumerate(all_objects):
        # Set current frame
        bpy.context.scene.frame_set(1)
        
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
    
    # Add lighting for better visualization
    print("Setting up lighting...")
    
    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(2, 2, 5))
    sun_light = bpy.context.object
    sun_light.data.energy = 3.0
    
    # Add area light for fill
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 3))
    area_light = bpy.context.object
    area_light.data.energy = 5.0
    area_light.data.size = 2.0
    
    # Set camera position for good trunk view
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (1.5, -1.5, 1.0)
        camera.rotation_euler = (1.1, 0, 0.785)
    
    print(f"TRUNK VECTOR ANIMATION READY!")
    print(f"Timeline: 1-{len(all_objects)} frames")
    print(f"Press SPACEBAR to play trunk posture animation")
    print(f"Orange-red arrows show trunk orientation changes over time")
    print(f"Vector goes from lumbar spine to cervical spine")
    print(f"Use mouse to rotate view and analyze trunk motion patterns")

if __name__ == "__main__":
    import_trunk_vector_sequence()
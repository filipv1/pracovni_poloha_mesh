import bpy
from pathlib import Path

def import_trunk_analysis_sequence():
    """Import trunk analysis with proper object separation"""
    
    print("SPOUSTIM IMPORT TRUNK ANALYZY...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\trunk_analysis_export")
    obj_files = sorted(base_dir.glob("trunk_analysis_*.obj"))
    
    print(f"Nalezeno {len(obj_files)} souboru trunk analyzy")
    
    all_objects = []
    
    # Create materials
    trunk_material = bpy.data.materials.new(name="TrunkVector_Enhanced")
    trunk_material.use_nodes = True
    trunk_bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.15, 0.15, 1.0)  # RED
    trunk_bsdf.inputs[12].default_value = 0.1
    trunk_bsdf.inputs[15].default_value = 0.8
    
    gravity_material = bpy.data.materials.new(name="GravityReference_Enhanced")
    gravity_material.use_nodes = True
    gravity_bsdf = gravity_material.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.1, 0.4, 1.0, 1.0)  # BLUE
    gravity_bsdf.inputs[12].default_value = 0.2
    
    angle_material = bpy.data.materials.new(name="AngleArc_Enhanced")
    angle_material.use_nodes = True
    angle_bsdf = angle_material.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.2, 1.0, 0.2, 1.0)  # GREEN
    angle_bsdf.inputs[12].default_value = 0.4
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importuji snimek {frame_idx}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"TrunkAnalysis_{frame_idx:04d}"
            all_objects.append(obj)
            
            # Assign trunk material (red)
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
            print(f"  CHYBA: Zadne objekty nebyly importovany")
    
    print(f"\nNASTAVUJI ANIMACI...")
    
    # Setup keyframes for visibility - OPRAVENA LOGIKA
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
    
    # Add lighting
    bpy.ops.object.light_add(type='SUN', location=(2, 2, 5))
    sun_light = bpy.context.object
    sun_light.data.energy = 3.0
    
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 3))
    area_light = bpy.context.object
    area_light.data.energy = 5.0
    area_light.data.size = 2.0
    
    # Set camera position
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (1.5, -1.5, 1.0)
        camera.rotation_euler = (1.1, 0, 0.785)
    
    print(f"ANIMACE PRIPRAVENA!")
    print(f"Timeline: 1-{len(all_objects)} snimku")
    print(f"MEZERA = spustit/zastavit animaci")
    print(f"CERVENA = Trunk vektor + MODRA = Gravitace + ZELENA = Uhel")

if __name__ == "__main__":
    import_trunk_analysis_sequence()
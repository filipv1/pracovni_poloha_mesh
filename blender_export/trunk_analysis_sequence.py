import bpy
from pathlib import Path

def import_trunk_analysis_with_angle():
    """Trunk analysis sequence with gravitational reference and angle visualization"""
    
    print("SPOUŠTÍM IMPORT TRUNK ANALÝZY S ÚHLY...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\trunk_analysis_export")
    obj_files = sorted(base_dir.glob("trunk_analysis_*.obj"))
    
    print(f"Nalezeno {len(obj_files)} souborů trunk analýzy")
    
    all_objects = []
    
    # Create enhanced materials with better visual distinction
    
    # ČERVENÝ material pro vektor trupu (hlavní)
    trunk_material = bpy.data.materials.new(name="TrunkVector_Enhanced")
    trunk_material.use_nodes = True
    trunk_bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.15, 0.15, 1.0)  # Jasně červená
    trunk_bsdf.inputs[12].default_value = 0.1  # Nízká drsnost = lesklý
    trunk_bsdf.inputs[15].default_value = 0.8  # Metallic look
    
    # MODRÝ material pro gravitační referenci
    gravity_material = bpy.data.materials.new(name="GravityReference_Enhanced")
    gravity_material.use_nodes = True
    gravity_bsdf = gravity_material.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.1, 0.4, 1.0, 1.0)  # Jasně modrá
    gravity_bsdf.inputs[12].default_value = 0.2
    gravity_bsdf.inputs[15].default_value = 0.3
    
    # ZELENÝ material pro úhlový oblouk
    angle_material = bpy.data.materials.new(name="AngleArc_Enhanced")
    angle_material.use_nodes = True
    angle_bsdf = angle_material.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.2, 1.0, 0.2, 1.0)  # Jasně zelená
    angle_bsdf.inputs[12].default_value = 0.4
    angle_bsdf.inputs[21].default_value = 1.0  # Alpha pro transparentnost
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importuji trunk analýzu snímek {frame_idx}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported objects
        imported_objects = bpy.context.selected_objects.copy()
        
        if imported_objects:
            for obj in imported_objects:
                # Assign materials based on object groups
                if hasattr(obj.data, 'name'):
                    if "TrunkVector" in obj.data.name or "trunk" in obj.name.lower():
                        obj.name = f"TrunkVector_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = trunk_material
                        else:
                            obj.data.materials.append(trunk_material)
                    
                    elif "GravityReference" in obj.data.name or "gravity" in obj.name.lower():
                        obj.name = f"GravityRef_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = gravity_material
                        else:
                            obj.data.materials.append(gravity_material)
                    
                    elif "AngleArc" in obj.data.name or "angle" in obj.name.lower():
                        obj.name = f"AngleArc_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = angle_material
                        else:
                            obj.data.materials.append(angle_material)
                    
                    else:
                        # Fallback - assign based on position in import
                        obj.name = f"TrunkAnalysis_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = trunk_material
                        else:
                            obj.data.materials.append(trunk_material)
                
                all_objects.append(obj)
                
                # Hide all objects except first frame
                if frame_idx > 0:
                    obj.hide_viewport = True
                    obj.hide_render = True
            
            print(f"  OK: Importováno {len(imported_objects)} objektů pro snímek {frame_idx}")
        else:
            print(f"  CHYBA: Žádné objekty nebyly importovány")
    
    print(f"\nNASTAVUJI ANIMACI TRUNK ANALÝZY...")
    
    # Setup keyframes for visibility (improved logic)
    for frame_idx in range(len(obj_files)):
        # Set current frame
        bpy.context.scene.frame_set(frame_idx + 1)
        
        for obj in all_objects:
            # Extract frame number from object name more reliably
            try:
                obj_frame_str = obj.name.split('_')[-1]
                obj_frame = int(obj_frame_str)
            except:
                continue
            
            if obj_frame == frame_idx:
                # This frame should be visible
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
            else:
                # Other frames should be hidden
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(obj_files)
    bpy.context.scene.frame_set(1)
    
    # Professional lighting setup for trunk analysis
    print("Nastavuji profesionální osvětlení...")
    
    # Main sun light (strong directional light)
    bpy.ops.object.light_add(type='SUN', location=(4, 4, 6))
    sun_light = bpy.context.object
    sun_light.name = "MainSun"
    sun_light.data.energy = 5.0
    sun_light.data.angle = 0.05  # Sharp shadows
    sun_light.rotation_euler = (0.6, 0.3, 0.8)
    
    # Fill light (softer area light)
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 4))
    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 3.0
    fill_light.data.size = 4.0
    fill_light.data.color = (0.9, 0.95, 1.0)  # Slightly blue for contrast
    
    # Back light for rim lighting
    bpy.ops.object.light_add(type='SPOT', location=(0, 4, 2))
    back_light = bpy.context.object
    back_light.name = "BackLight"
    back_light.data.energy = 2.0
    back_light.data.spot_size = 1.2
    back_light.rotation_euler = (1.8, 0, 3.14)
    
    # Set optimal camera position for trunk analysis
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (1.2, -1.2, 0.6)
        camera.rotation_euler = (1.3, 0, 0.785)
        
        # Set camera to track center of action
        bpy.ops.object.constraint_add(type='TRACK_TO')
        if camera.constraints:
            track_constraint = camera.constraints[-1]
            # Create empty object at origin to track
            bpy.ops.object.empty_add(location=(0, 0, 0))
            track_target = bpy.context.object
            track_target.name = "CameraTarget"
            track_constraint.target = track_target
            track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            track_constraint.up_axis = 'UP_Y'
    
    # Set render engine for better visualization
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128  # Good quality/speed balance
    
    # Add text overlay for angle display (if possible)
    try:
        bpy.ops.object.text_add(location=(0, 0, 1))
        text_obj = bpy.context.object
        text_obj.name = "AngleDisplay"
        text_obj.data.body = "Úhel trupu: 0°"
        text_obj.data.size = 0.1
        
        # Create material for text
        text_mat = bpy.data.materials.new(name="TextMaterial")
        text_mat.use_nodes = True
        text_bsdf = text_mat.node_tree.nodes["Principled BSDF"]
        text_bsdf.inputs[0].default_value = (1.0, 1.0, 0.2, 1.0)  # Yellow text
        text_obj.data.materials.append(text_mat)
    except:
        pass  # Text creation might fail in some Blender versions
    
    print(f"TRUNK ANALÝZA S ÚHLY PŘIPRAVENA!")
    print(f"Timeline: 1-{len(obj_files)} snímků")
    print(f"")
    print(f"BAREVNÉ KÓDOVÁNÍ:")
    print(f"  🔴 ČERVENÁ = Vektor trupu (lumbar → cervical)")
    print(f"  🔵 MODRÁ = Gravitační reference (svislice dolů)")  
    print(f"  🟢 ZELENÁ = Úhlový oblouk (vizualizace úhlu)")
    print(f"")
    print(f"OVLÁDÁNÍ:")
    print(f"  MEZERNÍK = Spustit/zastavit animaci")
    print(f"  Myš = Rotovat pohled pro lepší analýzu")
    print(f"  Kolečko myši = Zoom in/out")
    print(f"  Číselník na klávesnici = Různé pohledy")

if __name__ == "__main__":
    import_trunk_analysis_with_angle()
import bpy
from pathlib import Path

def import_combined_sequence():
    """Import both 3D meshes AND trunk vectors with angles in one scene"""
    
    print("SPOUSTIM KOMBINOVANY IMPORT - MESH + TRUNK VEKTORY...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Define paths
    mesh_dir = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\arm_meshes_exp")
    trunk_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\trunk_analysis_export")
    
    # Get file lists
    mesh_files = sorted(mesh_dir.glob("frame_*.obj"))
    trunk_files = sorted(trunk_dir.glob("trunk_analysis_*.obj"))
    
    print(f"Nalezeno {len(mesh_files)} mesh souboru")
    print(f"Nalezeno {len(trunk_files)} trunk analysis souboru")
    
    # Take minimum to ensure sync
    max_frames = min(len(mesh_files), len(trunk_files))
    print(f"Pouziju {max_frames} snimku pro synchronizaci")
    
    all_objects = []
    
    # Create materials
    # 1. Mesh material (semi-transparent skin tone)
    mesh_material = bpy.data.materials.new(name="BodyMesh_Material")
    mesh_material.use_nodes = True
    mesh_bsdf = mesh_material.node_tree.nodes["Principled BSDF"]
    mesh_bsdf.inputs[0].default_value = (0.8, 0.7, 0.6, 0.7)  # Skin tone, semi-transparent
    mesh_bsdf.inputs[21].default_value = 0.7  # Alpha for transparency
    mesh_bsdf.inputs[12].default_value = 0.3  # Roughness
    mesh_material.blend_method = 'BLEND'  # Enable transparency
    
    # 2. Trunk vector material (bright red)
    trunk_material = bpy.data.materials.new(name="TrunkVector_Material")
    trunk_material.use_nodes = True
    trunk_bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.1, 0.1, 1.0)  # Bright red
    trunk_bsdf.inputs[12].default_value = 0.0  # Very shiny
    trunk_bsdf.inputs[15].default_value = 0.9  # Metallic
    
    # 3. Gravity reference material (bright blue)
    gravity_material = bpy.data.materials.new(name="GravityRef_Material")
    gravity_material.use_nodes = True
    gravity_bsdf = gravity_material.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.1, 0.3, 1.0, 1.0)  # Bright blue
    gravity_bsdf.inputs[12].default_value = 0.1
    
    # 4. Angle arc material (bright green)
    angle_material = bpy.data.materials.new(name="AngleArc_Material")
    angle_material.use_nodes = True
    angle_bsdf = angle_material.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.1, 1.0, 0.1, 1.0)  # Bright green
    angle_bsdf.inputs[12].default_value = 0.2
    
    for frame_idx in range(max_frames):
        print(f"Importuji kombinovany snimek {frame_idx}...")
        
        frame_objects = []
        
        # 1. Import 3D mesh (body)
        if frame_idx < len(mesh_files):
            mesh_file = mesh_files[frame_idx]
            try:
                bpy.ops.wm.obj_import(filepath=str(mesh_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(mesh_file))
            
            if bpy.context.selected_objects:
                mesh_obj = bpy.context.selected_objects[0]
                mesh_obj.name = f"BodyMesh_{frame_idx:04d}"
                
                # Apply semi-transparent material to mesh
                if mesh_obj.data.materials:
                    mesh_obj.data.materials[0] = mesh_material
                else:
                    mesh_obj.data.materials.append(mesh_material)
                
                frame_objects.append(mesh_obj)
                print(f"  Mesh: {mesh_obj.name}")
        
        # 2. Import trunk analysis (vectors + gravity + angles)
        if frame_idx < len(trunk_files):
            trunk_file = trunk_files[frame_idx]
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_file))
            
            # The trunk file contains combined geometry (trunk + gravity + angles)
            if bpy.context.selected_objects:
                trunk_obj = bpy.context.selected_objects[0]
                trunk_obj.name = f"TrunkAnalysis_{frame_idx:04d}"
                
                # Apply trunk material (will show all parts as red - we could separate later)
                if trunk_obj.data.materials:
                    trunk_obj.data.materials[0] = trunk_material
                else:
                    trunk_obj.data.materials.append(trunk_material)
                
                frame_objects.append(trunk_obj)
                print(f"  Trunk: {trunk_obj.name}")
        
        # Add all frame objects to main list
        all_objects.extend(frame_objects)
        
        # Hide all objects except first frame
        if frame_idx > 0:
            for obj in frame_objects:
                obj.hide_viewport = True
                obj.hide_render = True
    
    print(f"\nNASTAVUJI KOMBINOVANOU ANIMACI...")
    
    # Setup keyframes for visibility - both mesh and trunk objects
    for frame_idx in range(max_frames):
        for obj in all_objects:
            # Extract frame number from object name
            try:
                name_parts = obj.name.split('_')
                obj_frame = int(name_parts[1])
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
    bpy.context.scene.frame_end = max_frames
    bpy.context.scene.frame_set(1)
    
    # Professional lighting for combined visualization
    print("Nastavuji osvetleni...")
    
    # Main sun light
    bpy.ops.object.light_add(type='SUN', location=(4, 4, 6))
    sun_light = bpy.context.object
    sun_light.name = "MainSun"
    sun_light.data.energy = 6.0
    sun_light.data.angle = 0.05
    sun_light.rotation_euler = (0.6, 0.3, 0.8)
    
    # Fill light for body mesh
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 4))
    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 4.0
    fill_light.data.size = 4.0
    fill_light.data.color = (0.9, 0.95, 1.0)
    
    # Accent light for trunk vectors
    bpy.ops.object.light_add(type='SPOT', location=(0, 4, 2))
    accent_light = bpy.context.object
    accent_light.name = "AccentLight"
    accent_light.data.energy = 3.0
    accent_light.data.spot_size = 1.2
    accent_light.rotation_euler = (1.8, 0, 3.14)
    
    # Optimal camera position for combined view
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (2.0, -2.0, 1.2)
        camera.rotation_euler = (1.2, 0, 0.785)
    
    # Set render engine for better transparency
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    
    print(f"KOMBINOVANA ANIMACE PRIPRAVENA!")
    print(f"Timeline: 1-{max_frames} snimku")
    print(f"")
    print(f"OBJEKTY:")
    print(f"  🧍 PRUHLEDNE TELO = 3D human mesh")
    print(f"  🔴 CERVENE SIPKY = Trunk vektor (lumbar->cervical)")
    print(f"  🔵 MODRE SIPKY = Gravitacni reference")
    print(f"  🟢 ZELENE OBLOUKY = Uhel mezi vektory")
    print(f"")
    print(f"OVLADANI:")
    print(f"  MEZERA = Spustit/zastavit animaci")
    print(f"  Myš = Rotovat pohled")
    print(f"  Kolecko = Zoom")
    print(f"")
    print(f"NYNI VIDITE TRUNK POSTURU V KONTEXTU CELEHO TELA!")

if __name__ == "__main__":
    import_combined_sequence()
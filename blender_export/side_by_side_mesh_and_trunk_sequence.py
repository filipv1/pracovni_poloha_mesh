import bpy
from pathlib import Path

def import_side_by_side_sequence():
    """Import 3D mesh and trunk vectors side by side for better visualization"""
    
    print("SPOUSTIM SIDE-BY-SIDE VIZUALIZACI - MESH | TRUNK VEKTORY...")
    
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
    
    # Define spatial offsets for side-by-side layout
    MESH_OFFSET = (-1.0, 0.0, 0.0)      # Mesh vlevo
    TRUNK_OFFSET = (1.0, 0.0, 0.0)      # Trunk vektory vpravo
    SCALE_TRUNK = 3.0                    # Zvetseni trunk vektoru pro lepsi viditelnost
    
    # Create materials
    # 1. Mesh material (solid skin tone)
    mesh_material = bpy.data.materials.new(name="BodyMesh_SideBySide")
    mesh_material.use_nodes = True
    mesh_bsdf = mesh_material.node_tree.nodes["Principled BSDF"]
    mesh_bsdf.inputs[0].default_value = (0.9, 0.8, 0.7, 1.0)  # Skin tone, solid
    mesh_bsdf.inputs[12].default_value = 0.4  # Roughness
    mesh_bsdf.inputs[15].default_value = 0.1  # Slight metallic
    
    # 2. Trunk vector material (bright red, larger)
    trunk_material = bpy.data.materials.new(name="TrunkVector_SideBySide")
    trunk_material.use_nodes = True
    trunk_bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.1, 0.1, 1.0)  # Bright red
    trunk_bsdf.inputs[12].default_value = 0.0  # Very shiny
    trunk_bsdf.inputs[15].default_value = 0.9  # Metallic
    
    # 3. Gravity reference material (bright blue)
    gravity_material = bpy.data.materials.new(name="GravityRef_SideBySide")
    gravity_material.use_nodes = True
    gravity_bsdf = gravity_material.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.1, 0.5, 1.0, 1.0)  # Bright blue
    gravity_bsdf.inputs[12].default_value = 0.1
    
    # 4. Angle arc material (bright green)
    angle_material = bpy.data.materials.new(name="AngleArc_SideBySide")
    angle_material.use_nodes = True
    angle_bsdf = angle_material.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.2, 1.0, 0.2, 1.0)  # Bright green
    angle_bsdf.inputs[12].default_value = 0.2
    
    for frame_idx in range(max_frames):
        print(f"Importuji side-by-side snimek {frame_idx}...")
        
        frame_objects = []
        
        # 1. Import 3D mesh (body) - LEFT SIDE
        if frame_idx < len(mesh_files):
            mesh_file = mesh_files[frame_idx]
            try:
                bpy.ops.wm.obj_import(filepath=str(mesh_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(mesh_file))
            
            if bpy.context.selected_objects:
                mesh_obj = bpy.context.selected_objects[0]
                mesh_obj.name = f"BodyMesh_{frame_idx:04d}"
                
                # Move mesh to left side
                mesh_obj.location = MESH_OFFSET
                
                # Apply solid material to mesh
                if mesh_obj.data.materials:
                    mesh_obj.data.materials[0] = mesh_material
                else:
                    mesh_obj.data.materials.append(mesh_material)
                
                frame_objects.append(mesh_obj)
                print(f"  Mesh (vlevo): {mesh_obj.name}")
        
        # 2. Import trunk analysis - RIGHT SIDE (scaled up)
        if frame_idx < len(trunk_files):
            trunk_file = trunk_files[frame_idx]
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_file))
            
            if bpy.context.selected_objects:
                trunk_obj = bpy.context.selected_objects[0]
                trunk_obj.name = f"TrunkAnalysis_{frame_idx:04d}"
                
                # Move trunk vectors to right side and scale up
                trunk_obj.location = TRUNK_OFFSET
                trunk_obj.scale = (SCALE_TRUNK, SCALE_TRUNK, SCALE_TRUNK)
                
                # Apply bright trunk material
                if trunk_obj.data.materials:
                    trunk_obj.data.materials[0] = trunk_material
                else:
                    trunk_obj.data.materials.append(trunk_material)
                
                frame_objects.append(trunk_obj)
                print(f"  Trunk (vpravo, 3x): {trunk_obj.name}")
        
        # Add all frame objects to main list
        all_objects.extend(frame_objects)
        
        # Hide all objects except first frame
        if frame_idx > 0:
            for obj in frame_objects:
                obj.hide_viewport = True
                obj.hide_render = True
    
    print(f"\nNASTAVUJI SIDE-BY-SIDE ANIMACI...")
    
    # Setup keyframes for visibility
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
    
    # Create visual separators and labels
    print("Pridavam vizualni separatory a popisky...")
    
    # Add visual separator line
    bpy.ops.mesh.primitive_cube_add(size=0.02, location=(0, 0, 0))
    separator = bpy.context.object
    separator.name = "Separator_Line"
    separator.scale = (1, 1, 25)  # Tall thin line
    
    # Separator material
    sep_mat = bpy.data.materials.new(name="Separator_Material")
    sep_mat.use_nodes = True
    sep_bsdf = sep_mat.node_tree.nodes["Principled BSDF"]
    sep_bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)  # Gray
    separator.data.materials.append(sep_mat)
    
    # Add text labels
    try:
        # Left side label
        bpy.ops.object.text_add(location=(-1.0, 0, 1.5))
        left_label = bpy.context.object
        left_label.name = "Label_Mesh"
        left_label.data.body = "3D MESH"
        left_label.data.size = 0.15
        left_label.data.align_x = 'CENTER'
        
        # Right side label
        bpy.ops.object.text_add(location=(1.0, 0, 1.5))
        right_label = bpy.context.object
        right_label.name = "Label_Trunk"
        right_label.data.body = "TRUNK VECTORS"
        right_label.data.size = 0.15
        right_label.data.align_x = 'CENTER'
        
        # Text material
        text_mat = bpy.data.materials.new(name="Text_Material")
        text_mat.use_nodes = True
        text_bsdf = text_mat.node_tree.nodes["Principled BSDF"]
        text_bsdf.inputs[0].default_value = (1.0, 1.0, 0.2, 1.0)  # Yellow
        
        left_label.data.materials.append(text_mat)
        right_label.data.materials.append(text_mat)
        
    except:
        print("  Text labels nelze pridat v teto verzi Blenderu")
    
    # Professional lighting for side-by-side layout
    print("Nastavuji optimalni osvetleni...")
    
    # Main sun light (center, high)
    bpy.ops.object.light_add(type='SUN', location=(0, 4, 8))
    sun_light = bpy.context.object
    sun_light.name = "MainSun_Center"
    sun_light.data.energy = 8.0
    sun_light.data.angle = 0.1
    sun_light.rotation_euler = (0.3, 0, 0)
    
    # Left side light (for mesh)
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 3))
    left_light = bpy.context.object
    left_light.name = "MeshLight_Left"
    left_light.data.energy = 4.0
    left_light.data.size = 3.0
    left_light.data.color = (0.9, 0.9, 1.0)  # Cool white
    
    # Right side light (for trunk vectors)
    bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
    right_light = bpy.context.object
    right_light.name = "TrunkLight_Right"
    right_light.data.energy = 6.0
    right_light.data.size = 2.0
    right_light.data.color = (1.0, 0.9, 0.9)  # Warm white
    
    # Back fill light
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 1))
    back_light = bpy.context.object
    back_light.name = "FillLight_Back"
    back_light.data.energy = 2.0
    back_light.data.size = 4.0
    back_light.rotation_euler = (3.14, 0, 0)
    
    # Optimal camera position for side-by-side view
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (0, -4.0, 1.5)  # Center, back, slightly up
        camera.rotation_euler = (1.4, 0, 0)  # Look down slightly
        
        # Wider field of view to capture both sides
        camera.data.lens = 28  # Wide angle lens
    
    # Set render engine for better materials
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    
    # Add world lighting for better visibility
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs[1].default_value = 0.3  # World light strength
    
    print(f"SIDE-BY-SIDE VIZUALIZACE PRIPRAVENA!")
    print(f"Timeline: 1-{max_frames} snimku")
    print(f"")
    print(f"ROZMISTENI:")
    print(f"  👤 VLEVO = 3D human mesh (naturalni velikost)")
    print(f"  🎯 VPRAVO = Trunk vektory (3x zvetseno)")
    print(f"  📏 STRED = Oddelovaci cara")
    print(f"")
    print(f"BAREVNE KODOVANI VPRAVO:")
    print(f"  🔴 CERVENE SIPKY = Trunk vektor (lumbar->cervical)")
    print(f"  🔵 MODRE SIPKY = Gravitacni reference")
    print(f"  🟢 ZELENE OBLOUKY = Uhel mezi vektory")
    print(f"")
    print(f"OVLADANI:")
    print(f"  MEZERA = Spustit/zastavit animaci")
    print(f"  Myš = Rotovat pohled")
    print(f"  Kolecko = Zoom")
    print(f"  Tab = Wireframe/Solid")
    print(f"")
    print(f"OPTIMALNI PRO ANALYZU POSTURY BEZ PREKRYVANI!")

if __name__ == "__main__":
    import_side_by_side_sequence()
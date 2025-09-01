import bpy
from pathlib import Path

def import_side_by_side_arm_and_trunk_sequence():
    """Import 3D mesh and arm+trunk vectors side by side for arm analysis visualization"""
    
    print("SPOUSTIM SIDE-BY-SIDE VIZUALIZACI - MESH | ARM + TRUNK VEKTORY...")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Define paths
    mesh_dir = Path(r"C:\Users\vaclavik\test9\pracovni_poloha_mesh\arm_meshes_exp")
    arm_analysis_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\arm_analysis_export")
    
    # Get file lists
    mesh_files = sorted(mesh_dir.glob("frame_*.obj"))
    arm_files = sorted(arm_analysis_dir.glob("arm_analysis_*.obj"))
    
    print(f"Nalezeno {len(mesh_files)} mesh souboru")
    print(f"Nalezeno {len(arm_files)} arm analysis souboru")
    
    # Take minimum to ensure sync
    max_frames = min(len(mesh_files), len(arm_files))
    print(f"Pouziju {max_frames} snimku pro synchronizaci")
    
    all_objects = []
    
    # Define spatial offsets for side-by-side layout
    MESH_OFFSET = (-1.5, 0.0, 0.0)      # Mesh vlevo
    ARM_OFFSET = (1.5, 0.0, 0.0)        # Arm vektory vpravo
    SCALE_VECTORS = 2.5                  # Zvetseni vektoru pro lepsi viditelnost
    
    # Create basic materials
    # 1. Mesh material
    mesh_material = bpy.data.materials.new(name="BodyMesh_SideBySide")
    mesh_material.use_nodes = True
    mesh_bsdf = mesh_material.node_tree.nodes["Principled BSDF"]
    mesh_bsdf.inputs[0].default_value = (0.8, 0.7, 0.6, 1.0)  # Skin tone
    
    # 2. Trunk vector material (RED)
    trunk_material = bpy.data.materials.new(name="TrunkVector_Material")
    trunk_material.use_nodes = True
    trunk_bsdf = trunk_material.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.2, 0.2, 1.0)  # Red
    
    # 3. Gravity reference material (BLUE)
    gravity_material = bpy.data.materials.new(name="GravityRef_Material")
    gravity_material.use_nodes = True
    gravity_bsdf = gravity_material.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.2, 0.5, 1.0, 1.0)  # Blue
    
    # 4. Angle arc material (GREEN)
    angle_material = bpy.data.materials.new(name="AngleArc_Material")
    angle_material.use_nodes = True
    angle_bsdf = angle_material.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.2, 1.0, 0.3, 1.0)  # Green
    
    # 5. Left arm material (YELLOW)
    left_arm_material = bpy.data.materials.new(name="LeftArm_Material")
    left_arm_material.use_nodes = True
    left_arm_bsdf = left_arm_material.node_tree.nodes["Principled BSDF"]
    left_arm_bsdf.inputs[0].default_value = (1.0, 1.0, 0.2, 1.0)  # Yellow
    
    # 6. Right arm material (ORANGE)
    right_arm_material = bpy.data.materials.new(name="RightArm_Material")
    right_arm_material.use_nodes = True
    right_arm_bsdf = right_arm_material.node_tree.nodes["Principled BSDF"]
    right_arm_bsdf.inputs[0].default_value = (1.0, 0.5, 0.1, 1.0)  # Orange
    
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
                
                # Apply mesh material
                if mesh_obj.data.materials:
                    mesh_obj.data.materials[0] = mesh_material
                else:
                    mesh_obj.data.materials.append(mesh_material)
                
                frame_objects.append(mesh_obj)
                print(f"  Mesh (vlevo): {mesh_obj.name}")
        
        # 2. Import arm analysis vectors - RIGHT SIDE (scaled up)
        if frame_idx < len(arm_files):
            arm_file = arm_files[frame_idx]
            try:
                bpy.ops.wm.obj_import(filepath=str(arm_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(arm_file))
            
            # Get all imported objects for this frame
            imported_objects = bpy.context.selected_objects
            
            if imported_objects:
                for obj in imported_objects:
                    # Move all vectors to right side and scale up
                    obj.location = ARM_OFFSET
                    obj.scale = (SCALE_VECTORS, SCALE_VECTORS, SCALE_VECTORS)
                    
                    # Apply materials based on group names
                    if "TrunkVector" in obj.name:
                        obj.name = f"TrunkVector_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = trunk_material
                        else:
                            obj.data.materials.append(trunk_material)
                    elif "GravityReference" in obj.name:
                        obj.name = f"GravityRef_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = gravity_material
                        else:
                            obj.data.materials.append(gravity_material)
                    elif "AngleArc" in obj.name:
                        obj.name = f"AngleArc_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = angle_material
                        else:
                            obj.data.materials.append(angle_material)
                    elif "LeftArmVector" in obj.name:
                        obj.name = f"LeftArm_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = left_arm_material
                        else:
                            obj.data.materials.append(left_arm_material)
                    elif "RightArmVector" in obj.name:
                        obj.name = f"RightArm_{frame_idx:04d}"
                        if obj.data.materials:
                            obj.data.materials[0] = right_arm_material
                        else:
                            obj.data.materials.append(right_arm_material)
                    
                    frame_objects.append(obj)
                
                print(f"  Arm analysis (vpravo, {SCALE_VECTORS}x): {len(imported_objects)} vektoru")
        
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
                obj_frame = int(name_parts[-1])
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
    
    # Add visual separator line
    bpy.ops.mesh.primitive_cube_add(size=0.02, location=(0, 0, 0))
    separator = bpy.context.object
    separator.name = "Separator_Line"
    separator.scale = (1, 1, 20)  # Tall thin line
    
    # Basic camera position for side-by-side view
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (0, -4.0, 1.0)  # Center, back, slightly up
        camera.rotation_euler = (1.3, 0, 0)  # Look down slightly
    
    print(f"SIDE-BY-SIDE ARM ANALÝZA PŘIPRAVENA!")
    print(f"Timeline: 1-{max_frames} snímků")
    print(f"")
    print(f"ROZMÍSTĚNÍ:")
    print(f"  👤 VLEVO = 3D human mesh")
    print(f"  🎯 VPRAVO = Všechny vektory ({SCALE_VECTORS}x zvětšeno)")
    print(f"")
    print(f"BAREVNÉ KÓDOVÁNÍ VPRAVO:")
    print(f"  🔴 ČERVENÁ = Trunk vektor (lumbar→cervical)")
    print(f"  🔵 MODRÁ = Gravitační reference (svislice dolů)")
    print(f"  🟢 ZELENÁ = Úhlový oblouk (úhel trupu)")
    print(f"  🟡 ŽLUTÁ = Levá ruka (shoulder→elbow)")
    print(f"  🟠 ORANŽOVÁ = Pravá ruka (shoulder→elbow)")
    print(f"")
    print(f"OVLÁDÁNÍ:")
    print(f"  MEZERNÍK = Spustit/zastavit animaci")
    print(f"  Myš = Rotovat pohled")
    print(f"  Kolečko = Zoom")

if __name__ == "__main__":
    import_side_by_side_arm_and_trunk_sequence()
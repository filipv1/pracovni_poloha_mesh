#!/usr/bin/env python3
"""
ULTRA FAST Blender animation - Mesh swapping technique
Fastest possible loading and playback
"""

import bpy
from pathlib import Path
import time

print("\n" + "=" * 60)
print("ULTRA FAST MESH + VECTORS ANIMATION")
print("=" * 60)

# PERFORMANCE SETTINGS
FRAME_SKIP = 2  # Load every 2nd frame for 2x faster loading
MAX_FRAMES = 100  # Load only first 100 frames for testing (set None for all)
USE_SIMPLE_SHADING = True  # Use simple shading for better performance

start_time = time.time()

# Clear everything
print("Clearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Clear all data blocks
for block in bpy.data.meshes:
    if block.users == 0:
        bpy.data.meshes.remove(block)
for block in bpy.data.materials:
    if block.users == 0:
        bpy.data.materials.remove(block)

# Set base directory
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")

if not base_dir.exists():
    print(f"ERROR: Directory not found: {base_dir}")
else:
    # Find all frame files
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    total_available = len(frame_files)
    
    # Apply skip and limit
    indices_to_load = list(range(0, total_available, FRAME_SKIP))
    if MAX_FRAMES:
        indices_to_load = indices_to_load[:MAX_FRAMES]
    
    num_frames = len(indices_to_load)
    
    print(f"Found {total_available} total frames")
    print(f"Loading {num_frames} frames (skip={FRAME_SKIP}, max={MAX_FRAMES})")
    
    # CREATE SINGLE SET OF OBJECTS WITH SHARED MATERIALS
    print("Creating display objects...")
    
    # Create materials once
    materials = {}
    
    if USE_SIMPLE_SHADING:
        # Super simple materials for maximum performance
        mat_mesh = bpy.data.materials.new(name="Fast_Mesh")
        mat_mesh.use_nodes = False
        mat_mesh.diffuse_color = (0.8, 0.8, 0.8, 0.3)  # 30% alpha
        mat_mesh.blend_method = 'BLEND'
        materials['mesh'] = mat_mesh
        
        # Vector materials
        colors = {
            'trunk': (1.0, 0.2, 0.2, 1.0),
            'neck': (0.2, 0.2, 1.0, 1.0),
            'left_arm': (0.2, 1.0, 0.2, 1.0),
            'right_arm': (1.0, 1.0, 0.2, 1.0)
        }
        
        for name, color in colors.items():
            mat = bpy.data.materials.new(name=f"Fast_{name}")
            mat.use_nodes = False
            mat.diffuse_color = color
            materials[name] = mat
    else:
        # Node-based materials (from previous version)
        mat_mesh = bpy.data.materials.new(name="Mesh_Mat")
        mat_mesh.use_nodes = True
        mat_mesh.blend_method = 'BLEND'
        bsdf = mat_mesh.node_tree.nodes.get("Principled BSDF")
        if bsdf and "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.3
        materials['mesh'] = mat_mesh
        
        # Create vector materials
        for name, color in [('trunk', (1,0.2,0.2)), ('neck', (0.2,0.2,1)), 
                           ('left_arm', (0.2,1,0.2)), ('right_arm', (1,1,0.2))]:
            mat = bpy.data.materials.new(name=f"Vec_{name}")
            mat.use_nodes = True
            materials[name] = mat
    
    # Pre-load all mesh data into memory (faster switching)
    print("Pre-loading mesh data...")
    mesh_data_cache = {
        'mesh': [],
        'trunk': [],
        'neck': [],
        'left_arm': [],
        'right_arm': []
    }
    
    for i, frame_idx in enumerate(indices_to_load):
        if i % 20 == 0:
            print(f"  Loading mesh data {i+1}/{num_frames}...")
        
        # Load each object type
        for obj_type, cache_key in [('frame', 'mesh'), ('trunk', 'trunk'), 
                                    ('neck', 'neck'), ('left_arm', 'left_arm'), 
                                    ('right_arm', 'right_arm')]:
            
            filepath = base_dir / f"{obj_type}_{frame_idx:04d}.obj"
            if filepath.exists():
                # Import to get mesh data
                try:
                    bpy.ops.wm.obj_import(filepath=str(filepath))
                except:
                    bpy.ops.import_scene.obj(filepath=str(filepath))
                
                if bpy.context.selected_objects:
                    temp_obj = bpy.context.selected_objects[0]
                    # Store mesh data
                    mesh_data_cache[cache_key].append(temp_obj.data.copy())
                    # Delete temporary object
                    bpy.data.objects.remove(temp_obj)
            else:
                mesh_data_cache[cache_key].append(None)
    
    print(f"Loaded {sum(len(v) for v in mesh_data_cache.values())} mesh data blocks")
    
    # Create single set of display objects
    print("Creating display objects...")
    display_objects = {}
    
    # Mesh object
    if mesh_data_cache['mesh'] and mesh_data_cache['mesh'][0]:
        mesh_obj = bpy.data.objects.new("Display_Mesh", mesh_data_cache['mesh'][0])
        bpy.context.scene.collection.objects.link(mesh_obj)
        mesh_obj.data.materials.append(materials['mesh'])
        display_objects['mesh'] = mesh_obj
    
    # Vector objects
    for vec_type in ['trunk', 'neck', 'left_arm', 'right_arm']:
        if mesh_data_cache[vec_type] and mesh_data_cache[vec_type][0]:
            vec_obj = bpy.data.objects.new(f"Display_{vec_type}", mesh_data_cache[vec_type][0])
            bpy.context.scene.collection.objects.link(vec_obj)
            vec_obj.data.materials.append(materials[vec_type])
            vec_obj.show_in_front = True
            display_objects[vec_type] = vec_obj
    
    print("Setting up mesh swapping animation...")
    
    # Create frame change handler for mesh swapping
    def frame_change_handler(scene):
        frame = scene.frame_current - 1
        if frame < 0 or frame >= len(mesh_data_cache['mesh']):
            return
        
        # Swap mesh data for each object
        for obj_type, obj in display_objects.items():
            if obj and frame < len(mesh_data_cache[obj_type]):
                new_mesh = mesh_data_cache[obj_type][frame]
                if new_mesh:
                    obj.data = new_mesh
    
    # Register frame change handler
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_handler)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(1)
    
    # Set viewport for performance
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if USE_SIMPLE_SHADING:
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'MATERIAL'
                    else:
                        space.shading.type = 'MATERIAL'
                    
                    # Disable overlays for performance
                    space.overlay.show_floor = False
                    space.overlay.show_axis_x = False
                    space.overlay.show_axis_y = False
                    space.overlay.show_cursor = False
                    space.overlay.show_object_origins = False
                    space.overlay.show_relationship_lines = False
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("ULTRA FAST IMPORT COMPLETE!")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Frames loaded: {num_frames}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Objects created: {len(display_objects)} (reused)")
    print(f"Mesh data blocks: {sum(len(v) for v in mesh_data_cache.values())}")
    print("\nULTRA OPTIMIZATIONS:")
    print("✓ Single object set with mesh swapping")
    print("✓ Pre-loaded mesh data cache")
    print("✓ Frame handler instead of keyframes")
    print("✓ Simple shading mode" if USE_SIMPLE_SHADING else "✓ Material shading")
    print("✓ Minimal viewport overlays")
    print("\nVECTORS:")
    print("- RED: Trunk | BLUE: Neck | GREEN: Left arm | YELLOW: Right arm")
    print("\nPress SPACEBAR to play (should be 25+ FPS!)")
    print("\nTIP: Set FRAME_SKIP=5 and MAX_FRAMES=50 for instant loading!")
    print("=" * 60)
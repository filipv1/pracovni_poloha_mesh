#!/usr/bin/env python3
"""
OPTIMIZED Blender animation script - much faster loading and playback
Uses shared materials and simplified keyframe strategy
"""

import bpy
from pathlib import Path
import time

print("\n" + "=" * 60)
print("OPTIMIZED BLENDER MESH + VECTORS ANIMATION")
print("=" * 60)

# OPTIMIZATION SETTINGS
FRAME_SKIP = 1  # Set to 2 to load every 2nd frame, 3 for every 3rd, etc.
MAX_FRAMES = None  # Set to 100 to load only first 100 frames, None for all

start_time = time.time()

# Clear everything
print("Clearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Clear unused data blocks to free memory
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
    total_available_frames = len(frame_files)
    
    # Apply frame skip and max frames
    frame_files = frame_files[::FRAME_SKIP]
    if MAX_FRAMES:
        frame_files = frame_files[:MAX_FRAMES]
    
    num_frames = len(frame_files)
    
    print(f"Found {total_available_frames} total frames")
    print(f"Loading {num_frames} frames (skip={FRAME_SKIP})")
    
    # CREATE SHARED MATERIALS ONCE (huge optimization!)
    print("Creating shared materials...")
    
    materials = {}
    
    # Mesh material (semi-transparent)
    mat_mesh = bpy.data.materials.new(name="Shared_Mesh_Material")
    mat_mesh.use_nodes = True
    mat_mesh.blend_method = 'BLEND'
    bsdf = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.3  # 30% opacity
    materials['mesh'] = mat_mesh
    
    # Vector materials (opaque with emission)
    vector_configs = [
        ('trunk', (1.0, 0.2, 0.2)),  # Red
        ('neck', (0.2, 0.2, 1.0)),   # Blue
        ('left_arm', (0.2, 1.0, 0.2)),  # Green
        ('right_arm', (1.0, 1.0, 0.2))  # Yellow
    ]
    
    for name, color in vector_configs:
        mat = bpy.data.materials.new(name=f"Shared_{name}_Material")
        mat.use_nodes = True
        mat.blend_method = 'OPAQUE'
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            # Safe emission setting
            try:
                if "Emission" in bsdf.inputs:
                    bsdf.inputs["Emission"].default_value = (*color, 1.0)
                    if "Emission Strength" in bsdf.inputs:
                        bsdf.inputs["Emission Strength"].default_value = 2.0
            except:
                pass  # Skip emission if not available
        materials[name] = mat
    
    # Create collections for better organization
    print("Creating collections...")
    
    # Remove default collection
    if "Collection" in bpy.data.collections:
        bpy.data.collections.remove(bpy.data.collections["Collection"])
    
    # Create main collection
    main_collection = bpy.data.collections.new("Animation_Frames")
    bpy.context.scene.collection.children.link(main_collection)
    
    # Store frame collections
    frame_collections = []
    
    print("Importing objects...")
    import_start = time.time()
    
    # Import objects frame by frame
    for real_frame_idx, frame_file in enumerate(frame_files):
        # Calculate actual frame number (accounting for skip)
        frame_idx = real_frame_idx * FRAME_SKIP
        
        # Create collection for this frame
        frame_col = bpy.data.collections.new(f"Frame_{frame_idx:04d}")
        main_collection.children.link(frame_col)
        frame_collections.append(frame_col)
        
        # Files to import for this frame
        files_to_import = [
            (f"frame_{frame_idx:04d}.obj", "Mesh", 'mesh'),
            (f"trunk_{frame_idx:04d}.obj", "Trunk", 'trunk'),
            (f"neck_{frame_idx:04d}.obj", "Neck", 'neck'),
            (f"left_arm_{frame_idx:04d}.obj", "LeftArm", 'left_arm'),
            (f"right_arm_{frame_idx:04d}.obj", "RightArm", 'right_arm')
        ]
        
        for filename, obj_type, mat_key in files_to_import:
            filepath = base_dir / filename
            if filepath.exists():
                # Import the file
                try:
                    bpy.ops.wm.obj_import(filepath=str(filepath))
                except:
                    bpy.ops.import_scene.obj(filepath=str(filepath))
                
                if bpy.context.selected_objects:
                    obj = bpy.context.selected_objects[0]
                    obj.name = f"{obj_type}_{frame_idx:04d}"
                    
                    # Move to frame collection
                    if obj.name in bpy.context.scene.collection.objects:
                        bpy.context.scene.collection.objects.unlink(obj)
                    frame_col.objects.link(obj)
                    
                    # Apply SHARED material (huge optimization!)
                    obj.data.materials.clear()
                    obj.data.materials.append(materials[mat_key])
                    
                    # Display settings for vectors
                    if obj_type != "Mesh":
                        obj.show_in_front = True
                        obj.display_type = 'SOLID'
                    else:
                        obj.show_in_front = False
                        obj.display_type = 'SOLID'
        
        # Hide entire collection initially
        frame_col.hide_viewport = True
        frame_col.hide_render = True
        
        if real_frame_idx % 20 == 0:
            elapsed = time.time() - import_start
            rate = (real_frame_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (num_frames - real_frame_idx - 1) / rate if rate > 0 else 0
            print(f"  Imported {real_frame_idx + 1}/{num_frames} frames ({rate:.1f} fps, ETA: {eta:.0f}s)")
    
    print(f"Import completed in {time.time() - import_start:.1f} seconds")
    print("Setting up optimized animation...")
    
    # OPTIMIZED KEYFRAME STRATEGY - use collection visibility
    anim_start = time.time()
    
    for timeline_frame in range(1, num_frames + 1):
        frame_idx = timeline_frame - 1
        
        # Set frame
        bpy.context.scene.frame_set(timeline_frame)
        
        # Use collection visibility (much faster than individual objects!)
        for i, col in enumerate(frame_collections):
            if i == frame_idx:
                col.hide_viewport = False
                col.hide_render = False
            else:
                col.hide_viewport = True
                col.hide_render = True
            
            # Keyframe the collection
            col.id_data.keyframe_insert(
                data_path=f'children["{col.name}"].hide_viewport',
                frame=timeline_frame
            )
            col.id_data.keyframe_insert(
                data_path=f'children["{col.name}"].hide_render',
                frame=timeline_frame
            )
        
        if timeline_frame % 50 == 0:
            print(f"  Keyframes: {timeline_frame}/{num_frames}")
    
    print(f"Animation setup completed in {time.time() - anim_start:.1f} seconds")
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(1)
    
    # Make sure first frame is visible
    frame_collections[0].hide_viewport = False
    frame_collections[0].hide_render = False
    
    # Set viewport to material preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.shading.use_scene_lights = True
                    space.shading.use_scene_world = False
                    
                    # Optimize viewport for performance
                    space.overlay.show_relationship_lines = False
                    space.overlay.show_extras = False
                    space.overlay.show_floor = False
                    space.overlay.show_axis_x = False
                    space.overlay.show_axis_y = False
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("OPTIMIZED IMPORT COMPLETE!")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Total frames: {num_frames}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Materials created: {len(materials)} (shared)")
    print(f"Collections created: {num_frames}")
    print("\nPERFORMANCE IMPROVEMENTS:")
    print("✓ Shared materials (5 instead of 2495)")
    print("✓ Collection-based visibility")
    print("✓ Optimized keyframe strategy")
    print("✓ Viewport optimizations")
    print("\nVECTOR COLORS:")
    print("- RED: Trunk vector")
    print("- BLUE: Neck vector")
    print("- GREEN: Left arm")
    print("- YELLOW: Right arm")
    print("\nCONTROLS:")
    print("- Press SPACEBAR to play")
    print("- Timeline: 1-" + str(num_frames))
    print("\nTIP: Change FRAME_SKIP to 2 or 3 for even faster loading!")
    print("=" * 60)
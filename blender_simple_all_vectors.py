#!/usr/bin/env python3
"""
Simple Blender script to visualize 3D mesh with all vectors
No camera, no lights - just mesh and vectors
"""

import bpy
from pathlib import Path

print("\n" + "=" * 60)
print("SIMPLE BLENDER MESH + VECTORS IMPORT")
print("=" * 60)

# Clear everything
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Set base directory - ADJUST THIS PATH
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")

if not base_dir.exists():
    print(f"ERROR: Directory not found: {base_dir}")
else:
    # Find all frame files to determine number of frames
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    print(f"Found {num_frames} frames")
    print("Importing...")
    
    all_objects = []
    
    # Import for each frame
    for frame_idx in range(num_frames):
        frame_objects = []
        
        # 1. Import mesh
        mesh_file = base_dir / f"frame_{frame_idx:04d}.obj"
        if mesh_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(mesh_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(mesh_file))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"Mesh_{frame_idx:04d}"
                frame_objects.append(obj)
                print(f"  Frame {frame_idx}: Mesh imported")
        
        # 2. Import trunk vector (RED)
        trunk_file = base_dir / f"trunk_{frame_idx:04d}.obj"
        if trunk_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_file))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"Trunk_{frame_idx:04d}"
                # Simple color without materials
                obj.color = (1.0, 0.2, 0.2, 1.0)  # RED
                obj.show_wire = True
                obj.show_in_front = True
                frame_objects.append(obj)
                print(f"  Frame {frame_idx}: Trunk vector imported")
        
        # 3. Import neck vector (BLUE)
        neck_file = base_dir / f"neck_{frame_idx:04d}.obj"
        if neck_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(neck_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(neck_file))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"Neck_{frame_idx:04d}"
                obj.color = (0.2, 0.2, 1.0, 1.0)  # BLUE
                obj.show_wire = True
                obj.show_in_front = True
                frame_objects.append(obj)
                print(f"  Frame {frame_idx}: Neck vector imported")
        
        # 4. Import left arm vector (GREEN)
        left_arm_file = base_dir / f"left_arm_{frame_idx:04d}.obj"
        if left_arm_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(left_arm_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(left_arm_file))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"LeftArm_{frame_idx:04d}"
                obj.color = (0.2, 1.0, 0.2, 1.0)  # GREEN
                obj.show_wire = True
                obj.show_in_front = True
                frame_objects.append(obj)
                print(f"  Frame {frame_idx}: Left arm vector imported")
        
        # 5. Import right arm vector (YELLOW)
        right_arm_file = base_dir / f"right_arm_{frame_idx:04d}.obj"
        if right_arm_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(right_arm_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(right_arm_file))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"RightArm_{frame_idx:04d}"
                obj.color = (1.0, 1.0, 0.2, 1.0)  # YELLOW
                obj.show_wire = True
                obj.show_in_front = True
                frame_objects.append(obj)
                print(f"  Frame {frame_idx}: Right arm vector imported")
        
        # Hide all objects except first frame
        if frame_idx > 0:
            for obj in frame_objects:
                obj.hide_viewport = True
                obj.hide_render = True
        
        all_objects.append(frame_objects)
        
        # Progress
        if frame_idx % 10 == 0:
            print(f"Imported {frame_idx}/{num_frames} frames...")
    
    print("\nSetting up animation...")
    
    # Setup simple frame-by-frame animation
    for frame_idx, frame_objs in enumerate(all_objects):
        frame_num = frame_idx + 1
        
        # Show current frame objects
        for obj in frame_objs:
            obj.hide_viewport = (frame_idx != 0)  # Show first frame
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
            obj.hide_render = (frame_idx != 0)
            obj.keyframe_insert(data_path="hide_render", frame=frame_num)
            
            # Hide on previous frame
            if frame_idx > 0:
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx)
                obj.keyframe_insert(data_path="hide_render", frame=frame_idx)
            
            # Hide on next frame
            if frame_idx < num_frames - 1:
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(1)
    
    # Set viewport to show colors
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'OBJECT'
    
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE!")
    print(f"Total frames: {num_frames}")
    print("\nVECTOR COLORS:")
    print("- RED: Trunk vector (spine)")
    print("- BLUE: Neck vector")
    print("- GREEN: Left arm")
    print("- YELLOW: Right arm")
    print("\nPress SPACEBAR to play animation")
    print("=" * 60)
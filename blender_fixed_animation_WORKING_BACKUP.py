#!/usr/bin/env python3
"""
Fixed Blender animation script - properly handles frame visibility
"""

import bpy
from pathlib import Path

print("\n" + "=" * 60)
print("FIXED BLENDER MESH + VECTORS ANIMATION")
print("=" * 60)

# Clear everything
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Set base directory
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")

if not base_dir.exists():
    print(f"ERROR: Directory not found: {base_dir}")
else:
    # Find all frame files
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    print(f"Found {num_frames} frames")
    print("Importing all objects...")
    
    # Store all objects organized by frame
    frames_data = {}
    
    # Import everything first
    for frame_idx in range(num_frames):
        frames_data[frame_idx] = []
        
        # List of files to import for this frame
        files_to_import = [
            (f"frame_{frame_idx:04d}.obj", "Mesh", (0.8, 0.8, 0.8, 1.0)),
            (f"trunk_{frame_idx:04d}.obj", "Trunk", (1.0, 0.2, 0.2, 1.0)),
            (f"neck_{frame_idx:04d}.obj", "Neck", (0.2, 0.2, 1.0, 1.0)),
            (f"left_arm_{frame_idx:04d}.obj", "LeftArm", (0.2, 1.0, 0.2, 1.0)),
            (f"right_arm_{frame_idx:04d}.obj", "RightArm", (1.0, 1.0, 0.2, 1.0))
        ]
        
        for filename, obj_type, color in files_to_import:
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
                    obj.color = color
                    
                    # Special display settings for vectors (not mesh)
                    if obj_type != "Mesh":
                        # Make vectors visible through mesh
                        obj.show_in_front = True  # Display in front of other objects
                        obj.show_wire = False  # Solid display
                        obj.display_type = 'SOLID'  # Full solid display
                        
                        # Create emission material for better visibility
                        mat = bpy.data.materials.new(name=f"{obj_type}_Mat_{frame_idx}")
                        mat.use_nodes = True
                        mat.blend_method = 'OPAQUE'  # Fully opaque
                        
                        # Get principled BSDF
                        bsdf = mat.node_tree.nodes.get("Principled BSDF")
                        if bsdf:
                            # Set base color
                            if "Base Color" in bsdf.inputs:
                                bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
                            
                            # Make it emissive for better visibility
                            if "Emission" in bsdf.inputs:
                                bsdf.inputs["Emission"].default_value = (*color[:3], 1.0)
                                if "Emission Strength" in bsdf.inputs:
                                    bsdf.inputs["Emission Strength"].default_value = 2.0
                            
                            # No transparency
                            if "Alpha" in bsdf.inputs:
                                bsdf.inputs["Alpha"].default_value = 1.0
                        
                        # Apply material
                        obj.data.materials.clear()
                        obj.data.materials.append(mat)
                    else:
                        # Mesh settings - make it slightly transparent
                        obj.show_in_front = False
                        obj.display_type = 'SOLID'
                        
                        # Create semi-transparent material for mesh
                        mat = bpy.data.materials.new(name=f"Mesh_Mat_{frame_idx}")
                        mat.use_nodes = True
                        mat.blend_method = 'BLEND'  # Enable transparency
                        
                        bsdf = mat.node_tree.nodes.get("Principled BSDF")
                        if bsdf:
                            if "Base Color" in bsdf.inputs:
                                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
                            if "Alpha" in bsdf.inputs:
                                bsdf.inputs["Alpha"].default_value = 0.3  # 30% opacity
                        
                        obj.data.materials.clear()
                        obj.data.materials.append(mat)
                    
                    # Add to frame data
                    frames_data[frame_idx].append(obj)
                    
                    # Initially hide everything
                    obj.hide_viewport = True
                    obj.hide_render = True
        
        if frame_idx % 50 == 0:
            print(f"  Imported frame {frame_idx}/{num_frames}")
    
    print(f"\nTotal objects imported: {sum(len(objs) for objs in frames_data.values())}")
    print("Setting up animation keyframes...")
    
    # Now set up the animation properly
    # For each frame in the timeline
    for timeline_frame in range(1, num_frames + 1):
        frame_idx = timeline_frame - 1  # 0-based index
        
        # Go to this frame in timeline
        bpy.context.scene.frame_set(timeline_frame)
        
        # Set visibility for ALL objects at this frame
        for check_idx, objects in frames_data.items():
            for obj in objects:
                if check_idx == frame_idx:
                    # This frame's objects should be visible
                    obj.hide_viewport = False
                    obj.hide_render = False
                else:
                    # All other frames' objects should be hidden
                    obj.hide_viewport = True
                    obj.hide_render = True
                
                # Insert keyframe at current timeline position
                obj.keyframe_insert(data_path="hide_viewport", frame=timeline_frame)
                obj.keyframe_insert(data_path="hide_render", frame=timeline_frame)
        
        if timeline_frame % 50 == 0:
            print(f"  Set keyframes for frame {timeline_frame}/{num_frames}")
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(1)
    
    # Make sure first frame is visible
    for obj in frames_data[0]:
        obj.hide_viewport = False
        obj.hide_render = False
    
    # Set viewport to material preview for transparency
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'  # Use material preview to see transparency
                    space.shading.use_scene_lights = True
                    space.shading.use_scene_world = False
                    space.overlay.show_extras = True
    
    print("\n" + "=" * 60)
    print("ANIMATION SETUP COMPLETE!")
    print(f"Total frames: {num_frames}")
    print(f"Total objects: {sum(len(objs) for objs in frames_data.values())}")
    print("\nVECTOR COLORS:")
    print("- RED: Trunk vector")
    print("- BLUE: Neck vector")
    print("- GREEN: Left arm")
    print("- YELLOW: Right arm")
    print("\nCONTROLS:")
    print("- Press SPACEBAR to play")
    print("- Use timeline slider to navigate")
    print("- Make sure timeline shows frames 1-" + str(num_frames))
    print("=" * 60)
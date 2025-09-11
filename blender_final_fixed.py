#!/usr/bin/env python3
"""
FINAL FIXED Blender Animation Script
Fixes blinking issue and ensures smooth animation
"""

import bpy
import sys
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True, help="Directory with OBJ files")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--resolution_x", type=int, default=1920, help="Width")
    parser.add_argument("--resolution_y", type=int, default=1080, help="Height")
    parser.add_argument("--samples", type=int, default=64, help="Render samples")
    
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    return parser.parse_args(argv)

def main():
    args = parse_arguments()
    
    base_dir = Path(args.obj_dir).resolve()
    output_path = Path(args.output).resolve()
    
    print(f"Scanning directory: {base_dir}")
    
    # Count ALL frame files (not just frame_*.obj)
    frame_files = sorted([f for f in base_dir.glob("frame_*.obj")])
    trunk_files = sorted([f for f in base_dir.glob("trunk_skin_*.obj")])
    
    # Get actual frame count from filenames
    frame_numbers = []
    for f in frame_files:
        try:
            num = int(f.stem.split('_')[1])
            frame_numbers.append(num)
        except:
            pass
    
    if not frame_numbers:
        print("ERROR: No valid frame files found")
        sys.exit(1)
    
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    num_frames = max_frame - min_frame + 1
    
    print(f"Frame range: {min_frame} to {max_frame} ({num_frames} frames)")
    print(f"Found {len(frame_files)} frame files, {len(trunk_files)} trunk files")
    
    # Clear scene
    print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Remove default objects
    for obj_name in ['Cube', 'Light', 'Camera']:
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    
    # Add camera
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0, 0.785)
    bpy.context.scene.camera = camera
    
    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 1.5
    
    # Add area light for better illumination
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 8))
    area = bpy.context.object
    area.data.energy = 50
    area.data.size = 10
    
    # Store all imported objects by frame number
    all_frame_objects = {}
    
    print("Importing OBJ files...")
    import_count = 0
    
    # Import frames based on actual frame numbers
    for frame_idx in frame_numbers:
        frame_objects = []
        
        # Import frame mesh
        frame_path = base_dir / f"frame_{frame_idx:04d}.obj"
        if frame_path.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(frame_path))
            except:
                bpy.ops.import_scene.obj(filepath=str(frame_path))
            
            for obj in bpy.context.selected_objects:
                obj.name = f"Mesh_{frame_idx:04d}"
                # Gray material
                mat = bpy.data.materials.new(name=f"MeshMat_{frame_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                frame_objects.append(obj)
                import_count += 1
        
        # Import trunk_skin
        trunk_path = base_dir / f"trunk_skin_{frame_idx:04d}.obj"
        if trunk_path.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_path))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_path))
            
            for obj in bpy.context.selected_objects:
                obj.name = f"TrunkSkin_{frame_idx:04d}"
                # Red material with emission
                mat = bpy.data.materials.new(name=f"TrunkMat_{frame_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (1.0, 0.2, 0.2, 1.0)
                # Add emission for better visibility
                if "Emission Color" in bsdf.inputs:
                    bsdf.inputs["Emission Color"].default_value = (1.0, 0.2, 0.2, 1.0)
                    if "Emission Strength" in bsdf.inputs:
                        bsdf.inputs["Emission Strength"].default_value = 1.0
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                obj.show_in_front = True
                frame_objects.append(obj)
                import_count += 1
        
        # Store frame objects
        all_frame_objects[frame_idx] = frame_objects
        
        # Hide all initially
        for obj in frame_objects:
            obj.hide_viewport = True
            obj.hide_render = True
        
        if frame_idx % 20 == 0:
            print(f"  Imported frame {frame_idx} ({import_count} objects so far)")
    
    print(f"Import complete: {import_count} total objects")
    
    # Setup animation - FIXED VERSION
    print("Creating animation keyframes...")
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(frame_numbers)  # Use actual frame count
    
    # Create mapping from timeline frame to actual frame index
    frame_mapping = {}
    for i, frame_idx in enumerate(sorted(frame_numbers)):
        frame_mapping[i + 1] = frame_idx
    
    # For each frame in the timeline
    for timeline_frame in range(1, len(frame_numbers) + 1):
        actual_frame_idx = frame_mapping[timeline_frame]
        scene.frame_set(timeline_frame)
        
        # Go through ALL objects and set visibility
        for frame_idx, objects in all_frame_objects.items():
            should_be_visible = (frame_idx == actual_frame_idx)
            
            for obj in objects:
                # Set visibility for this frame
                obj.hide_viewport = not should_be_visible
                obj.hide_render = not should_be_visible
                
                # Insert keyframe with CONSTANT interpolation to prevent blinking
                obj.keyframe_insert(data_path="hide_viewport", frame=timeline_frame)
                obj.keyframe_insert(data_path="hide_render", frame=timeline_frame)
                
                # Set interpolation to CONSTANT (no blending between frames)
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        for keyframe in fcurve.keyframe_points:
                            keyframe.interpolation = 'CONSTANT'
    
    print(f"Animation ready: {len(frame_numbers)} frames")
    
    # Render settings
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.audio_codec = 'NONE'
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    
    # Use EEVEE
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = args.samples
    
    # Render
    print(f"Rendering {len(frame_numbers)} frames at {args.resolution_x}x{args.resolution_y}...")
    bpy.ops.render.render(animation=True)
    
    print(f"[SUCCESS] Animation rendered to: {output_path}")
    sys.exit(0)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Optimized Blender Headless Render Script for Large Frame Counts
Handles 499+ frames efficiently without loading all at once
"""

import bpy
import sys
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments passed after '--' """
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True, help="Directory with OBJ files")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--resolution_x", type=int, default=1920, help="Width")
    parser.add_argument("--resolution_y", type=int, default=1080, help="Height")
    parser.add_argument("--samples", type=int, default=64, help="Render samples")
    
    # Find where script arguments start (after '--')
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    return parser.parse_args(argv)

def setup_scene():
    """Clear scene and setup basic settings"""
    print("Setting up scene...")
    
    # Clear everything
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Remove default cube, light, camera if they exist
    for obj_name in ['Cube', 'Light', 'Camera']:
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    
    # Add camera
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0, 0.785)  # Point at origin
    
    # Add light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    light = bpy.context.object
    light.data.energy = 1.5
    
    # Add ambient light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 8))
    area_light = bpy.context.object
    area_light.data.energy = 50
    area_light.data.size = 10

def import_frame_objects(base_dir, frame_idx):
    """Import objects for a single frame"""
    imported_objects = []
    
    # List of files to import for this frame
    files_to_import = [
        (f"frame_{frame_idx:04d}.obj", "Mesh", (0.8, 0.8, 0.8, 1.0)),
        (f"trunk_{frame_idx:04d}.obj", "Trunk", (1.0, 0.2, 0.2, 1.0)),
        (f"trunk_skin_{frame_idx:04d}.obj", "TrunkSkin", (1.0, 0.2, 0.2, 1.0)),
        (f"neck_{frame_idx:04d}.obj", "Neck", (0.2, 0.2, 1.0, 1.0)),
        (f"neck_skin_{frame_idx:04d}.obj", "NeckSkin", (0.2, 0.2, 1.0, 1.0)),
        (f"head_{frame_idx:04d}.obj", "Head", (0.5, 0.2, 1.0, 1.0)),
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
                    obj.show_in_front = True
                    obj.show_wire = False
                    obj.display_type = 'SOLID'
                    
                    # Create emission material for better visibility
                    mat = bpy.data.materials.new(name=f"{obj_type}_Mat_{frame_idx}")
                    mat.use_nodes = True
                    mat.blend_method = 'OPAQUE'
                    
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
                    mat.blend_method = 'BLEND'
                    
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf:
                        if "Base Color" in bsdf.inputs:
                            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
                        if "Alpha" in bsdf.inputs:
                            bsdf.inputs["Alpha"].default_value = 0.3
                    
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                
                # Add to imported objects list
                imported_objects.append(obj)
    
    return imported_objects

def clear_frame_objects(objects):
    """Remove objects from scene"""
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def render_frame_by_frame(base_dir, output_path, num_frames, fps, resolution, samples):
    """Render animation frame by frame without loading all at once"""
    print(f"Starting frame-by-frame render of {num_frames} frames...")
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    
    # Use EEVEE for speed
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_motion_blur = False
    
    # Set timeline
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Create temporary directory for frames
    import tempfile
    import os
    temp_dir = Path(tempfile.mkdtemp(prefix="blender_frames_"))
    print(f"Rendering frames to temp dir: {temp_dir}")
    
    # Render each frame individually
    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx + 1)
        
        # Import objects for this frame
        objects = import_frame_objects(base_dir, frame_idx)
        
        # Render this frame
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        scene.render.filepath = str(frame_path)
        scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)
        
        # Clear objects to free memory
        clear_frame_objects(objects)
        
        # Progress report
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Rendered frame {frame_idx + 1}/{num_frames}")
    
    print("All frames rendered, creating video...")
    
    # Use FFmpeg to create video from frames
    import subprocess
    
    # Build FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", str(temp_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    
    # Clean up temp frames
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Video created: {output_path}")
    return True

def render_batch_method(base_dir, output_path, num_frames, fps, resolution, samples):
    """Alternative method: Use Blender's batch rendering with dynamic loading"""
    print(f"Using Blender batch rendering for {num_frames} frames...")
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    
    # Use EEVEE for speed
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_motion_blur = False
    
    # Set timeline
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Create a handler to dynamically load/unload objects
    def frame_change_handler(scene):
        frame = scene.frame_current - 1  # 0-indexed
        
        # Clear all mesh objects
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Import objects for current frame
        import_frame_objects(base_dir, frame)
    
    # Register the handler
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(frame_change_handler)
    
    # Import first frame
    import_frame_objects(base_dir, 0)
    
    # Render animation
    print("Starting batch render...")
    bpy.ops.render.render(animation=True)
    
    print(f"Render complete: {output_path}")
    return True

def main():
    args = parse_arguments()
    
    # Convert to Path
    base_dir = Path(args.obj_dir)
    output_path = Path(args.output)
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Count available frames
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        print(f"ERROR: No frame_*.obj files found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {num_frames} frames")
    
    # Setup scene
    setup_scene()
    
    # Choose rendering method based on frame count
    resolution = (args.resolution_x, args.resolution_y)
    
    if num_frames <= 100:
        # For small frame counts, use original batch method
        print("Using batch method (small frame count)")
        success = render_batch_method(base_dir, output_path, num_frames, 
                                    args.fps, resolution, args.samples)
    else:
        # For large frame counts, use frame-by-frame method
        print("Using frame-by-frame method (large frame count)")
        success = render_frame_by_frame(base_dir, output_path, num_frames,
                                       args.fps, resolution, args.samples)
    
    if success:
        print("[OK] Render completed successfully")
        sys.exit(0)
    else:
        print("[ERROR] Render failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
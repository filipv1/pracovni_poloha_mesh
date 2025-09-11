#!/usr/bin/env python3
"""
Blender Headless Render Script
Based on blender_fixed_animation_WORKING_BACKUP.py
Renders animation to MP4 without UI
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
    light.data.energy = 2.0
    
    # Set camera as active
    bpy.context.scene.camera = camera

def import_animation_frames(base_dir):
    """Import all OBJ files and setup animation"""
    print(f"Importing frames from: {base_dir}")
    
    # Find all frame files
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        print("ERROR: No frame files found!")
        return 0
    
    print(f"Found {num_frames} frames")
    
    # Store all objects organized by frame
    frames_data = {}
    
    # Import everything first
    for frame_idx in range(num_frames):
        frames_data[frame_idx] = []
        
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
                    
                    # Add to frame data
                    frames_data[frame_idx].append(obj)
                    
                    # Initially hide everything
                    obj.hide_viewport = True
                    obj.hide_render = True
        
        if frame_idx % 50 == 0:
            print(f"  Imported frame {frame_idx}/{num_frames}")
    
    print(f"Total objects imported: {sum(len(objs) for objs in frames_data.values())}")
    print("Setting up animation keyframes...")
    
    # Set up animation keyframes
    for timeline_frame in range(1, num_frames + 1):
        frame_idx = timeline_frame - 1
        
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
    
    return num_frames

def setup_render_settings(output_path, fps, resolution_x, resolution_y, samples):
    """Configure render settings for MP4 output"""
    print("Configuring render settings...")
    
    scene = bpy.context.scene
    
    # Output settings - ensure no frame numbers are added
    # Remove extension if present and add it back properly
    output_base = str(output_path)
    if output_base.endswith('.mp4'):
        output_base = output_base[:-4]
    
    scene.render.filepath = output_base  # Blender will add extension
    scene.render.image_settings.file_format = 'FFMPEG'
    
    # Video codec settings
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    
    # Resolution
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    
    # Frame rate
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    
    # Render engine settings (Blender 4.5 uses EEVEE_NEXT)
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Faster than Cycles
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_motion_blur = False  # Disable for clarity
    
    # Transparency for compositing
    scene.render.film_transparent = False
    
    print(f"  Output base: {output_base}")
    print(f"  Full path: {scene.render.filepath}")
    print(f"  Resolution: {resolution_x}x{resolution_y}")
    print(f"  FPS: {fps}")
    print(f"  Samples: {samples}")
    print(f"  Codec: H.264/MPEG4")

def render_animation():
    """Render the animation"""
    print("Starting render...")
    
    # Render animation
    try:
        bpy.ops.render.render(animation=True, write_still=True)
        print("Render completed successfully!")
        return True
    except Exception as e:
        print(f"Render failed: {e}")
        return False

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("BLENDER HEADLESS RENDER")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate input directory
    base_dir = Path(args.obj_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Setup scene
    setup_scene()
    
    # Import animation frames
    num_frames = import_animation_frames(base_dir)
    if num_frames == 0:
        print("ERROR: No frames to render")
        sys.exit(1)
    
    # Setup render settings
    setup_render_settings(
        args.output,
        args.fps,
        args.resolution_x,
        args.resolution_y,
        args.samples
    )
    
    # Render animation
    if render_animation():
        print("\n" + "="*60)
        print("RENDER COMPLETE!")
        print(f"Output saved to: {args.output}")
        print(f"Total frames: {num_frames}")
        print(f"Duration: {num_frames/args.fps:.1f} seconds")
        print("="*60)
        sys.exit(0)
    else:
        print("\nERROR: Render failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
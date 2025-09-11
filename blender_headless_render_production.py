#!/usr/bin/env python3
"""
Production Blender Headless Render Script for Large Frame Counts
Handles 499+ frames efficiently using Blender's internal capabilities
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
    bpy.context.scene.camera = camera  # Set as active camera
    
    # Add light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    light = bpy.context.object
    light.data.energy = 1.5
    
    # Add ambient light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 8))
    area_light = bpy.context.object
    area_light.data.energy = 50
    area_light.data.size = 10

def create_placeholder_for_frame(frame_idx):
    """Create placeholder objects that will be replaced dynamically"""
    placeholders = []
    
    # Object types to create placeholders for
    object_types = [
        ("Mesh", (0.8, 0.8, 0.8, 1.0)),
        ("Trunk", (1.0, 0.2, 0.2, 1.0)),
        ("TrunkSkin", (1.0, 0.2, 0.2, 1.0)),
        ("Neck", (0.2, 0.2, 1.0, 1.0)),
        ("NeckSkin", (0.2, 0.2, 1.0, 1.0)),
        ("Head", (0.5, 0.2, 1.0, 1.0)),
        ("LeftArm", (0.2, 1.0, 0.2, 1.0)),
        ("RightArm", (1.0, 1.0, 0.2, 1.0))
    ]
    
    for obj_type, color in object_types:
        # Create empty placeholder
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        empty = bpy.context.object
        empty.name = f"{obj_type}_{frame_idx:04d}_placeholder"
        placeholders.append(empty)
    
    return placeholders

def setup_frame_loading_driver(base_dir, num_frames):
    """Setup Blender to dynamically load frames during render"""
    
    # Python script that will be executed on frame change
    driver_script = f"""
import bpy
from pathlib import Path

# Global function definition
global load_frame_objects

def load_frame_objects(frame_num):
    base_dir = Path(r"{base_dir}")
    frame_idx = frame_num - 1  # 0-indexed
    
    # Clear existing mesh objects
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH' and not obj.name.startswith("__keep__"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Files to import
    files_to_import = [
        (f"frame_{{frame_idx:04d}}.obj", "Mesh", (0.8, 0.8, 0.8, 1.0)),
        (f"trunk_{{frame_idx:04d}}.obj", "Trunk", (1.0, 0.2, 0.2, 1.0)),
        (f"trunk_skin_{{frame_idx:04d}}.obj", "TrunkSkin", (1.0, 0.2, 0.2, 1.0)),
        (f"neck_{{frame_idx:04d}}.obj", "Neck", (0.2, 0.2, 1.0, 1.0)),
        (f"neck_skin_{{frame_idx:04d}}.obj", "NeckSkin", (0.2, 0.2, 1.0, 1.0)),
        (f"head_{{frame_idx:04d}}.obj", "Head", (0.5, 0.2, 1.0, 1.0)),
        (f"left_arm_{{frame_idx:04d}}.obj", "LeftArm", (0.2, 1.0, 0.2, 1.0)),
        (f"right_arm_{{frame_idx:04d}}.obj", "RightArm", (1.0, 1.0, 0.2, 1.0))
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
                obj.name = f"{{obj_type}}_{{frame_idx:04d}}"
                
                # Apply material
                if obj_type != "Mesh":
                    obj.show_in_front = True
                    mat = bpy.data.materials.new(name=f"{{obj_type}}_Mat")
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf and "Base Color" in bsdf.inputs:
                        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
                        # Try to set emission if available
                        if "Emission Color" in bsdf.inputs:
                            bsdf.inputs["Emission Color"].default_value = (*color[:3], 1.0)
                        if "Emission Strength" in bsdf.inputs:
                            bsdf.inputs["Emission Strength"].default_value = 2.0
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                else:
                    mat = bpy.data.materials.new(name="Mesh_Mat")
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

# Handler function
def frame_change_handler(scene):
    load_frame_objects(scene.frame_current)

# Register handler
bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(frame_change_handler)

# Load first frame
load_frame_objects(1)
"""
    
    # Execute the driver script
    exec(driver_script)
    print(f"Registered frame loading handler for {num_frames} frames")

def render_animation_efficiently(base_dir, output_path, num_frames, fps, resolution, samples):
    """Render animation using batch processing for efficiency"""
    print(f"Rendering {num_frames} frames...")
    
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
    # Motion blur setting removed - not available in Blender 4.5
    
    # Set timeline
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Process in batches if too many frames
    BATCH_SIZE = 100
    
    if num_frames <= BATCH_SIZE:
        # Small enough to process all at once
        print(f"Processing all {num_frames} frames in one batch...")
        
        # Import all frames
        frames_data = {}
        for frame_idx in range(num_frames):
            frames_data[frame_idx] = []
            
            # List of files to import for this frame
            files_to_import = [
                (f"frame_{frame_idx:04d}.obj", "Mesh", (0.8, 0.8, 0.8, 1.0)),
                (f"trunk_{frame_idx:04d}.obj", "Trunk", (1.0, 0.2, 0.2, 1.0)),
                (f"trunk_skin_{frame_idx:04d}.obj", "TrunkSkin", (1.0, 0.2, 0.2, 1.0)),
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
                        
                        # Apply simple material
                        if obj_type != "Mesh":
                            obj.show_in_front = True
                            mat = bpy.data.materials.new(name=f"{obj_type}_Mat_{frame_idx}")
                            mat.use_nodes = True
                            bsdf = mat.node_tree.nodes.get("Principled BSDF")
                            if bsdf and "Base Color" in bsdf.inputs:
                                bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
                            obj.data.materials.clear()
                            obj.data.materials.append(mat)
                        else:
                            mat = bpy.data.materials.new(name=f"Mesh_Mat_{frame_idx}")
                            mat.use_nodes = True
                            obj.data.materials.clear()
                            obj.data.materials.append(mat)
                        
                        frames_data[frame_idx].append(obj)
                        obj.hide_viewport = True
                        obj.hide_render = True
            
            if frame_idx % 50 == 0:
                print(f"  Imported frame {frame_idx}/{num_frames}")
        
        # Setup keyframes for visibility
        print("Setting up animation keyframes...")
        for frame in range(1, num_frames + 1):
            idx = frame - 1
            bpy.context.scene.frame_set(frame)
            
            for check_idx, objects in frames_data.items():
                for obj in objects:
                    if check_idx == idx:
                        obj.hide_viewport = False
                        obj.hide_render = False
                    else:
                        obj.hide_viewport = True
                        obj.hide_render = True
                    
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                    obj.keyframe_insert(data_path="hide_render", frame=frame)
        
    else:
        # Too many frames - use dynamic loading
        print(f"Using dynamic loading for {num_frames} frames...")
        setup_frame_loading_driver(base_dir, num_frames)
    
    # Render animation
    print("Starting render...")
    bpy.ops.render.render(animation=True)
    
    print(f"Render complete: {output_path}")
    return True

def main():
    args = parse_arguments()
    
    # Convert to Path - make output absolute
    base_dir = Path(args.obj_dir).resolve()
    output_path = Path(args.output).resolve()
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Count available frames (look for frame_XXXX.obj files)
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        print(f"ERROR: No frame_*.obj files found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {num_frames} frames")
    
    # Setup scene
    setup_scene()
    
    # Render
    resolution = (args.resolution_x, args.resolution_y)
    success = render_animation_efficiently(base_dir, output_path, num_frames,
                                          args.fps, resolution, args.samples)
    
    if success:
        print("[OK] Render completed successfully")
        sys.exit(0)
    else:
        print("[ERROR] Render failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
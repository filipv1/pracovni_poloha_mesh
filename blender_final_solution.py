#!/usr/bin/env python3
"""
FINAL SOLUTION - No blinking, proper frame-by-frame animation
Uses stepped keyframes to ensure only one frame visible at a time
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
    
    print(f"Directory: {base_dir}")
    
    # Count frames
    frame_files = sorted(base_dir.glob("frame_*.obj"))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        print("ERROR: No frame files found")
        sys.exit(1)
    
    print(f"Found {num_frames} frames")
    
    # Clear scene
    print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Remove defaults
    for obj_name in ['Cube', 'Light', 'Camera']:
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    
    # Add camera
    bpy.ops.object.camera_add(location=(5, -5, 3))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0, 0.785)
    bpy.context.scene.camera = camera
    
    # Add lights
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 8))
    area = bpy.context.object
    area.data.energy = 100
    area.data.size = 15
    
    # Import all frames and store them
    print("Importing frames...")
    all_meshes = []
    all_trunks = []
    
    for frame_idx in range(num_frames):
        # Import mesh
        frame_path = base_dir / f"frame_{frame_idx:04d}.obj"
        if frame_path.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(frame_path))
            except:
                bpy.ops.import_scene.obj(filepath=str(frame_path))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"Mesh_{frame_idx:04d}"
                
                # Simple gray material
                mat = bpy.data.materials.new(name=f"MeshMat_{frame_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                
                all_meshes.append(obj)
                
                # Start hidden
                obj.hide_viewport = True
                obj.hide_render = True
        
        # Import trunk if exists (optional)
        trunk_path = base_dir / f"trunk_skin_{frame_idx:04d}.obj"
        if trunk_path.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_path))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_path))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"Trunk_{frame_idx:04d}"
                
                # Red material
                mat = bpy.data.materials.new(name=f"TrunkMat_{frame_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                
                obj.show_in_front = True
                all_trunks.append(obj)
                
                # Start hidden
                obj.hide_viewport = True
                obj.hide_render = True
        
        if frame_idx % 20 == 0:
            print(f"  Imported {frame_idx}/{num_frames}")
    
    print(f"Imported {len(all_meshes)} meshes, {len(all_trunks)} trunks")
    
    # CRITICAL: Create proper stepped animation
    print("Creating stepped animation...")
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Method: Use two keyframes per transition with CONSTANT interpolation
    for frame_num in range(1, num_frames + 1):
        frame_idx = frame_num - 1
        scene.frame_set(frame_num)
        
        # Update meshes
        for obj_idx, obj in enumerate(all_meshes):
            if obj_idx == frame_idx:
                # This object should be visible on this frame
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num)
                
                # Hide it on next frame (if not last)
                if frame_num < num_frames:
                    scene.frame_set(frame_num + 1)
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
                    scene.frame_set(frame_num)  # Go back
            else:
                # This object should be hidden
                if frame_num == 1:  # Only set initial state
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num)
        
        # Update trunks (same logic)
        for obj_idx, obj in enumerate(all_trunks):
            if obj_idx == frame_idx:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num)
                
                if frame_num < num_frames:
                    scene.frame_set(frame_num + 1)
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
                    scene.frame_set(frame_num)
            else:
                if frame_num == 1:
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num)
    
    # Set all keyframes to CONSTANT interpolation
    print("Setting constant interpolation...")
    for obj in all_meshes + all_trunks:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'CONSTANT'
                    keyframe.handle_left_type = 'VECTOR'
                    keyframe.handle_right_type = 'VECTOR'
    
    print(f"Animation ready: {num_frames} frames")
    
    # Verify frame 1
    scene.frame_set(1)
    visible_meshes = sum(1 for obj in all_meshes if not obj.hide_render)
    visible_trunks = sum(1 for obj in all_trunks if not obj.hide_render)
    print(f"Frame 1 check: {visible_meshes} meshes, {visible_trunks} trunks visible")
    
    # Render settings
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    
    # EEVEE
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = args.samples
    
    # Render
    print(f"Rendering {num_frames} frames at {args.resolution_x}x{args.resolution_y}...")
    bpy.ops.render.render(animation=True)
    
    print(f"[SUCCESS] Rendered to: {output_path}")
    sys.exit(0)

if __name__ == "__main__":
    main()
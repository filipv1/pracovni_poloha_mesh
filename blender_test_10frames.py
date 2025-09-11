#!/usr/bin/env python3
"""
MINIMAL BLENDER TEST - 10 FRAMES ONLY
Debug version to find why MP4 is not created
"""

import bpy
import sys
import os
from pathlib import Path

print("\n" + "="*60)
print("BLENDER 10-FRAME TEST - DEBUG VERSION")
print("="*60)

# Clear everything first
print("Clearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Remove default objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

# Add camera
print("Adding camera...")
bpy.ops.object.camera_add(location=(5, -5, 3))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Add light
print("Adding light...")
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.object
light.data.energy = 2.0

# Base directory
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_skin_5614")
print(f"OBJ directory: {base_dir}")
print(f"Directory exists: {base_dir.exists()}")

# Import ONLY first 10 frames
num_frames = 10
frames_data = {}

print(f"\nImporting {num_frames} frames...")

for frame_idx in range(num_frames):
    frames_data[frame_idx] = []
    
    # Only import frame mesh and trunk vector for simplicity
    files_to_import = [
        (f"frame_{frame_idx:04d}.obj", "Mesh"),
        (f"trunk_skin_{frame_idx:04d}.obj", "Trunk"),
    ]
    
    for filename, obj_type in files_to_import:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  Importing: {filename}")
            
            # Import OBJ
            try:
                bpy.ops.wm.obj_import(filepath=str(filepath))
            except:
                bpy.ops.import_scene.obj(filepath=str(filepath))
            
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = f"{obj_type}_{frame_idx:04d}"
                
                # Simple material
                if obj_type == "Trunk":
                    # Red for trunk
                    mat = bpy.data.materials.new(name=f"Red_{frame_idx}")
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                
                frames_data[frame_idx].append(obj)
                
                # Initially hide
                obj.hide_viewport = True
                obj.hide_render = True

print(f"Imported {len(frames_data)} frames with {sum(len(objs) for objs in frames_data.values())} objects")

# Setup animation keyframes
print("\nSetting up animation keyframes...")

for timeline_frame in range(1, num_frames + 1):
    frame_idx = timeline_frame - 1
    bpy.context.scene.frame_set(timeline_frame)
    
    # Set visibility
    for check_idx, objects in frames_data.items():
        for obj in objects:
            if check_idx == frame_idx:
                obj.hide_viewport = False
                obj.hide_render = False
            else:
                obj.hide_viewport = True
                obj.hide_render = True
            
            obj.keyframe_insert(data_path="hide_viewport", frame=timeline_frame)
            obj.keyframe_insert(data_path="hide_render", frame=timeline_frame)

print(f"Keyframes set for {num_frames} frames")

# Timeline settings
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames
bpy.context.scene.frame_set(1)

# RENDER SETTINGS - CRITICAL PART
print("\n" + "="*60)
print("RENDER SETTINGS")
print("="*60)

scene = bpy.context.scene

# Use ABSOLUTE path with no ambiguity
output_path = r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\test_10frames_output"
print(f"Output path (base): {output_path}")

# Set render path
scene.render.filepath = output_path

# Configure for MP4
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'

# Low quality for fast test
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.fps = 10

# Use EEVEE for speed (Blender 4.5 uses EEVEE_NEXT)
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.eevee.taa_render_samples = 16

print(f"File format: {scene.render.image_settings.file_format}")
print(f"Codec: {scene.render.ffmpeg.codec}")
print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
print(f"FPS: {scene.render.fps}")
print(f"Frames: {scene.frame_start} to {scene.frame_end}")

# ACTUAL OUTPUT PATH
print(f"\nFINAL render.filepath: {scene.render.filepath}")

# RENDER!
print("\n" + "="*60)
print("STARTING RENDER")
print("="*60)

try:
    bpy.ops.render.render(animation=True, write_still=True)
    print("RENDER COMMAND EXECUTED")
except Exception as e:
    print(f"RENDER ERROR: {e}")

# CHECK OUTPUT
expected_files = [
    output_path + ".mp4",
    output_path + "0001-0010.mp4",
    output_path + ".avi",
    r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\test_10frames_output.mp4"
]

print("\n" + "="*60)
print("CHECKING FOR OUTPUT FILES")
print("="*60)

for expected in expected_files:
    if Path(expected).exists():
        size = Path(expected).stat().st_size / 1024
        print(f"✓ FOUND: {expected} ({size:.1f} KB)")
    else:
        print(f"✗ Not found: {expected}")

# Also check directory for any new files
output_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh")
mp4_files = list(output_dir.glob("*.mp4"))
avi_files = list(output_dir.glob("*.avi"))

print(f"\nAll MP4 files in directory: {[f.name for f in mp4_files]}")
print(f"All AVI files in directory: {[f.name for f in avi_files]}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
#!/usr/bin/env python3
"""
Test pipeline with just 50 frames for speed
"""

import subprocess
import sys
from pathlib import Path

print("\nTEST: 50 FRAMES PIPELINE")
print("="*60)

# Check if blender_export_skin_5614 exists
export_dir = Path("blender_export_skin_5614")
if not export_dir.exists():
    print("[ERROR] Export directory not found. Run export first.")
    sys.exit(1)

# Check how many OBJ files we have
obj_files = list(export_dir.glob("frame_*.obj"))
print(f"Found {len(obj_files)} frame OBJ files")

if len(obj_files) < 50:
    print(f"[ERROR] Need at least 50 frames, found only {len(obj_files)}")
    sys.exit(1)

# Create test script for 50 frames
blender_script = """
import bpy
from pathlib import Path

print("RENDERING 50 FRAMES TEST")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Add camera
bpy.ops.object.camera_add(location=(5, -5, 3))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Import 50 frames
base_dir = Path(r"C:\\Users\\vaclavik\\ruce4\\pracovni_poloha_mesh\\blender_export_skin_5614")
frames_data = {}

for i in range(50):
    frames_data[i] = []
    
    # Import mesh
    mesh_path = base_dir / f"frame_{i:04d}.obj"
    if mesh_path.exists():
        try:
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        except:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
        
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"Mesh_{i:04d}"
            frames_data[i].append(obj)
            obj.hide_viewport = True
            obj.hide_render = True
    
    # Import trunk
    trunk_path = base_dir / f"trunk_skin_{i:04d}.obj"
    if trunk_path.exists():
        try:
            bpy.ops.wm.obj_import(filepath=str(trunk_path))
        except:
            bpy.ops.import_scene.obj(filepath=str(trunk_path))
        
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"Trunk_{i:04d}"
            
            # Red material
            mat = bpy.data.materials.new(name=f"Red_{i}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            
            frames_data[i].append(obj)
            obj.hide_viewport = True
            obj.hide_render = True
    
    if i % 10 == 0:
        print(f"Imported frame {i}/50")

# Setup keyframes
for frame in range(1, 51):
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

# Timeline
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 50

# Render settings
scene = bpy.context.scene
scene.render.filepath = r"C:\\Users\\vaclavik\\ruce4\\pracovni_poloha_mesh\\test_50frames"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 25
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.eevee.taa_render_samples = 16

print("Starting render...")
bpy.ops.render.render(animation=True)
print("Render complete!")
"""

# Write Blender script
script_path = Path("test_50frames_blender.py")
script_path.write_text(blender_script)
print(f"Created Blender script: {script_path}")

# Run Blender
blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
cmd = [blender_exe, "--background", "--python", str(script_path)]

print("\nStarting Blender render...")
print("This will take about 2-3 minutes for 50 frames...")

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("\n[OK] Render completed!")
    
    # Check for output
    possible_outputs = [
        "test_50frames.mp4",
        "test_50frames0001-0050.mp4",
        "test_50frames001-050.mp4",
    ]
    
    for output in possible_outputs:
        if Path(output).exists():
            size = Path(output).stat().st_size / 1024
            print(f"[OK] Found video: {output} ({size:.1f} KB)")
            break
    else:
        print("[ERROR] Video file not found")
else:
    print(f"\n[ERROR] Render failed: {result.returncode}")
    if result.stderr:
        print(f"Error: {result.stderr[:500]}")

print("="*60)
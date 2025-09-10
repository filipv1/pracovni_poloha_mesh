#!/usr/bin/env python3
"""Quick Blender test with just first 10 frames"""

import bpy
from pathlib import Path

print("QUICK BLENDER TEST - 10 FRAMES")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Import just first 10 frames
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_skin_5614")

for i in range(10):
    print(f"Importing frame {i}")
    
    # Import mesh
    obj_path = base_dir / f"frame_{i:04d}.obj"
    if obj_path.exists():
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_path))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_path))

# Setup render
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10

# Output settings
scene.render.filepath = "C:/Users/vaclavik/ruce4/pracovni_poloha_mesh/test_10frames"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 10

print("Rendering...")
bpy.ops.render.render(animation=True)
print("Done! Output: test_10frames.mp4")
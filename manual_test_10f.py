
import bpy
from pathlib import Path

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

# Base directory
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_skin_5614")

# Import 10 frames
all_objects = []
for i in range(10):
    frame_objects = []
    
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
            frame_objects.append(obj)
    
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
            
            frame_objects.append(obj)
    
    all_objects.append(frame_objects)
    print(f"Frame {i}: {len(frame_objects)} objects")

# Animate by hiding/showing
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10

for frame_num in range(1, 11):
    idx = frame_num - 1
    scene.frame_set(frame_num)
    
    # Hide all objects first
    for frame_idx, objs in enumerate(all_objects):
        for obj in objs:
            if frame_idx == idx:
                obj.hide_render = False
                obj.hide_viewport = False
            else:
                obj.hide_render = True
                obj.hide_viewport = True
            
            obj.keyframe_insert(data_path="hide_render", frame=frame_num)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)

print(f"Total keyframes created: {10 * len([o for objs in all_objects for o in objs]) * 2}")

# Render settings
scene.render.filepath = r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\manual_test_10f"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 10
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.eevee.taa_render_samples = 16

# Render
print("Rendering...")
bpy.ops.render.render(animation=True)
print("Done!")

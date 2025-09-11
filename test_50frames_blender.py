
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
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_skin_5614")
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
scene.render.filepath = r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\test_50frames"
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

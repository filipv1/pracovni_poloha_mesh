import bpy
import os
from pathlib import Path

def import_trunk_analysis_sequence():
    """Import trunk analysis sequence with gravitational reference and angles"""
    
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"C:\Users\vaclavik\ruce2\pracovni_poloha_mesh\trunk_analysis_export")
    obj_files = sorted(base_dir.glob("trunk_analysis_*.obj"))
    
    print(f"Found {len(obj_files)} trunk analysis OBJ files")
    
    all_objects = []
    
    # Create materials
    # RED material for trunk vector
    trunk_mat = bpy.data.materials.new(name="TrunkVector_Material")
    trunk_mat.use_nodes = True
    trunk_bsdf = trunk_mat.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (1.0, 0.2, 0.2, 1.0)  # RED
    trunk_bsdf.inputs[12].default_value = 0.1  # Low roughness
    
    # BLUE material for gravity reference
    gravity_mat = bpy.data.materials.new(name="GravityReference_Material")
    gravity_mat.use_nodes = True
    gravity_bsdf = gravity_mat.node_tree.nodes["Principled BSDF"]
    gravity_bsdf.inputs[0].default_value = (0.2, 0.5, 1.0, 1.0)  # BLUE
    gravity_bsdf.inputs[12].default_value = 0.2
    
    # GREEN material for angle arc
    angle_mat = bpy.data.materials.new(name="AngleArc_Material")
    angle_mat.use_nodes = True
    angle_bsdf = angle_mat.node_tree.nodes["Principled BSDF"]
    angle_bsdf.inputs[0].default_value = (0.2, 1.0, 0.3, 1.0)  # GREEN
    angle_bsdf.inputs[12].default_value = 0.3
    
    for frame_idx, obj_file in enumerate(obj_files):
        print(f"Importing trunk analysis frame {frame_idx}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get all imported objects from this frame
        imported_objects = bpy.context.selected_objects.copy()
        
        if imported_objects:
            # Usually we get one combined object, split it by groups if possible
            for obj_idx, obj in enumerate(imported_objects):
                if obj_idx == 0:  # First object gets trunk material (red)
                    obj.name = f"TrunkVector_{frame_idx:04d}"
                    if obj.data.materials:
                        obj.data.materials[0] = trunk_mat
                    else:
                        obj.data.materials.append(trunk_mat)
                else:
                    obj.name = f"TrunkAnalysis_{frame_idx:04d}_{obj_idx}"
                    if obj.data.materials:
                        obj.data.materials[0] = trunk_mat
                    else:
                        obj.data.materials.append(trunk_mat)
                
                all_objects.append(obj)
                
                # Hide all objects except first frame
                if frame_idx > 0:
                    obj.hide_viewport = True
                    obj.hide_render = True
            
            print(f"  OK: Imported {len(imported_objects)} objects for frame {frame_idx}")
        else:
            print(f"  ERROR: No objects imported")
    
    print(f"\nNASTAVUJI ANIMACI TRUNK ANALYZY...")
    
    # Setup keyframes for visibility
    for frame_idx in range(len(obj_files)):
        for obj in all_objects:
            # Extract frame number from object name
            try:
                name_parts = obj.name.split('_')
                obj_frame = int(name_parts[1])  # Should be the frame number
            except:
                continue
            
            if obj_frame == frame_idx:
                # This frame should be visible
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
            else:
                # Other frames should be hidden
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(obj_files)
    bpy.context.scene.frame_set(1)
    
    # Enhanced lighting setup
    bpy.ops.object.light_add(type='SUN', location=(3, 3, 5))
    sun_light = bpy.context.object
    sun_light.data.energy = 4.0
    sun_light.data.angle = 0.1
    
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 3))
    area_light = bpy.context.object
    area_light.data.energy = 8.0
    area_light.data.size = 3.0
    
    # Optimal camera for trunk analysis
    if bpy.data.objects.get('Camera'):
        camera = bpy.data.objects['Camera']
        camera.location = (1.0, -1.0, 0.8)
        camera.rotation_euler = (1.2, 0, 0.7)
    
    print(f"TRUNK ANALYZA ANIMACE PRIPRAVENA!")
    print(f"Timeline: 1-{len(obj_files)} snimku")
    print(f"CERVENA = Vektor trupu (lumbar -> cervical)")
    print(f"MODRA = Gravitacni reference (svislice dolu)")
    print(f"ZELENA = Uhlovy oblouk (uhel mezi vektory)")
    print(f"Stisknete MEZERA pro spusteni animace")

if __name__ == "__main__":
    import_trunk_analysis_sequence()
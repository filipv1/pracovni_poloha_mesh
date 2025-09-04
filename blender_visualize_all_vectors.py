#!/usr/bin/env python3
"""
Blender script to visualize 3D mesh with all vectors (trunk, neck, arms)
Loads OBJ files created by export_all_vectors_to_blender.py
"""

import bpy
import os
from pathlib import Path

def clear_scene():
    """Clear all objects from scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear unused data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

def create_material(name, color, emission_strength=0):
    """Create material with specified color"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    
    # Get principled BSDF
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Set base color
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        elif "Color" in bsdf.inputs:  # Older Blender versions
            bsdf.inputs["Color"].default_value = (*color, 1.0)
        
        # Set roughness
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.5
        
        # Try to set emission (may not exist in older versions)
        if emission_strength > 0:
            try:
                if "Emission" in bsdf.inputs:
                    bsdf.inputs["Emission"].default_value = (*color, 1.0)
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emission_strength
            except:
                # Fallback for older Blender - just make it brighter
                if "Base Color" in bsdf.inputs:
                    brightened = tuple(min(1.0, c * 1.5) for c in color)
                    bsdf.inputs["Base Color"].default_value = (*brightened, 1.0)
    
    return mat

def import_obj_sequence(base_dir):
    """Import all OBJ sequences from directory"""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return None
    
    # Find all frame files
    frame_files = sorted(base_path.glob("frame_*.obj"))
    if not frame_files:
        print(f"ERROR: No frame_*.obj files found in {base_dir}")
        return None
    
    num_frames = len(frame_files)
    print(f"Found {num_frames} frames to import")
    
    # Create materials for different objects
    materials = {
        'mesh': create_material("Mesh_Material", (0.8, 0.8, 0.8)),
        'trunk': create_material("Trunk_Vector", (1.0, 0.2, 0.2), 0.3),  # Red
        'neck': create_material("Neck_Vector", (0.2, 0.2, 1.0), 0.3),   # Blue
        'left_arm': create_material("Left_Arm_Vector", (0.2, 1.0, 0.2), 0.3),  # Green
        'right_arm': create_material("Right_Arm_Vector", (1.0, 1.0, 0.2), 0.3),  # Yellow
    }
    
    # Store all imported objects by frame
    frame_objects = {i: {} for i in range(num_frames)}
    
    print("\nIMPORTING MESHES AND VECTORS...")
    print("-" * 40)
    
    # Import meshes
    for frame_idx, frame_file in enumerate(frame_files):
        if frame_idx % 10 == 0:
            print(f"Importing frame {frame_idx}/{num_frames}...")
        
        # Import mesh
        try:
            bpy.ops.wm.obj_import(filepath=str(frame_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(frame_file))
        
        if bpy.context.selected_objects:
            mesh_obj = bpy.context.selected_objects[0]
            mesh_obj.name = f"Mesh_{frame_idx:04d}"
            mesh_obj.data.materials.append(materials['mesh'])
            frame_objects[frame_idx]['mesh'] = mesh_obj
            
            # Hide all but first frame
            if frame_idx > 0:
                mesh_obj.hide_viewport = True
                mesh_obj.hide_render = True
    
    # Import trunk vectors
    print("\nImporting trunk vectors...")
    for frame_idx in range(num_frames):
        trunk_file = base_path / f"trunk_{frame_idx:04d}.obj"
        if trunk_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(trunk_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(trunk_file))
            
            if bpy.context.selected_objects:
                trunk_obj = bpy.context.selected_objects[0]
                trunk_obj.name = f"Trunk_{frame_idx:04d}"
                trunk_obj.data.materials.append(materials['trunk'])
                frame_objects[frame_idx]['trunk'] = trunk_obj
                
                if frame_idx > 0:
                    trunk_obj.hide_viewport = True
                    trunk_obj.hide_render = True
    
    # Import neck vectors
    print("Importing neck vectors...")
    for frame_idx in range(num_frames):
        neck_file = base_path / f"neck_{frame_idx:04d}.obj"
        if neck_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(neck_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(neck_file))
            
            if bpy.context.selected_objects:
                neck_obj = bpy.context.selected_objects[0]
                neck_obj.name = f"Neck_{frame_idx:04d}"
                neck_obj.data.materials.append(materials['neck'])
                frame_objects[frame_idx]['neck'] = neck_obj
                
                if frame_idx > 0:
                    neck_obj.hide_viewport = True
                    neck_obj.hide_render = True
    
    # Import left arm vectors
    print("Importing left arm vectors...")
    for frame_idx in range(num_frames):
        left_arm_file = base_path / f"left_arm_{frame_idx:04d}.obj"
        if left_arm_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(left_arm_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(left_arm_file))
            
            if bpy.context.selected_objects:
                left_arm_obj = bpy.context.selected_objects[0]
                left_arm_obj.name = f"LeftArm_{frame_idx:04d}"
                left_arm_obj.data.materials.append(materials['left_arm'])
                frame_objects[frame_idx]['left_arm'] = left_arm_obj
                
                if frame_idx > 0:
                    left_arm_obj.hide_viewport = True
                    left_arm_obj.hide_render = True
    
    # Import right arm vectors
    print("Importing right arm vectors...")
    for frame_idx in range(num_frames):
        right_arm_file = base_path / f"right_arm_{frame_idx:04d}.obj"
        if right_arm_file.exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(right_arm_file))
            except:
                bpy.ops.import_scene.obj(filepath=str(right_arm_file))
            
            if bpy.context.selected_objects:
                right_arm_obj = bpy.context.selected_objects[0]
                right_arm_obj.name = f"RightArm_{frame_idx:04d}"
                right_arm_obj.data.materials.append(materials['right_arm'])
                frame_objects[frame_idx]['right_arm'] = right_arm_obj
                
                if frame_idx > 0:
                    right_arm_obj.hide_viewport = True
                    right_arm_obj.hide_render = True
    
    return frame_objects, num_frames

def setup_animation(frame_objects, num_frames):
    """Setup keyframe animation for all objects"""
    
    print("\nSETTING UP ANIMATION...")
    print("-" * 40)
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    # Create keyframes for each frame
    for frame_idx in range(num_frames):
        frame_num = frame_idx + 1
        
        # Set all objects for this frame
        for obj_type in ['mesh', 'trunk', 'neck', 'left_arm', 'right_arm']:
            
            # Current frame objects should be visible
            if obj_type in frame_objects[frame_idx]:
                obj = frame_objects[frame_idx][obj_type]
                
                # Make visible at this frame
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num)
                
                # Hide before and after
                if frame_idx > 0:
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num - 1)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num - 1)
                
                if frame_idx < num_frames - 1:
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
                    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
    
    # Reset to frame 1
    bpy.context.scene.frame_set(1)
    
    print(f"Animation ready: {num_frames} frames")

def setup_camera_and_lighting():
    """Setup camera and lighting for better visualization"""
    
    # Add camera if none exists
    if not bpy.data.objects.get("Camera"):
        bpy.ops.object.camera_add(location=(2.5, -2.5, 2))
        camera = bpy.context.object
        camera.rotation_euler = (1.1, 0, 0.785)
    
    # Add light
    if not bpy.data.objects.get("Light"):
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 3))
        light = bpy.context.object
        light.data.energy = 1.5
    
    # Set viewport shading to solid or material preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

def main():
    """Main function to run the import and setup"""
    
    print("\n" + "=" * 60)
    print("BLENDER MESH + VECTORS VISUALIZATION")
    print("=" * 60)
    
    # Clear scene
    clear_scene()
    
    # Set the path to OBJ files
    # Update this path to match your export directory
    base_dir = r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all"
    
    # Alternative: use relative path
    # base_dir = Path(__file__).parent / "blender_export_all"
    
    # Import all sequences
    result = import_obj_sequence(base_dir)
    
    if result:
        frame_objects, num_frames = result
        
        # Setup animation
        setup_animation(frame_objects, num_frames)
        
        # Setup camera and lighting
        setup_camera_and_lighting()
        
        print("\n" + "=" * 60)
        print("IMPORT COMPLETE!")
        print(f"Total frames: {num_frames}")
        print("\nCONTROLS:")
        print("- Press SPACEBAR to play animation")
        print("- Use timeline to scrub through frames")
        print("- Numpad 0 for camera view")
        print("\nVECTOR COLORS:")
        print("- RED: Trunk vector (lumbar to cervical)")
        print("- BLUE: Neck vector (cervical to head)")
        print("- GREEN: Left arm vector")
        print("- YELLOW: Right arm vector")
        print("=" * 60)
    else:
        print("ERROR: Import failed!")

if __name__ == "__main__":
    main()
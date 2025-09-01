#!/usr/bin/env python3
"""
Export trunk vector sequence with gravitational reference and angle visualization
Vylepšená verze - zobrazuje trunk vektor, gravitační referenci a úhel mezi nimi
"""

import pickle
import numpy as np
import os
from pathlib import Path
import math

# SMPL-X joint indices for trunk analysis
SMPL_X_JOINT_INDICES = {
    'pelvis': 0,          # Root/pelvis joint
    'spine1': 3,          # Lower spine (lumbar region)  
    'spine2': 6,          # Mid spine
    'spine3': 9,          # Upper spine
    'neck': 12,           # Neck base (cervical region)
    'head': 15,           # Head
}

def create_arrow_mesh(start_point, end_point, arrow_radius=0.01, shaft_segments=8, head_length_ratio=0.2):
    """Create arrow mesh geometry from start to end point"""
    
    # Calculate arrow direction and length
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    
    if length < 1e-6:  # Avoid zero-length arrows
        direction = np.array([0, 0, 1])
        length = 0.1
    else:
        direction = direction / length
    
    # Calculate shaft and head dimensions
    shaft_length = length * (1 - head_length_ratio)
    head_length = length * head_length_ratio
    head_radius = arrow_radius * 2.5
    
    vertices = []
    faces = []
    
    # Create coordinate system for the arrow
    if abs(direction[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, direction)
    
    # Shaft vertices (cylinder)
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = arrow_radius * (np.cos(angle) * right + np.sin(angle) * up)
        
        # Bottom of shaft (at start_point)
        vertices.append(start_point + offset)
        # Top of shaft
        shaft_end = start_point + direction * shaft_length
        vertices.append(shaft_end + offset)
    
    # Create shaft faces (cylinder)
    base_idx = 0
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        # Two triangles per segment
        bottom_curr = base_idx + i * 2
        bottom_next = base_idx + next_i * 2
        top_curr = bottom_curr + 1
        top_next = bottom_next + 1
        
        # Triangle 1
        faces.append([bottom_curr, top_curr, bottom_next])
        # Triangle 2
        faces.append([top_curr, top_next, bottom_next])
    
    # Arrow head vertices (cone)
    head_base_center = start_point + direction * shaft_length
    arrow_tip = end_point
    head_base_idx = len(vertices)
    
    # Add center point for head base
    vertices.append(head_base_center)
    center_idx = len(vertices) - 1
    
    # Head base circle
    for i in range(shaft_segments):
        angle = 2 * np.pi * i / shaft_segments
        offset = head_radius * (np.cos(angle) * right + np.sin(angle) * up)
        vertices.append(head_base_center + offset)
    
    # Add tip vertex
    vertices.append(arrow_tip)
    tip_idx = len(vertices) - 1
    
    # Create head faces
    for i in range(shaft_segments):
        next_i = (i + 1) % shaft_segments
        
        base_curr = head_base_idx + 1 + i
        base_next = head_base_idx + 1 + next_i
        
        # Base triangle (connecting to center)
        faces.append([center_idx, base_next, base_curr])
        
        # Side triangle (connecting to tip)
        faces.append([base_curr, base_next, tip_idx])
    
    return np.array(vertices), np.array(faces)

def create_angle_arc(center, start_vec, end_vec, radius=0.15, segments=16):
    """Create arc mesh to visualize angle between two vectors"""
    
    # Normalize vectors
    start_norm = start_vec / np.linalg.norm(start_vec)
    end_norm = end_vec / np.linalg.norm(end_vec)
    
    # Calculate angle
    dot_product = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    if angle < 0.01:  # Very small angle, skip arc
        return np.array([]), np.array([])
    
    # Create rotation axis (perpendicular to both vectors)
    cross_product = np.cross(start_norm, end_norm)
    if np.linalg.norm(cross_product) < 1e-6:
        return np.array([]), np.array([])
    
    rotation_axis = cross_product / np.linalg.norm(cross_product)
    
    vertices = [center]  # Center point
    faces = []
    
    # Create arc points
    for i in range(segments + 1):
        t = i / segments * angle
        
        # Rodrigues rotation formula
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        
        rotated_vec = (start_norm * cos_t + 
                      np.cross(rotation_axis, start_norm) * sin_t + 
                      rotation_axis * np.dot(rotation_axis, start_norm) * (1 - cos_t))
        
        vertices.append(center + rotated_vec * radius)
    
    # Create triangular faces for arc
    for i in range(segments):
        faces.append([0, i + 1, i + 2])  # Center to arc segments
    
    return np.array(vertices), np.array(faces)

def calculate_trunk_angle_to_gravity(trunk_vector):
    """Calculate angle between trunk vector and gravity (vertical down)"""
    gravity_vector = np.array([0, 0, -1])  # Pointing down (negative Z)
    
    # Normalize vectors
    trunk_norm = trunk_vector / np.linalg.norm(trunk_vector)
    
    # Calculate angle
    dot_product = np.clip(np.dot(trunk_norm, gravity_vector), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, angle_rad

def export_trunk_vectors_with_angle_to_obj(pkl_file, output_dir):
    """Export trunk vector sequence with gravitational reference and angle visualization"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"EXPORTUJI TRUNK VEKTORY S UHLY Z {len(meshes)} SNIMKU DO BLENDERU")
    print("=" * 70)
    
    angle_data = []  # For statistics
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']  # Shape: (117, 3)
        
        # Extract lumbar and cervical joint positions
        lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]  # Lower spine
        cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]  # Neck base
        
        # Calculate trunk vector
        trunk_vector = cervical_joint - lumbar_joint
        trunk_length = np.linalg.norm(trunk_vector)
        
        # Calculate angle to gravity
        angle_deg, angle_rad = calculate_trunk_angle_to_gravity(trunk_vector)
        angle_data.append(angle_deg)
        
        # Create combined mesh with trunk arrow, gravity reference, and angle arc
        all_vertices = []
        all_faces = []
        current_vertex_offset = 0
        
        # 1. Trunk vector arrow (RED)
        trunk_vertices, trunk_faces = create_arrow_mesh(lumbar_joint, cervical_joint, arrow_radius=0.012)
        all_vertices.extend(trunk_vertices)
        all_faces.extend(trunk_faces + current_vertex_offset)
        current_vertex_offset += len(trunk_vertices)
        
        # 2. Gravitational reference vector (BLUE) - from lumbar joint downward
        gravity_end = lumbar_joint + np.array([0, 0, -trunk_length])  # Same length as trunk
        gravity_vertices, gravity_faces = create_arrow_mesh(lumbar_joint, gravity_end, arrow_radius=0.008)
        all_vertices.extend(gravity_vertices)
        all_faces.extend(gravity_faces + current_vertex_offset)
        current_vertex_offset += len(gravity_vertices)
        
        # 3. Angle arc visualization (GREEN)
        arc_vertices, arc_faces = create_angle_arc(lumbar_joint, trunk_vector, np.array([0, 0, -trunk_length]), radius=0.08)
        if len(arc_vertices) > 0:
            all_vertices.extend(arc_vertices)
            all_faces.extend(arc_faces + current_vertex_offset)
        
        # Convert to numpy arrays
        all_vertices = np.array(all_vertices)
        all_faces = np.array(all_faces)
        
        # OBJ filename
        obj_file = output_dir / f"trunk_analysis_{frame_idx:04d}.obj"
        
        with open(obj_file, 'w') as f:
            # Write header with analysis data
            f.write(f"# Trunk Analysis Frame {frame_idx}\n")
            f.write(f"# Lumbar: [{lumbar_joint[0]:.3f}, {lumbar_joint[1]:.3f}, {lumbar_joint[2]:.3f}]\n")
            f.write(f"# Cervical: [{cervical_joint[0]:.3f}, {cervical_joint[1]:.3f}, {cervical_joint[2]:.3f}]\n")
            f.write(f"# Trunk Length: {trunk_length:.3f}m\n")
            f.write(f"# Angle to Gravity: {angle_deg:.1f}°\n")
            f.write(f"# Vertices: {len(all_vertices)}, Faces: {len(all_faces)}\n")
            f.write(f"# RED=Trunk Vector, BLUE=Gravity Reference, GREEN=Angle Arc\n\n")
            
            # Write vertices
            vertex_idx = 0
            
            # Trunk vertices (will be colored RED)
            for i, v in enumerate(trunk_vertices):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                vertex_idx += 1
            
            f.write("g TrunkVector\n")
            for face in trunk_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            # Gravity vertices (will be colored BLUE)
            for i, v in enumerate(gravity_vertices):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("g GravityReference\n")
            for face in gravity_faces:
                f.write(f"f {face[0]+len(trunk_vertices)+1} {face[1]+len(trunk_vertices)+1} {face[2]+len(trunk_vertices)+1}\n")
            
            # Arc vertices (will be colored GREEN)
            if len(arc_vertices) > 0:
                for i, v in enumerate(arc_vertices):
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                f.write("g AngleArc\n")
                arc_offset = len(trunk_vertices) + len(gravity_vertices)
                for face in arc_faces:
                    f.write(f"f {face[0]+arc_offset+1} {face[1]+arc_offset+1} {face[2]+arc_offset+1}\n")
        
        print(f"  Snimek {frame_idx:3d}: Uhel={angle_deg:5.1f}°, Delka trupu={trunk_length:.3f}m")
    
    # Export angle statistics
    stats_file = output_dir / "trunk_angle_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("STATISTIKY ÚHLŮ TRUPU VŮČI GRAVITACI\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Celkový počet snímků: {len(angle_data)}\n")
        f.write(f"Minimální úhel: {min(angle_data):.1f}°\n")
        f.write(f"Maximální úhel: {max(angle_data):.1f}°\n")
        f.write(f"Průměrný úhel: {np.mean(angle_data):.1f}°\n")
        f.write(f"Medián úhlu: {np.median(angle_data):.1f}°\n")
        f.write(f"Směrodatná odchylka: {np.std(angle_data):.1f}°\n\n")
        
        f.write("INTERPRETACE:\n")
        f.write("0° = Perfektně vzpřímený\n")
        f.write("90° = Horizontální trup\n")
        f.write("180° = Hlava dolů\n\n")
        
        # Klasifikace úhlu
        avg_angle = np.mean(angle_data)
        if avg_angle < 30:
            classification = "Vzpřímená pozice"
        elif avg_angle < 60:
            classification = "Mírné ohnutí"
        elif avg_angle < 90:
            classification = "Výrazné ohnutí"
        else:
            classification = "Extrémní ohnutí"
        
        f.write(f"Klasifikace pozice: {classification}\n")
    
    # Create enhanced Blender import script
    blender_script = output_dir / "import_trunk_analysis.py"
    with open(blender_script, 'w') as f:
        f.write(f'''
import bpy
import os
from pathlib import Path

def import_trunk_analysis_sequence():
    """Import trunk analysis sequence with gravitational reference and angles"""
    
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    base_dir = Path(r"{output_dir.absolute()}")
    obj_files = sorted(base_dir.glob("trunk_analysis_*.obj"))
    
    print(f"Found {{len(obj_files)}} trunk analysis OBJ files")
    
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
        print(f"Importing trunk analysis frame {{frame_idx}}...")
        
        # Import OBJ 
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
        
        # Get imported objects (should be 3: TrunkVector, GravityReference, AngleArc)
        imported_objects = bpy.context.selected_objects
        
        if imported_objects:
            for obj in imported_objects:
                if "TrunkVector" in obj.name:
                    obj.name = f"TrunkVector_{{frame_idx:04d}}"
                    if obj.data.materials:
                        obj.data.materials[0] = trunk_mat
                    else:
                        obj.data.materials.append(trunk_mat)
                elif "GravityReference" in obj.name:
                    obj.name = f"GravityRef_{{frame_idx:04d}}"
                    if obj.data.materials:
                        obj.data.materials[0] = gravity_mat
                    else:
                        obj.data.materials.append(gravity_mat)
                elif "AngleArc" in obj.name:
                    obj.name = f"AngleArc_{{frame_idx:04d}}"
                    if obj.data.materials:
                        obj.data.materials[0] = angle_mat
                    else:
                        obj.data.materials.append(angle_mat)
                
                all_objects.append(obj)
                
                # Hide all objects except first frame
                if frame_idx > 0:
                    obj.hide_viewport = True
                    obj.hide_render = True
            
            print(f"  OK: Imported {{len(imported_objects)}} objects for frame {{frame_idx}}")
        else:
            print(f"  ERROR: No objects imported")
    
    print(f"\\nNASTAVUJI ANIMACI TRUNK ANALÝZY...")
    
    # Setup keyframes for visibility
    for frame_idx in range(len(obj_files)):
        for obj in all_objects:
            # Extract frame number from object name
            obj_frame = int(obj.name.split('_')[-1])
            
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
    
    print(f"TRUNK ANALÝZA ANIMACE PŘIPRAVENA!")
    print(f"Timeline: 1-{{len(obj_files)}} snímků")
    print(f"ČERVENÁ = Vektor trupu (lumbar → cervical)")
    print(f"MODRÁ = Gravitační reference (svislice dolů)")
    print(f"ZELENÁ = Úhlový oblouk (úhel mezi vektory)")
    print(f"Stiskněte MEZERNÍK pro spuštění animace")

if __name__ == "__main__":
    import_trunk_analysis_sequence()
''')
    
    print(f"\nSTATISTIKY UHLU:")
    print(f"  Prumery uhel: {np.mean(angle_data):.1f}°")
    print(f"  Rozsah: {min(angle_data):.1f}° - {max(angle_data):.1f}°")
    print(f"  Klasifikace: {classification}")
    
    print(f"\nBLENDER INSTRUKCE:")
    print(f"  1. Otevrte Blender")
    print(f"  2. Jdete do Scripting workspace")
    print(f"  3. Nactete script: {blender_script}")
    print(f"  4. Spustte script")
    print(f"  5. MEZERA = play animace")
    print(f"  6. CERVENA=Trup, MODRA=Gravitace, ZELENA=Uhel")
    
    return output_dir

if __name__ == "__main__":
    export_trunk_vectors_with_angle_to_obj("arm_meshes.pkl", "trunk_analysis_export")
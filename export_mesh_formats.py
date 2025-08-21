#!/usr/bin/env python3
"""
Export 3D meshes to various formats for computational analysis
Demonstrates that pipeline creates real 3D geometry, not just images
"""

import pickle
import numpy as np
from pathlib import Path
import json

def export_to_obj(vertices, faces, output_path):
    """Export mesh to OBJ format (widely supported)"""
    with open(output_path, 'w') as f:
        f.write("# 3D Human Mesh exported from SMPL-X pipeline\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✓ Exported OBJ: {output_path}")

def export_to_ply(vertices, faces, output_path):
    """Export mesh to PLY format (good for scientific computing)"""
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"✓ Exported PLY: {output_path}")

def export_vertices_numpy(vertices, output_path):
    """Export vertices as NumPy array for computational analysis"""
    np.save(output_path, vertices)
    print(f"✓ Exported NumPy vertices: {output_path}")

def export_smplx_parameters(smplx_params, output_path):
    """Export SMPL-X parameters for reuse/analysis"""
    # Convert PyTorch tensors to numpy arrays for JSON serialization
    params_json = {}
    for key, value in smplx_params.items():
        if hasattr(value, 'numpy'):
            params_json[key] = value.numpy().tolist()
        else:
            params_json[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
    
    with open(output_path, 'w') as f:
        json.dump(params_json, f, indent=2)
    
    print(f"✓ Exported SMPL-X parameters: {output_path}")

def calculate_mesh_properties(vertices, faces):
    """Calculate geometric properties for computational analysis"""
    properties = {}
    
    # Basic statistics
    properties['vertex_count'] = len(vertices)
    properties['face_count'] = len(faces)
    
    # Bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    properties['bounding_box'] = {
        'min': min_coords.tolist(),
        'max': max_coords.tolist(),
        'size': (max_coords - min_coords).tolist()
    }
    
    # Center of mass
    properties['center_of_mass'] = np.mean(vertices, axis=0).tolist()
    
    # Surface area approximation (sum of triangle areas)
    total_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        # Triangle area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        total_area += area
    
    properties['surface_area'] = total_area
    
    # Volume approximation (assuming closed mesh)
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        # Tetrahedron volume from origin
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    
    properties['volume'] = abs(volume)
    
    return properties

def export_computational_data():
    """Export all mesh data for computational analysis"""
    
    # Load mesh data
    mesh_file = 'quick_test_output/test_meshes.pkl'
    if not Path(mesh_file).exists():
        print("No mesh data found. Run the pipeline first.")
        return
    
    with open(mesh_file, 'rb') as f:
        mesh_sequence = pickle.load(f)
    
    print("EXPORTING 3D MESH DATA FOR COMPUTATIONAL ANALYSIS")
    print("=" * 60)
    print(f"Processing {len(mesh_sequence)} mesh frames...")
    
    # Create output directory
    export_dir = Path("mesh_exports")
    export_dir.mkdir(exist_ok=True)
    
    # Export each frame
    for i, mesh_data in enumerate(mesh_sequence):
        frame_name = f"frame_{i:03d}"
        print(f"\\nExporting Frame {i+1}:")
        
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        joints = mesh_data['joints']
        smplx_params = mesh_data['smplx_params']
        
        # Export in multiple formats
        export_to_obj(vertices, faces, export_dir / f"{frame_name}.obj")
        export_to_ply(vertices, faces, export_dir / f"{frame_name}.ply")
        export_vertices_numpy(vertices, export_dir / f"{frame_name}_vertices.npy")
        export_vertices_numpy(joints, export_dir / f"{frame_name}_joints.npy")
        export_smplx_parameters(smplx_params, export_dir / f"{frame_name}_params.json")
        
        # Calculate and export geometric properties
        properties = calculate_mesh_properties(vertices, faces)
        with open(export_dir / f"{frame_name}_properties.json", 'w') as f:
            json.dump(properties, f, indent=2)
        
        print(f"  Properties: {properties['vertex_count']} vertices, "
              f"Surface area: {properties['surface_area']:.4f}, "
              f"Volume: {properties['volume']:.4f}")
    
    # Export sequence-level data
    print(f"\\nExporting sequence data...")
    
    # Animation data (vertex positions over time)
    animation_vertices = np.array([mesh['vertices'] for mesh in mesh_sequence])
    np.save(export_dir / "animation_vertices.npy", animation_vertices)
    print(f"✓ Animation vertices: {animation_vertices.shape}")
    
    # Joint trajectories
    animation_joints = np.array([mesh['joints'] for mesh in mesh_sequence])
    np.save(export_dir / "animation_joints.npy", animation_joints)
    print(f"✓ Joint trajectories: {animation_joints.shape}")
    
    # Sequence metadata
    sequence_info = {
        'frame_count': len(mesh_sequence),
        'vertex_count': len(mesh_sequence[0]['vertices']),
        'face_count': len(mesh_sequence[0]['faces']),
        'joint_count': len(mesh_sequence[0]['joints']),
        'formats_exported': ['obj', 'ply', 'npy', 'json'],
        'notes': 'Exported from 3D Human Mesh Pipeline - Ready for computational analysis'
    }
    
    with open(export_dir / "sequence_info.json", 'w') as f:
        json.dump(sequence_info, f, indent=2)
    
    print(f"\\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print(f"Location: {export_dir.absolute()}")
    print(f"Files exported: {len(list(export_dir.glob('*')))} files")
    print(f"")
    print("AVAILABLE FOR COMPUTATIONAL ANALYSIS:")
    print("✓ OBJ files - Import into Blender, Maya, etc.")
    print("✓ PLY files - Scientific computing (Open3D, PCL)")
    print("✓ NumPy arrays - Direct Python analysis")
    print("✓ SMPL-X parameters - Regenerate mesh with different poses")
    print("✓ Geometric properties - Volume, surface area, bounding box")
    print("✓ Animation data - Vertex/joint trajectories over time")
    
    # Show example usage
    print(f"\\nEXAMPLE COMPUTATIONAL USAGE:")
    print("```python")
    print("import numpy as np")
    print("vertices = np.load('mesh_exports/frame_000_vertices.npy')")
    print("joints = np.load('mesh_exports/animation_joints.npy')")
    print("# Now you have full 3D geometry for any calculations")
    print("```")

def demonstrate_mesh_analysis():
    """Demonstrate computational analysis capabilities"""
    
    mesh_file = 'quick_test_output/test_meshes.pkl'
    if not Path(mesh_file).exists():
        print("No mesh data found")
        return
    
    with open(mesh_file, 'rb') as f:
        mesh_sequence = pickle.load(f)
    
    print("\\nDEMONSTRATION: COMPUTATIONAL MESH ANALYSIS")
    print("=" * 50)
    
    # Analyze first frame
    mesh_data = mesh_sequence[0]
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    joints = mesh_data['joints']
    
    print(f"Real 3D Mesh Data:")
    print(f"  Data type: {type(vertices)} - Real NumPy arrays")
    print(f"  Vertices: {vertices.shape} - Each vertex is (x,y,z) in 3D space")
    print(f"  Faces: {faces.shape} - Triangle connectivity")
    print(f"  Memory size: {vertices.nbytes + faces.nbytes} bytes")
    
    # Demonstrate calculations
    print(f"\\nExample Calculations:")
    
    # Distance between specific joints
    if len(joints) > 16:  # SMPL-X joint indices
        left_shoulder = joints[16]   # Left shoulder
        right_shoulder = joints[17]  # Right shoulder
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        print(f"  Shoulder width: {shoulder_distance:.4f} meters")
    
    # Mesh center
    center = np.mean(vertices, axis=0)
    print(f"  Mesh center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    # Height estimation
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    height = max_y - min_y
    print(f"  Estimated height: {height:.4f} meters")
    
    # Surface area calculation
    total_area = 0.0
    for face in faces[:100]:  # Sample calculation
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        total_area += area
    
    estimated_total_area = total_area * (len(faces) / 100)
    print(f"  Estimated surface area: {estimated_total_area:.4f} m²")
    
    print(f"\\n✓ This is REAL 3D geometry suitable for:")
    print("  - Biomechanical analysis")
    print("  - Volume/surface area calculations") 
    print("  - Animation and rigging")
    print("  - 3D printing preparation")
    print("  - Scientific simulations")
    print("  - Engineering analysis")

if __name__ == "__main__":
    export_computational_data()
    demonstrate_mesh_analysis()
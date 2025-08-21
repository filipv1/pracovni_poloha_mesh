#!/usr/bin/env python3
"""
MESH VISUALIZATION AND EXPORT TOOL
Properly visualizes and exports 3D human mesh from PKL files
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
from pathlib import Path

def load_mesh_data(pkl_file):
    """Load mesh data from PKL file"""
    with open(pkl_file, 'rb') as f:
        mesh_sequence = pickle.load(f)
    
    print(f"Loaded {len(mesh_sequence)} frames from {pkl_file}")
    
    # Analyze first mesh
    if mesh_sequence:
        first_mesh = mesh_sequence[0]
        print(f"\nMesh structure:")
        for key in first_mesh.keys():
            if isinstance(first_mesh[key], np.ndarray):
                print(f"  {key}: shape {first_mesh[key].shape}")
            else:
                print(f"  {key}: {type(first_mesh[key])}")
    
    return mesh_sequence

def visualize_mesh_properly(mesh_data, save_path=None, title="3D Human Mesh"):
    """Create proper mesh visualization with triangulated surface"""
    
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 5))
    
    # View 1: Front view with mesh
    ax1 = fig.add_subplot(141, projection='3d')
    
    # Create mesh collection (show actual surface)
    if len(faces) > 0:
        # Get vertex coordinates for each face
        mesh_polys = []
        for face in faces[::10]:  # Use every 10th face for performance
            if len(face) == 3:
                triangle = vertices[face]
                mesh_polys.append(triangle)
        
        # Create 3D polygon collection
        mesh_collection = Poly3DCollection(mesh_polys, alpha=0.3, facecolors='cyan', edgecolors='none')
        ax1.add_collection3d(mesh_collection)
    
    # Add point cloud overlay
    ax1.scatter(vertices[::50, 0], vertices[::50, 1], vertices[::50, 2], 
                c='blue', s=1, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Front View (Mesh)')
    ax1.view_init(elev=0, azim=0)
    
    # Set limits
    max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                         vertices[:,1].max()-vertices[:,1].min(),
                         vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
    mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View 2: Side view
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(vertices[::50, 0], vertices[::50, 1], vertices[::50, 2], 
                c='green', s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Side View')
    ax2.view_init(elev=0, azim=90)
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View 3: Top view
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(vertices[::50, 0], vertices[::50, 1], vertices[::50, 2], 
                c='red', s=1, alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Top View')
    ax3.view_init(elev=90, azim=0)
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View 4: 3D perspective
    ax4 = fig.add_subplot(144, projection='3d')
    
    # Show mesh with different style
    if len(faces) > 0:
        # Wireframe style for better visibility
        for face in faces[::20]:  # Every 20th face
            if len(face) == 3:
                triangle = vertices[face]
                triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
                ax4.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                        'b-', alpha=0.2, linewidth=0.5)
    
    # Add joints if available
    if 'joints' in mesh_data:
        joints = mesh_data['joints']
        ax4.scatter(joints[:22, 0], joints[:22, 1], joints[:22, 2], 
                   c='red', s=50, alpha=1.0, marker='o')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D Perspective')
    ax4.view_init(elev=20, azim=45)
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    return fig

def export_to_obj(mesh_data, output_path):
    """Export mesh to OBJ format for 3D software"""
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    
    with open(output_path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            if len(face) == 3:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Exported to OBJ: {output_path}")

def analyze_mesh_quality(mesh_data):
    """Analyze mesh quality and statistics"""
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    
    print("\n" + "="*50)
    print("MESH ANALYSIS:")
    print("="*50)
    
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    
    # Bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    size = max_coords - min_coords
    
    print(f"\nBounding box:")
    print(f"  Min: [{min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}]")
    print(f"  Max: [{max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f}]")
    print(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    
    # Check if this is SMPL-X mesh
    if len(vertices) == 10475 and len(faces) == 20908:
        print("\n✓ This is a valid SMPL-X mesh!")
        print("  Standard SMPL-X topology detected")
    elif len(vertices) == 6890 and len(faces) == 13776:
        print("\n✓ This is a valid SMPL mesh!")
        print("  Standard SMPL topology detected")
    else:
        print(f"\n⚠ Non-standard mesh topology")
    
    # Surface area estimate
    total_area = 0
    for face in faces[:1000]:  # Sample first 1000 faces
        if len(face) == 3:
            v1, v2, v3 = vertices[face]
            # Cross product for triangle area
            area = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
            total_area += area
    
    estimated_area = total_area * len(faces) / 1000
    print(f"\nEstimated surface area: {estimated_area:.3f} square units")
    
    return {
        'vertices': len(vertices),
        'faces': len(faces),
        'bounding_box': {'min': min_coords.tolist(), 'max': max_coords.tolist()},
        'size': size.tolist(),
        'estimated_area': float(estimated_area)
    }

def main():
    """Main function"""
    
    # Find PKL file
    pkl_files = list(Path('.').glob('*.pkl'))
    
    if not pkl_files:
        print("No PKL files found in current directory")
        return
    
    print("Found PKL files:")
    for i, pkl_file in enumerate(pkl_files):
        print(f"  {i+1}. {pkl_file}")
    
    # Use first PKL file or specified one
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    else:
        pkl_file = pkl_files[0]
    
    print(f"\nProcessing: {pkl_file}")
    
    # Load mesh data
    mesh_sequence = load_mesh_data(pkl_file)
    
    if not mesh_sequence:
        print("No mesh data found in PKL file")
        return
    
    # Process last frame (final mesh)
    final_mesh = mesh_sequence[-1]
    
    # Analyze mesh
    stats = analyze_mesh_quality(final_mesh)
    
    # Create proper visualization
    output_name = Path(pkl_file).stem
    visualize_mesh_properly(
        final_mesh,
        save_path=f"{output_name}_proper_visualization.png",
        title=f"3D Human Mesh - {output_name}"
    )
    
    # Export to OBJ for 3D software
    export_to_obj(final_mesh, f"{output_name}.obj")
    
    # Save statistics
    import json
    with open(f"{output_name}_analysis.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ COMPLETE!")
    print(f"Generated files:")
    print(f"  - {output_name}_proper_visualization.png (multi-view render)")
    print(f"  - {output_name}.obj (3D model for Blender/etc)")
    print(f"  - {output_name}_analysis.json (mesh statistics)")

if __name__ == "__main__":
    main()
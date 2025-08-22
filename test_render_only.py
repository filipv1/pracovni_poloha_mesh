#!/usr/bin/env python3
"""
Test just rendering without full pipeline
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # RUNPOD SAFE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
from pathlib import Path

print("TESTING RENDERING ONLY")
print("=" * 30)

try:
    # Create mock mesh data similar to what SMPL-X produces
    n_vertices = 1000  # Smaller for testing
    vertices = np.random.rand(n_vertices, 3) * 2 - 1  # Random vertices in [-1, 1]
    
    # Create some triangular faces
    n_faces = 500
    faces = []
    for i in range(n_faces):
        # Random triangle from available vertices
        face_indices = np.random.choice(n_vertices, 3, replace=False)
        faces.append(face_indices)
    
    faces = np.array(faces)
    
    print(f"OK Mock mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test matplotlib rendering (RunPod safe approach)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Rendering dense point cloud...")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              c='blue', s=3, alpha=0.9, depthshade=True)
    
    # Set equal aspect and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Test 3D Mesh Render')
    
    # Set aspect ratio and limits
    max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                         vertices[:,1].max()-vertices[:,1].min(),
                         vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
    mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=15, azim=45)
    ax.grid(True, alpha=0.3)
    
    output_path = "final_test_output/test_render.png"
    print(f"Saving to: {output_path}")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close(fig)
    
    print(f"OK Rendering saved successfully: {output_path}")
    
    # Check if file exists and has reasonable size
    if Path(output_path).exists():
        size = Path(output_path).stat().st_size
        print(f"OK File size: {size} bytes")
        if size > 1000:  # Reasonable size check
            print("SUCCESS: Rendering works correctly!")
        else:
            print("WARNING: File size too small")
    else:
        print("ERROR: Output file not found")
        
except Exception as e:
    print(f"ERROR Rendering failed: {e}")
    import traceback
    traceback.print_exc()

print("TEST COMPLETED")
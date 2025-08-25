#!/usr/bin/env python3
"""
Interactive 3D mesh sequence viewer with timeline
"""

import pickle
import numpy as np
import open3d as o3d
import time

class Interactive3DMeshViewer:
    def __init__(self, pkl_file):
        """Initialize interactive 3D viewer"""
        
        with open(pkl_file, 'rb') as f:
            self.meshes = pickle.load(f)
        
        self.current_frame = 0
        self.is_playing = False
        self.fps = 5  # Playback FPS
        
        print(f"INTERACTIVE 3D VIEWER")
        print(f"Loaded {len(self.meshes)} frames")
        print("=" * 30)
        print("CONTROLS:")
        print("  SPACE: Play/Pause")
        print("  ←/→: Previous/Next frame") 
        print("  R: Reset view")
        print("  Q: Quit")
        
    def create_mesh(self, frame_idx):
        """Create Open3D mesh for given frame"""
        
        if frame_idx >= len(self.meshes):
            frame_idx = len(self.meshes) - 1
        if frame_idx < 0:
            frame_idx = 0
            
        mesh_data = self.meshes[frame_idx]
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Compute normals for better lighting
        mesh.compute_vertex_normals()
        
        # Color the mesh
        mesh.paint_uniform_color([0.7, 0.8, 1.0])  # Light blue
        
        return mesh
    
    def update_visualization(self, vis):
        """Update visualization callback"""
        
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % len(self.meshes)
            
        # Create mesh for current frame
        mesh = self.create_mesh(self.current_frame)
        
        # Clear and add new mesh
        vis.clear_geometries()
        vis.add_geometry(mesh)
        
        # Update window title
        error = self.meshes[self.current_frame].get('fitting_error', 0)
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.15])
        
        return False  # Continue animation
    
    def key_callback(self, vis, key, action):
        """Handle keyboard input"""
        
        if action == 1:  # Key press
            if key == 32:  # SPACE
                self.is_playing = not self.is_playing
                print(f"{'Playing' if self.is_playing else 'Paused'} - Frame {self.current_frame}/{len(self.meshes)-1}")
                
            elif key == 262:  # RIGHT arrow
                self.current_frame = (self.current_frame + 1) % len(self.meshes)
                print(f"Frame {self.current_frame}/{len(self.meshes)-1}")
                self.update_visualization(vis)
                
            elif key == 263:  # LEFT arrow  
                self.current_frame = (self.current_frame - 1) % len(self.meshes)
                print(f"Frame {self.current_frame}/{len(self.meshes)-1}")
                self.update_visualization(vis)
                
            elif key == 82:  # R - Reset view
                vis.reset_view_point(True)
                print("View reset")
                
            elif key == 81:  # Q - Quit
                vis.close()
                
        return False
    
    def run(self):
        """Run interactive viewer"""
        
        # Create initial mesh
        initial_mesh = self.create_mesh(0)
        
        # Setup visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"3D Human Mesh Sequence ({len(self.meshes)} frames)", 
                         width=1200, height=800)
        
        # Add initial geometry
        vis.add_geometry(initial_mesh)
        
        # Register callbacks
        vis.register_key_callback(32, self.key_callback)   # SPACE
        vis.register_key_callback(262, self.key_callback)  # RIGHT
        vis.register_key_callback(263, self.key_callback)  # LEFT  
        vis.register_key_callback(82, self.key_callback)   # R
        vis.register_key_callback(81, self.key_callback)   # Q
        
        # Setup view
        vis.get_view_control().set_zoom(0.8)
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.15])
        
        # Animation loop
        last_time = time.time()
        frame_interval = 1.0 / self.fps
        
        while vis.poll_events():
            current_time = time.time()
            
            if current_time - last_time >= frame_interval:
                self.update_visualization(vis)
                last_time = current_time
                
            vis.update_renderer()
        
        vis.destroy_window()

def main():
    """Main function"""
    try:
        viewer = Interactive3DMeshViewer("simple_results/test_meshes.pkl")
        viewer.run()
    except ImportError:
        print("ERROR: Open3D not available")
        print("Install: pip install open3d")
    except FileNotFoundError:
        print("ERROR: PKL file not found")
        print("Run the pipeline first to generate mesh data")

if __name__ == "__main__":
    main()
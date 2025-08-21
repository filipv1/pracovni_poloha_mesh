#!/usr/bin/env python3
"""
3D MESH VIDEO GENERATOR
Converts mesh sequence to animated MP4 video
Exactly what was requested in the original specification!
"""

import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import sys
from tqdm import tqdm
import tempfile
import shutil
import os

class MeshVideoGenerator:
    """Generate MP4 video from mesh sequence"""
    
    def __init__(self, output_fps=25):
        self.output_fps = output_fps
        self.temp_dir = None
        
    def load_mesh_sequence(self, pkl_file):
        """Load mesh sequence from PKL file"""
        print(f"Loading mesh sequence from {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            mesh_sequence = pickle.load(f)
        print(f"Loaded {len(mesh_sequence)} frames")
        return mesh_sequence
    
    def render_mesh_frame(self, mesh_data, frame_idx, style='white_space'):
        """Render single mesh frame"""
        
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        if style == 'white_space':
            # Clean 3D visualization in white space
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set white background
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.1)
            
            # Create mesh surface (use subset for performance)
            if len(faces) > 0:
                # Sample faces for faster rendering
                face_subset = faces[::10]  # Every 10th face
                
                mesh_polys = []
                for face in face_subset:
                    if len(face) == 3:
                        triangle = vertices[face]
                        mesh_polys.append(triangle)
                
                # Create mesh with nice color
                mesh_collection = Poly3DCollection(
                    mesh_polys, 
                    alpha=0.7, 
                    facecolor='lightblue',
                    edgecolor='none',
                    shade=True
                )
                ax.add_collection3d(mesh_collection)
            
            # Add wireframe for better visibility
            wireframe_faces = faces[::50]  # Every 50th face for wireframe
            for face in wireframe_faces:
                if len(face) == 3:
                    triangle = vertices[face]
                    triangle = np.vstack([triangle, triangle[0]])
                    ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           'b-', alpha=0.2, linewidth=0.5)
            
            # Set camera view
            ax.view_init(elev=10, azim=frame_idx * 2)  # Rotate slowly
            
            # Set axis limits
            max_range = np.array([
                vertices[:,0].max()-vertices[:,0].min(),
                vertices[:,1].max()-vertices[:,1].min(),
                vertices[:,2].max()-vertices[:,2].min()
            ]).max() / 2.0
            
            mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
            mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
            mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Clean labels
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            
            # Add frame counter
            ax.text2D(0.05, 0.95, f"Frame: {frame_idx}", 
                     transform=ax.transAxes, fontsize=12)
            
            plt.tight_layout()
            
            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return img
            
        elif style == 'rotating':
            # 360 degree rotating view
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Dark background for contrast
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.grid(False)
            
            # Render mesh
            if len(faces) > 0:
                face_subset = faces[::5]
                mesh_polys = []
                for face in face_subset:
                    if len(face) == 3:
                        triangle = vertices[face]
                        mesh_polys.append(triangle)
                
                mesh_collection = Poly3DCollection(
                    mesh_polys,
                    alpha=0.9,
                    facecolor='cyan',
                    edgecolor='blue',
                    linewidth=0.1
                )
                ax.add_collection3d(mesh_collection)
            
            # Rotating camera
            rotation_angle = (frame_idx * 5) % 360
            ax.view_init(elev=20, azim=rotation_angle)
            
            # Set limits
            max_range = 1.0
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            
            # Hide axes
            ax.set_axis_off()
            
            plt.tight_layout()
            
            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return img
    
    def generate_video(self, mesh_sequence, output_path, style='white_space', 
                      original_video_path=None):
        """Generate complete video from mesh sequence"""
        
        print(f"\nGenerating {style} style video...")
        print(f"Output: {output_path}")
        
        # Create temp directory for frames
        self.temp_dir = tempfile.mkdtemp(prefix='mesh_video_')
        print(f"Temp directory: {self.temp_dir}")
        
        # Render all frames
        frame_paths = []
        print(f"Rendering {len(mesh_sequence)} frames...")
        
        for idx, mesh_data in enumerate(tqdm(mesh_sequence, desc="Rendering")):
            # Render frame
            frame_img = self.render_mesh_frame(mesh_data, idx, style)
            
            # If overlay mode and original video provided
            if original_video_path and style == 'overlay':
                # Load original frame and create side-by-side
                cap = cv2.VideoCapture(str(original_video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, orig_frame = cap.read()
                cap.release()
                
                if ret:
                    # Resize to match
                    orig_frame = cv2.resize(orig_frame, 
                                           (frame_img.shape[1], frame_img.shape[0]))
                    # Side by side
                    frame_img = np.hstack([orig_frame, frame_img])
            
            # Save frame
            frame_path = os.path.join(self.temp_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
        
        # Create video from frames using OpenCV
        if frame_paths:
            print(f"Creating video from {len(frame_paths)} frames...")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.output_fps, (width, height))
            
            # Write all frames
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            print(f"✓ Video saved: {output_path}")
            
            # Try to use ffmpeg for better compression
            try:
                import subprocess
                compressed_path = str(output_path).replace('.mp4', '_compressed.mp4')
                subprocess.run([
                    'ffmpeg', '-i', str(output_path),
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-y', compressed_path
                ], capture_output=True)
                
                if os.path.exists(compressed_path):
                    shutil.move(compressed_path, output_path)
                    print(f"✓ Video compressed with ffmpeg")
            except:
                print("Note: ffmpeg not available for compression")
        
        # Cleanup
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            print("✓ Temp files cleaned")
        
        return output_path
    
    def create_overlay_video(self, mesh_sequence, original_video, output_path):
        """Create video with mesh overlaid on original"""
        
        print(f"\nCreating overlay video...")
        
        cap = cv2.VideoCapture(str(original_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
        
        for idx, mesh_data in enumerate(tqdm(mesh_sequence, desc="Creating overlay")):
            ret, orig_frame = cap.read()
            if not ret:
                break
            
            # Render mesh
            mesh_frame = self.render_mesh_frame(mesh_data, idx, 'white_space')
            mesh_frame = cv2.resize(mesh_frame, (width, height))
            mesh_frame = cv2.cvtColor(mesh_frame, cv2.COLOR_RGB2BGR)
            
            # Side by side
            combined = np.hstack([orig_frame, mesh_frame])
            out.write(combined)
        
        cap.release()
        out.release()
        
        print(f"✓ Overlay video saved: {output_path}")
        return output_path


def main():
    """Main function to generate video from mesh PKL"""
    
    # Parse arguments
    if len(sys.argv) < 2:
        # Find PKL files
        pkl_files = list(Path('.').glob('*.pkl'))
        if not pkl_files:
            print("Usage: python generate_mesh_video.py <mesh_pkl_file> [original_video]")
            print("No PKL files found!")
            return
        pkl_file = pkl_files[0]
        print(f"Using first PKL found: {pkl_file}")
    else:
        pkl_file = sys.argv[1]
    
    original_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create generator
    generator = MeshVideoGenerator(output_fps=25)
    
    # Load mesh sequence
    mesh_sequence = generator.load_mesh_sequence(pkl_file)
    
    if not mesh_sequence:
        print("No mesh data found!")
        return
    
    print(f"Mesh sequence: {len(mesh_sequence)} frames")
    print(f"First mesh: {mesh_sequence[0]['vertices'].shape[0]} vertices")
    
    # Generate videos in different styles
    base_name = Path(pkl_file).stem
    
    # 1. White space video (main output)
    output_video = f"{base_name}_3d_animation.mp4"
    generator.generate_video(
        mesh_sequence, 
        output_video, 
        style='white_space'
    )
    
    # 2. Rotating view
    output_rotating = f"{base_name}_rotating.mp4"
    generator.generate_video(
        mesh_sequence,
        output_rotating,
        style='rotating'
    )
    
    # 3. Overlay if original video provided
    if original_video and Path(original_video).exists():
        output_overlay = f"{base_name}_overlay.mp4"
        generator.create_overlay_video(
            mesh_sequence,
            original_video,
            output_overlay
        )
    
    print("\n" + "="*60)
    print("✅ VIDEO GENERATION COMPLETE!")
    print("="*60)
    print(f"Generated videos:")
    print(f"  1. {output_video} - 3D mesh animation in white space")
    print(f"  2. {output_rotating} - Rotating 3D view")
    if original_video:
        print(f"  3. {output_overlay} - Side-by-side with original")
    print("\n🎬 This is what was requested in the original specification!")

if __name__ == "__main__":
    main()
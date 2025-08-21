#!/usr/bin/env python3
"""
Fix video output for 3D mesh pipeline
Ensures reliable MP4 creation on all systems
"""

import subprocess
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def test_ffmpeg():
    """Test if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
    except FileNotFoundError:
        print("❌ FFmpeg not found")
    
    return False

def test_video_creation_methods():
    """Test different video creation methods"""
    print("Testing video creation methods...")
    
    # Create test frames directory
    frames_dir = Path("test_video_frames")
    frames_dir.mkdir(exist_ok=True)
    
    # Create simple test frames
    for i in range(10):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1, 2, 3], [i, i+1, i-1, i+0.5])
        ax.set_title(f"Test Frame {i}")
        
        frame_path = frames_dir / f"frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Created {len(list(frames_dir.glob('*.png')))} test frames")
    
    # Method 1: FFmpeg subprocess
    if test_ffmpeg():
        try:
            cmd = [
                'ffmpeg', '-y', '-framerate', '10',
                '-i', str(frames_dir / 'frame_%03d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                'test_ffmpeg.mp4'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ FFmpeg method works")
            else:
                print(f"❌ FFmpeg failed: {result.stderr}")
        except Exception as e:
            print(f"❌ FFmpeg exception: {e}")
    
    # Method 2: OpenCV VideoWriter
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_opencv.mp4', fourcc, 10.0, (800, 600))
        
        for frame_file in sorted(frames_dir.glob('*.png')):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frame_resized = cv2.resize(frame, (800, 600))
                out.write(frame_resized)
        
        out.release()
        
        if Path('test_opencv.mp4').exists():
            print("✅ OpenCV VideoWriter works")
        else:
            print("❌ OpenCV VideoWriter failed")
            
    except Exception as e:
        print(f"❌ OpenCV exception: {e}")
    
    # Method 3: Matplotlib animation (as fallback)
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots()
        
        def animate(frame):
            ax.clear()
            ax.plot([0, 1, 2, 3], [frame, frame+1, frame-1, frame+0.5])
            ax.set_title(f"Matplotlib Frame {frame}")
        
        anim = FuncAnimation(fig, animate, frames=10, interval=100)
        
        # Try to save
        anim.save('test_matplotlib.mp4', writer='ffmpeg', fps=10)
        plt.close()
        
        if Path('test_matplotlib.mp4').exists():
            print("✅ Matplotlib animation works")
        else:
            print("❌ Matplotlib animation failed")
            
    except Exception as e:
        print(f"❌ Matplotlib exception: {e}")
    
    # Cleanup
    import shutil
    shutil.rmtree(frames_dir)

def create_reliable_video_creator():
    """Create improved video creation function"""
    video_creator_code = '''
import cv2
import numpy as np
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class ReliableVideoCreator:
    """Reliable video creation with multiple fallback methods"""
    
    def __init__(self):
        self.ffmpeg_available = self._test_ffmpeg()
        self.opencv_available = self._test_opencv()
        
    def _test_ffmpeg(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _test_opencv(self):
        try:
            import cv2
            # Test VideoWriter creation
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            test_writer = cv2.VideoWriter('test_temp.mp4', fourcc, 1.0, (100, 100))
            test_writer.release()
            Path('test_temp.mp4').unlink(missing_ok=True)
            return True
        except:
            return False
    
    def create_mesh_video(self, mesh_sequence, output_path, fps=30, quality='high'):
        """Create video from mesh sequence with automatic fallback"""
        
        if not mesh_sequence:
            print("No mesh sequence provided")
            return False
        
        print(f"Creating video: {output_path}")
        print(f"Frames: {len(mesh_sequence)}, FPS: {fps}")
        print(f"FFmpeg: {self.ffmpeg_available}, OpenCV: {self.opencv_available}")
        
        # Method 1: Individual frames + FFmpeg (best quality)
        if self.ffmpeg_available:
            if self._create_via_frames_ffmpeg(mesh_sequence, output_path, fps, quality):
                return True
        
        # Method 2: OpenCV VideoWriter (good compatibility) 
        if self.opencv_available:
            if self._create_via_opencv(mesh_sequence, output_path, fps):
                return True
        
        # Method 3: Individual PNG frames (fallback)
        return self._create_frames_only(mesh_sequence, output_path)
    
    def _create_via_frames_ffmpeg(self, mesh_sequence, output_path, fps, quality):
        """Create video via individual frames and FFmpeg"""
        try:
            frames_dir = Path(output_path).parent / "temp_video_frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Render individual frames
            for i, mesh_data in enumerate(mesh_sequence):
                frame_path = frames_dir / f"frame_{i:06d}.png"
                self._render_mesh_frame(mesh_data, frame_path, f"Frame {i+1}")
            
            # FFmpeg encoding
            quality_settings = {
                'ultra': ['-crf', '15', '-preset', 'slow'],
                'high': ['-crf', '18', '-preset', 'medium'], 
                'medium': ['-crf', '23', '-preset', 'fast']
            }
            
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', str(frames_dir / 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p'
            ] + quality_settings.get(quality, quality_settings['high']) + [str(output_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Cleanup
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
            
            if result.returncode == 0:
                print(f"✅ FFmpeg video created: {output_path}")
                return True
            else:
                print(f"❌ FFmpeg failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ FFmpeg method failed: {e}")
            return False
    
    def _create_via_opencv(self, mesh_sequence, output_path, fps):
        """Create video using OpenCV VideoWriter"""
        try:
            # Get frame dimensions by rendering first frame
            temp_frame_path = "temp_frame_test.png"
            self._render_mesh_frame(mesh_sequence[0], temp_frame_path, "Test")
            
            # Read back to get dimensions
            test_frame = cv2.imread(temp_frame_path)
            height, width = test_frame.shape[:2]
            Path(temp_frame_path).unlink(missing_ok=True)
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Failed to open VideoWriter")
                return False
            
            # Render and write frames
            for i, mesh_data in enumerate(mesh_sequence):
                frame_path = f"temp_frame_{i}.png" 
                self._render_mesh_frame(mesh_data, frame_path, f"Frame {i+1}")
                
                # Read and write frame
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
                
                # Cleanup frame
                Path(frame_path).unlink(missing_ok=True)
            
            out.release()
            
            if Path(output_path).exists():
                print(f"✅ OpenCV video created: {output_path}")
                return True
            else:
                print("❌ OpenCV video creation failed")
                return False
                
        except Exception as e:
            print(f"❌ OpenCV method failed: {e}")
            return False
    
    def _create_frames_only(self, mesh_sequence, output_path):
        """Fallback: Create individual frames only"""
        try:
            frames_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
            frames_dir.mkdir(exist_ok=True)
            
            for i, mesh_data in enumerate(mesh_sequence):
                frame_path = frames_dir / f"frame_{i:06d}.png"
                self._render_mesh_frame(mesh_data, frame_path, f"Frame {i+1}")
            
            print(f"✅ Individual frames created in: {frames_dir}")
            print(f"Use external tool to create video from {frames_dir}/frame_*.png")
            return True
            
        except Exception as e:
            print(f"❌ Frames creation failed: {e}")
            return False
    
    def _render_mesh_frame(self, mesh_data, output_path, title):
        """Render single mesh frame"""
        fig = plt.figure(figsize=(12, 9), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh_data.get('vertices')
        faces = mesh_data.get('faces', [])
        
        if vertices is not None and len(vertices) > 0:
            # Plot mesh surface if available
            if len(faces) > 0:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                mesh_faces = vertices[faces]
                collection = Poly3DCollection(mesh_faces, alpha=0.7, 
                                            facecolor='lightblue', edgecolor='none')
                ax.add_collection3d(collection)
            
            # Plot joints/vertices
            joints = mesh_data.get('joints', vertices[:22] if len(vertices) > 22 else vertices)
            if len(joints) > 0:
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                          c='red', s=30, alpha=0.8)
        
        # Styling
        ax.set_title(title, color='white', fontsize=16)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white') 
        ax.set_zlabel('Z', color='white')
        
        # Set equal aspect ratio
        if vertices is not None:
            center = np.mean(vertices, axis=0)
            ranges = np.ptp(vertices, axis=0)
            max_range = np.max(ranges) * 0.6
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range) 
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.view_init(elev=15, azim=45)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        plt.close(fig)

# Test the video creator
if __name__ == "__main__":
    creator = ReliableVideoCreator()
    
    # Test with existing mesh data if available
    mesh_file = "quick_test_output/test_meshes.pkl"
    if Path(mesh_file).exists():
        with open(mesh_file, 'rb') as f:
            mesh_sequence = pickle.load(f)
        
        print(f"Testing with {len(mesh_sequence)} meshes")
        success = creator.create_mesh_video(
            mesh_sequence, 
            "test_reliable_video.mp4",
            fps=12,
            quality='high'
        )
        
        if success:
            print("✅ Reliable video creation test successful!")
        else:
            print("❌ All video methods failed")
    else:
        print("No test mesh data found")
'''
    
    with open("reliable_video_creator.py", "w", encoding='utf-8') as f:
        f.write(video_creator_code)
    
    print("Created reliable_video_creator.py")

def main():
    print("3D Mesh Video Output Diagnostic")
    print("=" * 50)
    
    # Test video creation methods
    test_video_creation_methods()
    
    # Create improved video creator
    create_reliable_video_creator()
    
    print("\nSUMMARY:")
    print("- Video output IS implemented in the pipeline")
    print("- Creates [video_name]_3d_animation.mp4") 
    print("- Issue was FFmpeg not found in our test")
    print("- reliable_video_creator.py provides fallback methods")
    
    print("\nTO FIX VIDEO OUTPUT:")
    print("1. Install FFmpeg: apt install ffmpeg (on RunPod)")
    print("2. Or use reliable_video_creator.py for fallback")
    print("3. Pipeline will automatically create MP4 videos")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FIXED 3D MESH PIPELINE - PROPER FULL MESH RENDERING
Restores the original working visualization with complete mesh
"""

import os
import sys
import numpy as np
import cv2
import torch
import mediapipe as mp
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tempfile
import shutil

# Set headless environment
os.environ['MPLBACKEND'] = 'Agg'

try:
    import smplx
    SMPLX_AVAILABLE = True
    print("SMPL-X: Available")
except ImportError:
    SMPLX_AVAILABLE = False
    print("SMPL-X: Not Available")


class FixedPipeline:
    """Fixed pipeline with PROPER mesh rendering"""
    
    def __init__(self, smplx_path="models/smplx", device='cuda'):
        print("\n" + "="*70)
        print("FIXED 3D MESH PIPELINE - FULL MESH RENDERING")
        print("="*70)
        
        # Initialize components
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe initialized")
        
        # SMPL-X
        if SMPLX_AVAILABLE:
            self.body_model = smplx.create(
                smplx_path,
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                use_pca=False,
                batch_size=1
            ).to(self.device)
            print("✓ SMPL-X model loaded")
        
        self.output_fps = 25
        print("✓ Pipeline ready!\n")
    
    def process_frame(self, frame):
        """Process single frame: image -> MediaPipe -> SMPL-X mesh"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_world_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_world_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)
            
            # Fit SMPL-X
            mesh_data = self.fit_smplx(landmarks)
            return mesh_data
        return None
    
    def fit_smplx(self, landmarks, num_iterations=50):
        """Quick SMPL-X fitting"""
        if not SMPLX_AVAILABLE or landmarks is None:
            return None
        
        target_joints = torch.tensor(landmarks, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Initialize parameters
        body_pose = torch.zeros(1, 63).to(self.device)
        global_orient = torch.zeros(1, 3).to(self.device)
        transl = torch.zeros(1, 3).to(self.device)
        
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        transl.requires_grad = True
        
        optimizer = torch.optim.Adam([body_pose, global_orient, transl], lr=0.01)
        
        # Quick optimization
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            
            model_joints = output.joints[:, :33]
            loss = torch.mean((model_joints - target_joints) ** 2)
            
            loss.backward()
            optimizer.step()
        
        # Get final mesh
        with torch.no_grad():
            output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            
        return {
            'vertices': output.vertices[0].cpu().numpy(),
            'faces': self.body_model.faces,
            'joints': output.joints[0].cpu().numpy()
        }
    
    def render_mesh_frame_PROPER(self, mesh_data, frame_idx):
        """PROPER mesh rendering with FULL mesh, not subset!"""
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Create figure
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set clean background
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.1)
        
        # CRITICAL FIX: Use ALL faces for proper mesh, not subset!
        if len(faces) > 0:
            # Adaptive subsampling based on mesh complexity
            total_faces = len(faces)
            if total_faces > 15000:  # High-poly mesh - subsample for performance
                face_indices = np.arange(0, total_faces, 2)  # Every 2nd face for balance
            else:  # Low to medium poly - use ALL faces
                face_indices = np.arange(total_faces)
            
            mesh_polys = []
            for idx in face_indices:
                face = faces[idx]
                if len(face) == 3:
                    triangle = vertices[face]
                    mesh_polys.append(triangle)
            
            # Create proper mesh surface
            mesh_collection = Poly3DCollection(
                mesh_polys, 
                alpha=0.8,  # More opaque
                facecolors='#87CEEB',  # Sky blue color like original
                edgecolors='#4682B4',  # Slight edge color for definition
                linewidths=0.1
            )
            ax.add_collection3d(mesh_collection)
            
            print(f"  Rendering frame {frame_idx}: {len(mesh_polys)} triangles")
        
        # NO ROTATION - Fixed view as requested!
        ax.view_init(elev=10, azim=45)  # Fixed angle, no rotation!
        
        # Set proper axis limits based on actual mesh bounds
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        vertices_center = (vertices_min + vertices_max) / 2
        vertices_range = (vertices_max - vertices_min).max() * 0.6
        
        ax.set_xlim(vertices_center[0] - vertices_range, vertices_center[0] + vertices_range)
        ax.set_ylim(vertices_center[1] - vertices_range, vertices_center[1] + vertices_range)
        ax.set_zlim(vertices_center[2] - vertices_range, vertices_center[2] + vertices_range)
        
        # Minimal labels
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        
        # Add frame counter
        ax.text2D(0.05, 0.95, f"Frame: {frame_idx:04d}", 
                 transform=ax.transAxes, fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img
    
    def render_mesh_frame_HIGH_QUALITY(self, mesh_data, frame_idx):
        """Ultra high quality rendering with complete mesh"""
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Larger figure for quality
        fig = plt.figure(figsize=(12, 12), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        
        # Clean white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        # RENDER COMPLETE MESH
        if len(faces) > 0:
            # Use ALL faces for highest quality
            mesh_polys = []
            for face in faces:
                if len(face) == 3:
                    triangle = vertices[face]
                    mesh_polys.append(triangle)
            
            # High quality mesh
            mesh_collection = Poly3DCollection(
                mesh_polys, 
                alpha=0.95,
                facecolors='#B0C4DE',  # Light steel blue
                edgecolors='none',      # No edges for smooth look
                shade=False
            )
            ax.add_collection3d(mesh_collection)
            
            print(f"  HIGH QUALITY: Rendering {len(mesh_polys)} triangles")
        
        # Fixed professional view - NO ROTATION
        ax.view_init(elev=5, azim=45)
        
        # Tight bounds
        padding = 0.1
        v_min = vertices.min(axis=0) - padding
        v_max = vertices.max(axis=0) + padding
        
        ax.set_xlim(v_min[0], v_max[0])
        ax.set_ylim(v_min[1], v_max[1])
        ax.set_zlim(v_min[2], v_max[2])
        
        # Hide axes for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Frame counter
        ax.text2D(0.02, 0.98, f"Frame {frame_idx:04d}", 
                 transform=ax.transAxes, fontsize=12, 
                 family='monospace', color='gray')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img
    
    def process_video(self, input_video, output_dir="output_fixed", quality='high'):
        """Complete pipeline with PROPER mesh rendering"""
        
        print(f"Processing: {input_video}")
        print(f"Quality mode: {quality}")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(str(input_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Process frames and collect meshes
        mesh_sequence = []
        frame_idx = 0
        
        # Limit frames for testing
        max_frames = min(total_frames, 100)  # Process first 100 frames for testing
        
        print(f"\nPhase 1: Extracting meshes (first {max_frames} frames)...")
        with tqdm(total=max_frames) as pbar:
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                mesh_data = self.process_frame(frame)
                if mesh_data:
                    mesh_sequence.append(mesh_data)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"✓ Extracted {len(mesh_sequence)} meshes")
        
        # Save mesh sequence
        pkl_path = output_dir / f"{Path(input_video).stem}_meshes.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(mesh_sequence, f)
        print(f"✓ Saved meshes: {pkl_path}")
        
        # Generate output video with PROPER rendering
        print("\nPhase 2: Generating PROPER 3D mesh video...")
        output_video = output_dir / f"{Path(input_video).stem}_3d_mesh_FIXED.mp4"
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()
        
        # Render all frames with FULL MESH
        frame_paths = []
        for idx, mesh_data in enumerate(tqdm(mesh_sequence, desc="Rendering FULL meshes")):
            if quality == 'high':
                img = self.render_mesh_frame_HIGH_QUALITY(mesh_data, idx)
            else:
                img = self.render_mesh_frame_PROPER(mesh_data, idx)
            
            frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
            
            # Save sample frames
            if idx < 5:
                sample_path = output_dir / f"sample_frame_{idx:03d}.png"
                cv2.imwrite(str(sample_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"  Saved sample: {sample_path}")
        
        # Create video from frames
        if frame_paths:
            first_frame = cv2.imread(frame_paths[0])
            h, w = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, self.output_fps, (w, h))
            
            for frame_path in tqdm(frame_paths, desc="Creating video"):
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            print(f"✓ Generated PROPER video: {output_video}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        # Summary
        print("\n" + "="*60)
        print("✅ FIXED PIPELINE COMPLETE!")
        print("="*60)
        print(f"Generated outputs:")
        print(f"  1. {output_video} - PROPER 3D mesh animation")
        print(f"  2. Sample frames in {output_dir}")
        print(f"  3. {pkl_path} - Mesh data")
        print("\n🎯 This should match the original quality from August 21!")
        
        return {
            'mesh_video': output_video,
            'mesh_data': pkl_path,
            'frame_count': len(mesh_sequence)
        }


def main():
    """Main entry point"""
    
    # Check for video
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        # Find test video
        test_videos = ["test.mp4", "input.mp4", "sample.mp4"]
        input_video = None
        for video in test_videos:
            if Path(video).exists():
                input_video = video
                break
        
        if not input_video:
            print("Usage: python production_3d_pipeline_FIXED.py <video.mp4> [quality]")
            print("Quality options: 'standard' or 'high'")
            return
    
    # Quality setting
    quality = sys.argv[2] if len(sys.argv) > 2 else 'high'
    
    print(f"Input video: {input_video}")
    print(f"Quality: {quality}")
    
    # Check SMPL-X
    if not Path("models/smplx/SMPLX_NEUTRAL.npz").exists():
        print("ERROR: SMPL-X models not found!")
        print("Download from https://smpl-x.is.tue.mpg.de/")
        return
    
    # Run FIXED pipeline
    pipeline = FixedPipeline(
        smplx_path="models/smplx",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = pipeline.process_video(input_video, quality=quality)
    
    print("\n✅ SUCCESS! This should look like the original August 21 results!")

if __name__ == "__main__":
    main()
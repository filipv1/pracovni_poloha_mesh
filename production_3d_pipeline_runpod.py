#!/usr/bin/env python3
"""
RUNPOD PRODUCTION PIPELINE - NO VISUALIZATION
Optimized for headless GPU servers - generates mesh data without any GUI
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
from pathlib import Path
import json
import pickle
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')

# Set headless environment
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import libraries with proper error handling
try:
    import smplx
    SMPLX_AVAILABLE = True
    print("SMPL-X: Available")
except ImportError:
    SMPLX_AVAILABLE = False
    print("SMPL-X: Not Available")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
    print("Trimesh: Available")
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Trimesh: Not Available")

# DISABLE Open3D completely for headless environment
OPEN3D_AVAILABLE = False
print("Open3D: Disabled for headless RunPod")


class HeadlessVisualizer:
    """Headless visualization - saves data without GUI"""
    
    def __init__(self):
        print("OK Visualizer: Headless mode (no GUI)")
    
    def render_single_mesh(self, mesh_data, title="3D Human Mesh", save_path=None, show_joints=True):
        """Save mesh data without rendering"""
        if save_path:
            # Save as matplotlib figure without display
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            vertices = mesh_data['vertices']
            
            # Simple point cloud visualization
            ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2], 
                      c='blue', s=1, alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            
            # Save without showing
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"OK Saved visualization: {save_path}")
        return save_path


class MediaPipeDetector:
    """MediaPipe pose detection"""
    
    def __init__(self, model_complexity=2):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("OK MediaPipe initialized")
    
    def process_frame(self, frame):
        """Extract 3D pose landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_world_landmarks:
            landmarks = []
            for landmark in results.pose_world_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        return None
    
    def __del__(self):
        self.pose.close()


class SMPLXFitter:
    """SMPL-X mesh fitting from MediaPipe landmarks"""
    
    def __init__(self, smplx_path="models/smplx", device='cuda', gender='neutral'):
        if not SMPLX_AVAILABLE:
            raise ImportError("SMPL-X is required but not installed")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load SMPL-X model
        self.body_model = smplx.create(
            smplx_path,
            model_type='smplx',
            gender=gender,
            num_betas=10,
            use_pca=False,
            batch_size=1
        ).to(self.device)
        
        print(f"OK SMPL-X Model: Loaded successfully ({gender})")
        
        # MediaPipe to SMPL-X joint mapping (simplified)
        self.joint_mapping = {
            0: 0,   # Pelvis
            11: 5,  # Left shoulder
            12: 6,  # Right shoulder
            13: 7,  # Left elbow
            14: 8,  # Right elbow
            15: 9,  # Left wrist
            16: 10, # Right wrist
            23: 2,  # Left hip
            24: 1,  # Right hip
            25: 4,  # Left knee
            26: 3,  # Right knee
            27: 7,  # Left ankle
            28: 8,  # Right ankle
        }
    
    def fit_mesh(self, landmarks_3d, num_iterations=100):
        """Fit SMPL-X mesh to MediaPipe landmarks"""
        if landmarks_3d is None or len(landmarks_3d) < 33:
            return None
        
        # Convert landmarks to torch
        target_joints = torch.tensor(landmarks_3d, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Initialize parameters
        body_pose = torch.zeros(1, 63).to(self.device)
        global_orient = torch.zeros(1, 3).to(self.device)
        transl = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(self.device)
        
        # Make parameters optimizable
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        transl.requires_grad = True
        
        # Optimizer
        optimizer = optim.Adam([body_pose, global_orient, transl], lr=0.01)
        
        # Optimization loop (simplified)
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            
            # Compute loss (simplified)
            model_joints = output.joints[:, :33]
            loss = torch.mean((model_joints - target_joints) ** 2)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"    Optimization iter {i}: Loss={loss.item():.6f}")
        
        # Get final mesh
        with torch.no_grad():
            output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            vertices = output.vertices[0].cpu().numpy()
            faces = self.body_model.faces
            
        print(f"  OK Mesh fitted: {len(vertices)} vertices, {len(faces)} faces")
        
        return {
            'vertices': vertices,
            'faces': faces,
            'joints': output.joints[0].cpu().numpy()
        }


class RunPodPipeline:
    """Main pipeline for RunPod - headless processing"""
    
    def __init__(self, smplx_path="models/smplx", device='cuda'):
        print("\n" + "="*70)
        print("RUNPOD 3D HUMAN MESH PIPELINE - HEADLESS MODE")
        print("="*70)
        
        self.detector = MediaPipeDetector(model_complexity=2)
        self.smplx_fitter = SMPLXFitter(smplx_path, device)
        self.visualizer = HeadlessVisualizer()
        
        print("OK Pipeline Ready!\n")
    
    def process_video(self, video_path, output_dir="outputs", max_frames=None, frame_skip=1):
        """Process video and generate 3D meshes"""
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing: {video_path.name}")
        print(f"  Frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Frame skip: {frame_skip}")
        print()
        
        mesh_sequence = []
        frame_idx = 0
        processed = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # MediaPipe detection
                landmarks = self.detector.process_frame(frame)
                
                if landmarks is not None:
                    print(f"Frame {frame_idx}/{total_frames}")
                    
                    # SMPL-X fitting
                    mesh_data = self.smplx_fitter.fit_mesh(landmarks, num_iterations=100)
                    
                    if mesh_data:
                        mesh_sequence.append(mesh_data)
                        processed += 1
                        
                        # Save sample visualizations (first few frames)
                        if processed <= 3:
                            sample_path = output_dir / f"sample_frame_{processed}.png"
                            self.visualizer.render_single_mesh(
                                mesh_data, 
                                f"Frame {frame_idx}", 
                                str(sample_path)
                            )
            
            frame_idx += 1
        
        cap.release()
        
        # Save mesh sequence
        if mesh_sequence:
            # Save as pickle
            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            with open(mesh_file, 'wb') as f:
                pickle.dump(mesh_sequence, f)
            print(f"\nOK Saved mesh sequence: {mesh_file}")
            
            # Save final frame visualization
            final_vis = output_dir / f"{video_path.stem}_final.png"
            self.visualizer.render_single_mesh(
                mesh_sequence[-1],
                "Final Mesh",
                str(final_vis)
            )
            
            # Save statistics
            stats = {
                'total_frames': total_frames,
                'processed_frames': processed,
                'mesh_count': len(mesh_sequence),
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps
                }
            }
            
            stats_file = output_dir / f"{video_path.stem}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"OK Saved statistics: {stats_file}")
            
            print(f"\n{'='*50}")
            print(f"PIPELINE COMPLETE!")
            print(f"  Processed: {processed} frames")
            print(f"  Meshes generated: {len(mesh_sequence)}")
            print(f"  Output directory: {output_dir}")
            print(f"{'='*50}\n")
            
            return {
                'mesh_file': mesh_file,
                'stats': stats,
                'output_dir': output_dir
            }
        
        return None


def main():
    """Main entry point"""
    
    # Check SMPL-X models
    models_path = Path("models/smplx")
    required_models = ["SMPLX_NEUTRAL.npz"]
    
    for model_file in required_models:
        if not (models_path / model_file).exists():
            print(f"ERROR: Missing SMPL-X model: {model_file}")
            print(f"Please download from https://smpl-x.is.tue.mpg.de/")
            return
    
    print("OK SMPL-X models found")
    
    # Find test video
    test_videos = ["test.mp4", "sample.mp4", "input.mp4", "video.mp4"]
    test_video = None
    
    for video in test_videos:
        if Path(video).exists():
            test_video = video
            break
    
    if not test_video:
        print("ERROR: No test video found")
        print(f"Please place one of these: {', '.join(test_videos)}")
        return
    
    # Initialize pipeline
    pipeline = RunPodPipeline(
        smplx_path="models/smplx",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process video
    results = pipeline.process_video(
        test_video,
        output_dir="runpod_output",
        max_frames=150,  # Limit for testing
        frame_skip=2      # Process every 2nd frame
    )
    
    if results:
        print("SUCCESS! Check 'runpod_output' directory for results")
    else:
        print("Pipeline failed - no meshes generated")


if __name__ == "__main__":
    main()
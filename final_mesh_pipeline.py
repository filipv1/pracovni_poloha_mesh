#!/usr/bin/env python3
"""
Final 3D Human Mesh Pipeline - Production Ready
Optimized for trunk_analysis conda environment with SMPL-X models
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

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


class EnhancedMediaPipeConverter:
    """Enhanced converter from MediaPipe landmarks to SMPL-X joint format"""
    
    def __init__(self):
        # MediaPipe to SMPL-X joint mapping (more accurate)
        self.mp_to_smplx = {
            # Core body joints
            0: 'head',          # nose -> head
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow',    14: 'right_elbow',
            15: 'left_wrist',    16: 'right_wrist',
            23: 'left_hip',      24: 'right_hip',
            25: 'left_knee',     26: 'right_knee',
            27: 'left_ankle',    28: 'right_ankle',
            29: 'left_heel',     30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        # SMPL-X joint order (first 22 joints)
        self.smplx_joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 
            'left_knee', 'right_knee', 'spine2', 
            'left_ankle', 'right_ankle', 'spine3', 
            'left_foot', 'right_foot', 'neck', 
            'left_collar', 'right_collar', 'head',
            'left_shoulder', 'right_shoulder', 
            'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist'
        ]
    
    def convert_to_smplx_joints(self, mediapipe_landmarks):
        """Convert MediaPipe world landmarks to SMPL-X joint positions"""
        if mediapipe_landmarks is None:
            return None
        
        # Extract 3D coordinates from MediaPipe (world coordinates)
        mp_points = np.array([[lm.x, lm.y, lm.z] for lm in mediapipe_landmarks.landmark])
        
        # Initialize SMPL-X joint positions
        smplx_joints = np.zeros((len(self.smplx_joint_names), 3))
        
        # Direct mappings
        direct_mappings = {
            'head': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14, 
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Map direct joints
        joint_indices = {name: i for i, name in enumerate(self.smplx_joint_names)}
        
        for joint_name, mp_idx in direct_mappings.items():
            if joint_name in joint_indices and mp_idx < len(mp_points):
                smplx_joints[joint_indices[joint_name]] = mp_points[mp_idx]
        
        # Calculate derived joints
        self._calculate_derived_joints(mp_points, smplx_joints, joint_indices)
        
        return smplx_joints
    
    def _calculate_derived_joints(self, mp_points, smplx_joints, joint_indices):
        """Calculate positions of joints not directly available from MediaPipe"""
        
        # Pelvis (center of hips)
        if len(mp_points) > 24:
            left_hip = mp_points[23]
            right_hip = mp_points[24]
            smplx_joints[joint_indices['pelvis']] = (left_hip + right_hip) / 2
        
        # Spine chain
        if len(mp_points) > 12:
            shoulder_center = (mp_points[11] + mp_points[12]) / 2
            hip_center = smplx_joints[joint_indices['pelvis']]
            spine_vector = shoulder_center - hip_center
            
            # Distribute spine joints
            smplx_joints[joint_indices['spine1']] = hip_center + spine_vector * 0.2
            smplx_joints[joint_indices['spine2']] = hip_center + spine_vector * 0.5
            smplx_joints[joint_indices['spine3']] = hip_center + spine_vector * 0.8
            smplx_joints[joint_indices['neck']] = hip_center + spine_vector * 0.95
        
        # Feet (below ankles)
        ankle_to_foot_offset = np.array([0, 0, -0.05])
        if smplx_joints[joint_indices['left_ankle']].any():
            smplx_joints[joint_indices['left_foot']] = smplx_joints[joint_indices['left_ankle']] + ankle_to_foot_offset
        if smplx_joints[joint_indices['right_ankle']].any():
            smplx_joints[joint_indices['right_foot']] = smplx_joints[joint_indices['right_ankle']] + ankle_to_foot_offset
        
        # Collar bones
        if smplx_joints[joint_indices['neck']].any():
            neck = smplx_joints[joint_indices['neck']]
            if smplx_joints[joint_indices['left_shoulder']].any():
                left_shoulder = smplx_joints[joint_indices['left_shoulder']]
                smplx_joints[joint_indices['left_collar']] = neck + (left_shoulder - neck) * 0.3
            if smplx_joints[joint_indices['right_shoulder']].any():
                right_shoulder = smplx_joints[joint_indices['right_shoulder']]
                smplx_joints[joint_indices['right_collar']] = neck + (right_shoulder - neck) * 0.3


class ProductionSMPLXFitter:
    """Production-ready SMPL-X mesh fitter with temporal consistency"""
    
    def __init__(self, model_path, device='cpu', gender='neutral'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.gender = gender
        
        # Initialize SMPL-X model
        if SMPLX_AVAILABLE and self._check_model_files():
            try:
                self.smplx_model = smplx.SMPLX(
                    model_path=str(self.model_path),
                    gender=gender,
                    use_face_contour=False,
                    use_hands=False,
                    num_betas=10,
                    num_expression_coeffs=0,
                    create_global_orient=True,
                    create_body_pose=True,
                    create_transl=True
                ).to(self.device)
                self.model_loaded = True
                print(f"SMPL-X Model: Loaded ({gender})")
            except Exception as e:
                print(f"SMPL-X Model: Failed to load - {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print("SMPL-X Model: Not available")
        
        self.converter = EnhancedMediaPipeConverter()
        
        # Temporal consistency
        self.previous_params = None
        self.temporal_weight = 0.1
        
    def _check_model_files(self):
        """Check if SMPL-X model files exist"""
        required_files = [
            f"SMPLX_{self.gender.upper()}.npz",
            f"SMPLX_{self.gender.upper()}.pkl"
        ]
        
        for file in required_files:
            if not (self.model_path / file).exists():
                print(f"Missing SMPL-X file: {file}")
                return False
        return True
    
    def fit_smplx_to_joints(self, target_joints, iterations=150, lr=0.02):
        """Fit SMPL-X model to target joint positions with optimization"""
        
        if not self.model_loaded:
            return self._create_skeleton_mesh(target_joints)
        
        batch_size = 1
        
        # Initialize parameters
        if self.previous_params is not None:
            # Use previous frame as initialization for temporal consistency
            body_pose = self.previous_params['body_pose'].clone().requires_grad_(True)
            global_orient = self.previous_params['global_orient'].clone().requires_grad_(True)
            transl = self.previous_params['transl'].clone().requires_grad_(True)
            betas = self.previous_params['betas'].clone().requires_grad_(True)
        else:
            # Cold start
            body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
            global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
            transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
            betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        # Target joints tensor
        target_tensor = torch.tensor(target_joints, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Optimizer
        optimizer = optim.AdamW([body_pose, global_orient, transl, betas], lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        # Joint confidence weights (higher for more reliable joints)
        joint_weights = torch.ones(len(target_joints), device=self.device)
        reliable_joints = [0, 1, 2, 4, 5, 7, 8, 16, 17]  # pelvis, hips, knees, ankles, shoulders
        joint_weights[reliable_joints] = 2.0
        
        best_loss = float('inf')
        best_params = None
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.smplx_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                betas=betas
            )
            
            # Joint loss (first 22 joints match our target)
            pred_joints = output.joints[:, :len(target_joints)]
            joint_diff = (pred_joints - target_tensor) * joint_weights.view(1, -1, 1)
            joint_loss = torch.mean(torch.sum(joint_diff**2, dim=-1))
            
            # Regularization
            pose_reg = torch.mean(body_pose**2) * 0.0001
            shape_reg = torch.mean(betas**2) * 0.00001
            
            # Temporal consistency (if previous frame exists)
            temporal_loss = 0.0
            if self.previous_params is not None:
                temporal_loss = (
                    torch.mean((body_pose - self.previous_params['body_pose'])**2) * self.temporal_weight +
                    torch.mean((betas - self.previous_params['betas'])**2) * self.temporal_weight * 0.1
                )
            
            # Total loss
            total_loss = joint_loss + pose_reg + shape_reg + temporal_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([body_pose, global_orient, transl, betas], 1.0)
            optimizer.step()
            
            # Track best parameters
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = {
                    'body_pose': body_pose.clone().detach(),
                    'global_orient': global_orient.clone().detach(),
                    'transl': transl.clone().detach(),
                    'betas': betas.clone().detach()
                }
            
            # Learning rate decay
            if i % 30 == 0:
                scheduler.step()
            
            # Progress logging
            if i % 50 == 0 or i == iterations - 1:
                print(f"  Iter {i:3d}: Loss={total_loss.item():.6f}, Joint={joint_loss.item():.6f}")
        
        # Store parameters for next frame
        self.previous_params = best_params
        
        # Generate final mesh
        with torch.no_grad():
            final_output = self.smplx_model(**best_params)
            
            vertices = final_output.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces
            joints = final_output.joints[0].cpu().numpy()
            
            mesh_result = {
                'vertices': vertices,
                'faces': faces,
                'joints': joints,
                'smplx_params': {k: v.cpu().numpy() for k, v in best_params.items()},
                'fitting_loss': best_loss
            }
            
            print(f"  Mesh fitted: {len(vertices)} vertices, loss={best_loss:.6f}")
            return mesh_result
    
    def _create_skeleton_mesh(self, joints):
        """Fallback skeleton when SMPL-X is not available"""
        connections = [
            (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),  # legs
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),                          # spine
            (12, 16), (12, 17), (16, 18), (17, 19), (18, 20), (19, 21)          # arms
        ]
        
        return {
            'vertices': joints,
            'faces': [],
            'joints': joints,
            'connections': connections
        }


class AdvancedVisualizer:
    """Advanced mesh visualization using matplotlib with professional quality"""
    
    def __init__(self, figure_size=(15, 10)):
        self.figure_size = figure_size
        self.fig = None
        self.ax = None
        
        # Visualization settings
        self.mesh_color = [0.8, 0.8, 1.0]     # Light blue
        self.joint_color = [1.0, 0.2, 0.2]   # Red
        self.skeleton_color = [0.2, 0.8, 0.2] # Green
        
        plt.style.use('dark_background')  # Professional dark theme
    
    def setup_3d_plot(self, title="3D Human Mesh"):
        """Setup professional 3D plot"""
        self.fig = plt.figure(figsize=self.figure_size, facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        
        self.ax.set_title(title, fontsize=20, color='white', pad=20)
        self.ax.set_xlabel('X', fontsize=14, color='white')
        self.ax.set_ylabel('Y', fontsize=14, color='white')
        self.ax.set_zlabel('Z', fontsize=14, color='white')
        
        # Dark theme styling
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(True, alpha=0.3)
    
    def render_mesh_data(self, mesh_data, title="3D Mesh", save_path=None, show_wireframe=True):
        """Render mesh data with professional quality"""
        if mesh_data is None:
            print("No mesh data to render")
            return
        
        self.setup_3d_plot(title)
        
        vertices = mesh_data.get('vertices')
        faces = mesh_data.get('faces', [])
        joints = mesh_data.get('joints', vertices if vertices is not None else [])
        connections = mesh_data.get('connections', [])
        
        if vertices is None or len(vertices) == 0:
            print("No vertices to render")
            return
        
        # Render mesh faces
        if len(faces) > 0:
            self._render_mesh_surface(vertices, faces, show_wireframe)
        
        # Render skeleton connections
        if len(connections) > 0:
            self._render_skeleton(vertices, connections)
        
        # Render joints
        if len(joints) > 0:
            self._render_joints(joints)
        
        # Set optimal view
        self._set_optimal_view(vertices)
        
        # Add lighting effect
        self.ax.view_init(elev=10, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            print(f"Visualization saved: {save_path}")
        
        return self.fig, self.ax
    
    def _render_mesh_surface(self, vertices, faces, show_wireframe):
        """Render mesh surface with proper lighting"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create mesh surface
        mesh_faces = vertices[faces]
        
        # Main surface
        surface = Poly3DCollection(mesh_faces, alpha=0.7, facecolor=self.mesh_color,
                                 edgecolor='none', linewidth=0)
        self.ax.add_collection3d(surface)
        
        # Optional wireframe
        if show_wireframe:
            wireframe = Poly3DCollection(mesh_faces, alpha=0.2, facecolor='none',
                                       edgecolor='cyan', linewidth=0.5)
            self.ax.add_collection3d(wireframe)
    
    def _render_skeleton(self, vertices, connections):
        """Render skeleton connections"""
        for start_idx, end_idx in connections:
            if start_idx < len(vertices) and end_idx < len(vertices):
                start, end = vertices[start_idx], vertices[end_idx]
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           color=self.skeleton_color, linewidth=3, alpha=0.8)
    
    def _render_joints(self, joints):
        """Render joint points"""
        joints = np.array(joints)
        self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                       c=[self.joint_color], s=60, alpha=0.9, depthshade=True)
    
    def _set_optimal_view(self, vertices):
        """Set optimal viewing parameters"""
        vertices = np.array(vertices)
        
        # Calculate bounds
        center = np.mean(vertices, axis=0)
        ranges = np.ptp(vertices, axis=0)
        max_range = np.max(ranges) * 0.6
        
        # Set equal aspect ratio
        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    def create_video_from_meshes(self, mesh_sequence, output_path, fps=24, quality='high'):
        """Create high-quality video from mesh sequence"""
        if not mesh_sequence:
            print("No mesh sequence provided")
            return
        
        print(f"Creating video with {len(mesh_sequence)} frames at {fps} FPS...")
        
        # Setup figure for animation
        self.setup_3d_plot("3D Human Mesh Animation")
        
        def animate_frame(frame_idx):
            self.ax.clear()
            mesh_data = mesh_sequence[frame_idx]
            self.render_mesh_data(mesh_data, f"Frame {frame_idx:04d}", show_wireframe=(quality == 'high'))
            self.ax.set_title(f"3D Human Mesh - Frame {frame_idx:04d}", fontsize=16, color='white')
        
        # Create animation
        anim = FuncAnimation(self.fig, animate_frame, frames=len(mesh_sequence),
                           interval=1000//fps, blit=False, repeat=False)
        
        try:
            # Save with high quality settings
            writer_kwargs = {
                'fps': fps,
                'bitrate': 5000 if quality == 'high' else 2000,
                'extra_args': ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
            }
            
            anim.save(output_path, writer='ffmpeg', **writer_kwargs)
            print(f"Video saved: {output_path}")
            
        except Exception as e:
            print(f"Failed to save video: {e}")
            print("Saving individual frames instead...")
            
            frames_dir = Path(output_path).parent / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            for i, mesh_data in enumerate(mesh_sequence):
                frame_path = frames_dir / f"frame_{i:06d}.png"
                self.render_mesh_data(mesh_data, f"Frame {i:04d}", str(frame_path))
                plt.close()


class CompletePipeline:
    """Complete production pipeline for 3D human mesh generation"""
    
    def __init__(self, smplx_model_path="models/smplx", device='cpu'):
        print("Initializing Complete 3D Human Mesh Pipeline")
        print("=" * 60)
        
        self.device = device
        self.smplx_path = Path(smplx_model_path)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Initialize components
        self.mesh_fitter = ProductionSMPLXFitter(self.smplx_path, device)
        self.visualizer = AdvancedVisualizer()
        
        print(f"Device: {device}")
        print(f"SMPL-X Path: {self.smplx_path}")
        print("Pipeline ready!")
    
    def process_video_to_mesh(self, video_path, output_dir="output", 
                            max_frames=None, frame_skip=1, quality='high'):
        """Process video and generate complete 3D mesh sequence"""
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return None
        
        print(f"\nProcessing Video: {video_path}")
        print("-" * 60)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
        
        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video Info:")
        print(f"  Frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame Skip: {frame_skip}")
        print(f"  Quality: {quality}")
        
        # Process frames
        mesh_sequence = []
        successful_frames = 0
        frame_idx = 0
        
        print(f"\nProcessing Frames...")
        print("-" * 60)
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            print(f"Processing frame {frame_idx+1}/{total_frames}...")
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe pose detection
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Convert to SMPL-X joint format
                converter = EnhancedMediaPipeConverter()
                target_joints = converter.convert_to_smplx_joints(results.pose_world_landmarks)
                
                if target_joints is not None:
                    # Fit SMPL-X mesh
                    mesh_data = self.mesh_fitter.fit_smplx_to_joints(target_joints)
                    
                    if mesh_data:
                        mesh_sequence.append(mesh_data)
                        successful_frames += 1
                        
                        print(f"  Success! Mesh generated.")
                        
                        # Save sample frames
                        if successful_frames % 20 == 1:
                            sample_path = output_dir / f"sample_frame_{successful_frames:04d}.png"
                            self.visualizer.render_mesh_data(mesh_data, f"Sample Frame {successful_frames}", 
                                                           str(sample_path), show_wireframe=False)
                            plt.close()
                    else:
                        print(f"  Failed to fit mesh")
                else:
                    print(f"  Failed to convert landmarks")
            else:
                print(f"  No pose detected")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\nProcessing Complete!")
        print(f"Successful frames: {successful_frames}/{total_frames}")
        print("-" * 60)
        
        if mesh_sequence:
            # Save mesh data
            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            with open(mesh_file, 'wb') as f:
                pickle.dump(mesh_sequence, f)
            print(f"Mesh data saved: {mesh_file}")
            
            # Create final visualization of last frame
            if mesh_sequence:
                final_viz = output_dir / f"{video_path.stem}_final_mesh.png"
                self.visualizer.render_mesh_data(mesh_sequence[-1], "Final 3D Mesh", str(final_viz))
                plt.close()
                print(f"Final visualization: {final_viz}")
            
            # Create video animation
            video_output = output_dir / f"{video_path.stem}_3d_mesh.mp4"
            self.visualizer.create_video_from_meshes(mesh_sequence, str(video_output), 
                                                   fps=fps//frame_skip, quality=quality)
            
            print(f"\nGenerated Outputs:")
            print(f"  3D Mesh Video: {video_output}")
            print(f"  Final Mesh Image: {final_viz}")
            print(f"  Mesh Data: {mesh_file}")
            print(f"  Sample Frames: {output_dir}/sample_frame_*.png")
            
            return mesh_sequence
        else:
            print("No meshes were generated!")
            return None


def main():
    """Main function for testing and demonstration"""
    print("🚀 Final 3D Human Mesh Pipeline")
    print("=" * 60)
    
    # Check SMPL-X models
    models_dir = Path("models/smplx")
    if models_dir.exists() and any(models_dir.glob("*.npz")):
        print("✓ SMPL-X models found")
        model_path = str(models_dir)
    else:
        print("✗ SMPL-X models not found")
        print("Please download models from https://smpl-x.is.tue.mpg.de/")
        return
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = CompletePipeline(model_path, device)
    
    # Look for test videos
    test_videos = ["test.mp4", "sample.mp4", "input.mp4", "video.mp4", "demo.mp4"]
    test_video = None
    
    for video in test_videos:
        if Path(video).exists():
            test_video = video
            break
    
    if test_video:
        print(f"\n🎬 Found test video: {test_video}")
        
        # Process video
        mesh_sequence = pipeline.process_video_to_mesh(
            test_video,
            output_dir="final_output",
            max_frames=120,  # ~4 seconds at 30fps
            frame_skip=2,    # Process every 2nd frame
            quality='high'
        )
        
        if mesh_sequence:
            print(f"\n🎉 SUCCESS!")
            print(f"Generated {len(mesh_sequence)} 3D meshes")
            print(f"Check 'final_output/' directory for results")
        else:
            print(f"\n❌ FAILED to generate meshes")
            
    else:
        print(f"\n📁 No test video found")
        print(f"Place a video file (test.mp4, sample.mp4, etc.) in current directory")
        print(f"Then run this script again")


if __name__ == "__main__":
    main()
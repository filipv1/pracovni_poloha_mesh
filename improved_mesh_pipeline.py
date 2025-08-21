#!/usr/bin/env python3
"""
Improved 3D Human Mesh Pipeline for conda trunk_analysis environment
Uses Open3D, SMPL-X and MediaPipe for high-accuracy mesh fitting
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
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

try:
    import smplx
    SMPLX_AVAILABLE = True
    print("✓ SMPL-X available")
except ImportError:
    SMPLX_AVAILABLE = False
    print("✗ SMPL-X not available")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✓ Open3D available")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("✗ Open3D not available")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
    print("✓ Trimesh available")
except ImportError:
    TRIMESH_AVAILABLE = False
    print("✗ Trimesh not available")


class MediaPipeToSMPLXConverter:
    """Enhanced converter for MediaPipe landmarks to SMPL-X parameters"""
    
    def __init__(self):
        # MediaPipe pose landmark mapping to SMPL-X joints
        self.mp_to_smplx_joints = {
            # Core body joints (MediaPipe index -> SMPL-X joint index)
            0: 15,   # nose -> head
            11: 16,  # left_shoulder -> left_shoulder  
            12: 17,  # right_shoulder -> right_shoulder
            13: 18,  # left_elbow -> left_elbow
            14: 19,  # right_elbow -> right_elbow
            15: 20,  # left_wrist -> left_wrist
            16: 21,  # right_wrist -> right_wrist
            23: 1,   # left_hip -> left_hip
            24: 2,   # right_hip -> right_hip
            25: 4,   # left_knee -> left_knee
            26: 5,   # right_knee -> right_knee
            27: 7,   # left_ankle -> left_ankle
            28: 8,   # right_ankle -> right_ankle
        }
        
        # SMPL-X has 55 joints total, we'll map what we can
        self.smplx_joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
    
    def convert_landmarks_to_joints(self, mp_landmarks):
        """Convert MediaPipe landmarks to SMPL-X joint positions"""
        if mp_landmarks is None:
            return None
            
        # Extract 3D coordinates (MediaPipe uses normalized coordinates)
        landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks.landmark])
        
        # Initialize SMPL-X joint array (22 main joints)
        smplx_joints = np.zeros((len(self.smplx_joint_names), 3))
        
        # Map available joints
        for mp_idx, smplx_idx in self.mp_to_smplx_joints.items():
            if mp_idx < len(landmarks_3d) and smplx_idx < len(smplx_joints):
                smplx_joints[smplx_idx] = landmarks_3d[mp_idx]
        
        # Estimate missing joints
        self._estimate_missing_joints(landmarks_3d, smplx_joints)
        
        return smplx_joints
    
    def _estimate_missing_joints(self, mp_landmarks, smplx_joints):
        """Estimate positions of joints not directly available from MediaPipe"""
        
        # Pelvis (center of hips)
        if len(mp_landmarks) > 24:
            smplx_joints[0] = (mp_landmarks[23] + mp_landmarks[24]) / 2  # pelvis
            
        # Spine chain estimation
        if len(mp_landmarks) > 12:
            shoulder_center = (mp_landmarks[11] + mp_landmarks[12]) / 2
            hip_center = smplx_joints[0]
            spine_vector = shoulder_center - hip_center
            
            smplx_joints[3] = hip_center + spine_vector * 0.25  # spine1
            smplx_joints[6] = hip_center + spine_vector * 0.5   # spine2
            smplx_joints[9] = hip_center + spine_vector * 0.75  # spine3
            smplx_joints[12] = hip_center + spine_vector * 0.9  # neck
        
        # Feet estimation (slightly below ankles)
        if smplx_joints[7].any():  # left_ankle
            smplx_joints[10] = smplx_joints[7] + np.array([0, 0, -0.1])  # left_foot
        if smplx_joints[8].any():  # right_ankle  
            smplx_joints[11] = smplx_joints[8] + np.array([0, 0, -0.1])  # right_foot
        
        # Collar bones
        if smplx_joints[16].any() and smplx_joints[12].any():  # left_shoulder and neck
            smplx_joints[13] = smplx_joints[12] + (smplx_joints[16] - smplx_joints[12]) * 0.5  # left_collar
        if smplx_joints[17].any() and smplx_joints[12].any():  # right_shoulder and neck
            smplx_joints[14] = smplx_joints[12] + (smplx_joints[17] - smplx_joints[12]) * 0.5  # right_collar


class SMPLXMeshFitter:
    """High-accuracy SMPL-X mesh fitting using optimization"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        
        if SMPLX_AVAILABLE and os.path.exists(os.path.join(model_path, 'SMPLX_NEUTRAL.npz')):
            try:
                self.smplx_model = smplx.SMPLX(
                    model_path=model_path,
                    gender='neutral',
                    use_face_contour=False,
                    use_hands=False,
                    num_betas=10,
                    num_expression_coeffs=0,
                ).to(self.device)
                self.model_available = True
                print(f"✓ SMPL-X model loaded from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load SMPL-X model: {e}")
                self.model_available = False
        else:
            self.model_available = False
            print("✗ SMPL-X model files not found")
        
        self.converter = MediaPipeToSMPLXConverter()
    
    def fit_mesh(self, target_joints_3d, num_iterations=200, learning_rate=0.01):
        """Fit SMPL-X mesh to target joint positions"""
        
        if not self.model_available:
            return self._create_simple_mesh(target_joints_3d)
        
        batch_size = 1
        
        # Initialize SMPL-X parameters
        body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
        global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        # Convert target joints to tensor
        target_tensor = torch.tensor(target_joints_3d, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Setup optimizer
        params = [body_pose, global_orient, transl, betas]
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Joint weights (higher weight for more reliable joints)
        joint_weights = torch.ones(len(target_joints_3d), device=self.device)
        joint_weights[[1, 2, 16, 17, 4, 5, 7, 8]] *= 2.0  # Emphasize hips, shoulders, knees, ankles
        
        best_loss = float('inf')
        best_params = None
        
        print(f"Fitting SMPL-X mesh with {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.smplx_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                betas=betas
            )
            
            # Get relevant joints (first 22 match our target)
            pred_joints = output.joints[:, :len(target_joints_3d)]
            
            # Joint position loss
            joint_diff = pred_joints - target_tensor
            joint_loss = torch.mean(joint_weights * torch.sum(joint_diff**2, dim=-1))
            
            # Regularization terms
            pose_reg = torch.mean(body_pose**2) * 0.001
            shape_reg = torch.mean(betas**2) * 0.0001
            
            # Total loss
            total_loss = joint_loss + pose_reg + shape_reg
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = {
                    'body_pose': body_pose.clone(),
                    'global_orient': global_orient.clone(),
                    'transl': transl.clone(),
                    'betas': betas.clone()
                }
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration:3d}, Loss: {total_loss.item():.6f}, Joint Error: {joint_loss.item():.6f}")
        
        # Generate final mesh with best parameters
        with torch.no_grad():
            final_output = self.smplx_model(**best_params)
            
            vertices = final_output.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces
            joints = final_output.joints[0].cpu().numpy()
            
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'joints': joints,
                'smplx_params': {k: v.cpu().numpy() for k, v in best_params.items()}
            }
            
            print(f"✓ Mesh fitting completed. Final loss: {best_loss:.6f}")
            return mesh_data
    
    def _create_simple_mesh(self, joints_3d):
        """Create simple stick figure when SMPL-X is not available"""
        connections = [
            (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),  # legs
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),       # spine
            (12, 16), (12, 17), (16, 18), (17, 19),          # arms
            (18, 20), (19, 21)                               # forearms
        ]
        
        return {
            'vertices': joints_3d,
            'faces': [],
            'joints': joints_3d,
            'connections': connections
        }


class AdvancedMeshVisualizer:
    """Advanced 3D mesh visualization using Open3D and matplotlib"""
    
    def __init__(self, use_open3d=True):
        self.use_open3d = use_open3d and OPEN3D_AVAILABLE
        self.fig = None
        self.ax = None
        
        if self.use_open3d:
            print("✓ Using Open3D for high-quality rendering")
        else:
            print("✓ Using matplotlib for basic rendering")
    
    def visualize_mesh(self, mesh_data, title="3D Human Mesh", save_path=None):
        """Visualize 3D mesh with the best available renderer"""
        
        if mesh_data is None:
            print("No mesh data to visualize")
            return
        
        if self.use_open3d and len(mesh_data.get('faces', [])) > 0:
            self._visualize_with_open3d(mesh_data, title, save_path)
        else:
            self._visualize_with_matplotlib(mesh_data, title, save_path)
    
    def _visualize_with_open3d(self, mesh_data, title, save_path):
        """High-quality visualization using Open3D"""
        vertices = mesh_data['vertices']
        faces = mesh_data.get('faces', [])
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        if len(faces) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Compute normals for better lighting
            mesh.compute_vertex_normals()
            
            # Set colors
            mesh.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue
        
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        
        # Joint visualization
        joints = mesh_data.get('joints', vertices[:22] if len(vertices) > 22 else vertices)
        joint_pcd = o3d.geometry.PointCloud()
        joint_pcd.points = o3d.utility.Vector3dVector(joints)
        joint_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red joints
        
        # Visualize
        geometries = [mesh, joint_pcd, coord_frame]
        
        if save_path:
            # Save screenshot
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title, width=1200, height=800)
            for geom in geometries:
                vis.add_geometry(geom)
            vis.run()
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            print(f"✓ Saved visualization to {save_path}")
        else:
            # Interactive visualization
            o3d.visualization.draw_geometries(geometries, window_name=title)
    
    def _visualize_with_matplotlib(self, mesh_data, title, save_path):
        """Fallback visualization using matplotlib"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 9))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        self.ax.set_title(title, fontsize=16)
        
        vertices = mesh_data['vertices']
        faces = mesh_data.get('faces', [])
        connections = mesh_data.get('connections', [])
        
        # Plot vertices
        if len(vertices) > 0:
            self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c='red', s=30, alpha=0.8, label='Joints')
        
        # Plot skeleton connections
        if connections:
            for start_idx, end_idx in connections:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    start, end = vertices[start_idx], vertices[end_idx]
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                               'b-', linewidth=3, alpha=0.7)
        
        # Plot mesh faces (wireframe)
        if len(faces) > 0:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            face_vertices = vertices[faces]
            collection = Poly3DCollection(face_vertices, alpha=0.1, facecolor='lightblue', 
                                        edgecolor='blue', linewidth=0.5)
            self.ax.add_collection3d(collection)
        
        # Set equal aspect ratio
        max_range = np.array(vertices).ptp(axis=0).max() / 2.0
        mid = np.mean(vertices, axis=0)
        self.ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        self.ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        self.ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        else:
            plt.show()
    
    def create_animation_from_sequence(self, mesh_sequence, output_path, fps=30):
        """Create video animation from mesh sequence"""
        if not mesh_sequence:
            print("No mesh sequence provided")
            return
        
        print(f"Creating animation with {len(mesh_sequence)} frames...")
        
        if self.use_open3d:
            self._create_open3d_animation(mesh_sequence, output_path, fps)
        else:
            self._create_matplotlib_animation(mesh_sequence, output_path, fps)
    
    def _create_open3d_animation(self, mesh_sequence, output_path, fps):
        """Create high-quality animation using Open3D"""
        # Save individual frames
        frames_dir = Path(output_path).parent / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for i, mesh_data in enumerate(mesh_sequence):
            frame_path = frames_dir / f"frame_{i:06d}.png"
            self.visualize_mesh(mesh_data, f"Frame {i}", str(frame_path))
        
        # Convert frames to video using ffmpeg
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', str(frames_dir / 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(cmd, check=True)
            print(f"✓ Animation saved to {output_path}")
        except Exception as e:
            print(f"✗ Failed to create video: {e}")
    
    def _create_matplotlib_animation(self, mesh_sequence, output_path, fps):
        """Create basic animation using matplotlib"""
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def animate_frame(frame_idx):
            mesh_data = mesh_sequence[frame_idx]
            self._visualize_with_matplotlib(mesh_data, f"Frame {frame_idx}", None)
            return []
        
        anim = FuncAnimation(self.fig, animate_frame, frames=len(mesh_sequence),
                           interval=1000//fps, blit=False, repeat=True)
        
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"✓ Animation saved to {output_path}")
        except Exception as e:
            print(f"✗ Failed to save animation: {e}")


class CompleteMeshPipeline:
    """Complete pipeline for video processing and mesh generation"""
    
    def __init__(self, smplx_model_path=None, device='cpu'):
        self.device = device
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize components
        self.mesh_fitter = SMPLXMeshFitter(smplx_model_path or "models/smplx", device)
        self.visualizer = AdvancedMeshVisualizer()
        
        print("✓ Complete mesh pipeline initialized")
    
    def process_video(self, video_path, output_dir="output", max_frames=None, skip_frames=1):
        """Process entire video and generate mesh sequence"""
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not video_path.exists():
            print(f"✗ Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return None
        
        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {video_path}")
        print(f"Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
        print(f"Skip frames: {skip_frames}")
        
        mesh_sequence = []
        processed_frames = []
        frame_idx = 0
        processed_count = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe pose detection
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Convert to joint positions
                converter = MediaPipeToSMPLXConverter()
                joint_positions = converter.convert_landmarks_to_joints(results.pose_world_landmarks)
                
                if joint_positions is not None:
                    # Fit SMPL-X mesh
                    mesh_data = self.mesh_fitter.fit_mesh(joint_positions, num_iterations=100)
                    
                    if mesh_data:
                        mesh_sequence.append(mesh_data)
                        processed_frames.append(rgb_frame)
                        processed_count += 1
                        
                        # Save intermediate results
                        if processed_count % 10 == 0:
                            print(f"Processed {processed_count} frames...")
                            
                            # Save sample visualization
                            sample_path = output_dir / f"sample_frame_{processed_count:04d}.png"
                            self.visualizer.visualize_mesh(mesh_data, f"Frame {frame_idx}", str(sample_path))
                    else:
                        print(f"Failed to fit mesh for frame {frame_idx}")
                else:
                    print(f"Failed to convert landmarks for frame {frame_idx}")
            else:
                print(f"No pose detected in frame {frame_idx}")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\n✓ Processing completed: {processed_count} successful frames out of {total_frames}")
        
        if mesh_sequence:
            # Save mesh sequence
            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            with open(mesh_file, 'wb') as f:
                pickle.dump(mesh_sequence, f)
            print(f"✓ Mesh sequence saved to {mesh_file}")
            
            # Create animation
            animation_file = output_dir / f"{video_path.stem}_mesh_animation.mp4"
            self.visualizer.create_animation_from_sequence(mesh_sequence, str(animation_file), fps//skip_frames)
            
            # Create final visualization of last frame
            if mesh_sequence:
                final_viz = output_dir / f"{video_path.stem}_final_mesh.png"
                self.visualizer.visualize_mesh(mesh_sequence[-1], "Final Mesh", str(final_viz))
            
            return mesh_sequence
        else:
            print("✗ No meshes generated")
            return None


def main():
    """Main function for testing"""
    print("Improved 3D Human Mesh Pipeline")
    print("=" * 50)
    
    # Check for SMPL-X models
    models_dir = Path("models/smplx")
    if models_dir.exists() and any(models_dir.glob("*.npz")):
        print(f"✓ SMPL-X models found in {models_dir}")
        model_path = str(models_dir)
    else:
        print("✗ SMPL-X models not found")
        print("Download models from https://smpl-x.is.tue.mpg.de/")
        model_path = None
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    pipeline = CompleteMeshPipeline(model_path, device)
    
    # Look for test video
    test_videos = ["test.mp4", "sample.mp4", "input.mp4", "video.mp4"]
    test_video = None
    
    for video in test_videos:
        if Path(video).exists():
            test_video = video
            break
    
    if test_video:
        print(f"\n🎬 Processing test video: {test_video}")
        mesh_sequence = pipeline.process_video(
            test_video, 
            output_dir="test_output",
            max_frames=90,  # 3 seconds at 30fps
            skip_frames=2   # Process every 2nd frame for speed
        )
        
        if mesh_sequence:
            print("\n🎉 SUCCESS! Generated outputs:")
            print("- test_output/[video_name]_meshes.pkl")
            print("- test_output/[video_name]_mesh_animation.mp4") 
            print("- test_output/[video_name]_final_mesh.png")
            print("- test_output/sample_frame_*.png")
        else:
            print("\n❌ FAILED to generate meshes")
    else:
        print("\n📁 No test video found.")
        print("Place a video file (test.mp4, sample.mp4, etc.) in the current directory")
        print("Then run: pipeline.process_video('your_video.mp4', 'output_dir')")


if __name__ == "__main__":
    main()
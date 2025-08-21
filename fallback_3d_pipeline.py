#!/usr/bin/env python3
"""
Fallback 3D Human Mesh Pipeline
Uses available libraries to create SMPL-X fitting without PyTorch3D/Open3D
Optimized for Python 3.13 compatibility and CPU processing
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
import trimesh
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
    print("✗ SMPL-X not available - using simplified model")

try:
    import easymocap
    EASYMOCAP_AVAILABLE = True
    print("✓ EasyMoCap available")
except ImportError:
    EASYMOCAP_AVAILABLE = False
    print("✗ EasyMoCap not available - using basic fitting")


class MediaPipeToSMPLConverter:
    """Convert MediaPipe 33-point landmarks to SMPL-X parameters"""
    
    def __init__(self):
        # MediaPipe pose landmark indices
        self.mp_to_smpl_mapping = {
            # Torso
            11: 'left_shoulder',   12: 'right_shoulder',
            23: 'left_hip',        24: 'right_hip',
            
            # Arms  
            13: 'left_elbow',      14: 'right_elbow',
            15: 'left_wrist',      16: 'right_wrist',
            
            # Legs
            25: 'left_knee',       26: 'right_knee', 
            27: 'left_ankle',      28: 'right_ankle',
            
            # Head/neck
            0: 'nose',             9: 'left_eye',     10: 'right_eye',
            7: 'left_ear',         8: 'right_ear'
        }
        
        # SMPL-X joint names (simplified)
        self.smpl_joints = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
    
    def convert_landmarks(self, mp_landmarks):
        """Convert MediaPipe landmarks to SMPL joint positions"""
        if mp_landmarks is None:
            return None
            
        # Extract 3D coordinates
        landmarks_3d = []
        for landmark in mp_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z])
        
        landmarks_3d = np.array(landmarks_3d)
        
        # Map to SMPL joints
        smpl_joints_3d = np.zeros((len(self.smpl_joints), 3))
        
        # Basic mapping (simplified)
        if len(landmarks_3d) >= 33:
            # Pelvis (center of hips)
            smpl_joints_3d[0] = (landmarks_3d[23] + landmarks_3d[24]) / 2  # pelvis
            smpl_joints_3d[1] = landmarks_3d[23]  # left_hip
            smpl_joints_3d[2] = landmarks_3d[24]  # right_hip
            
            # Spine approximation
            shoulder_center = (landmarks_3d[11] + landmarks_3d[12]) / 2
            hip_center = smpl_joints_3d[0]
            spine_vector = shoulder_center - hip_center
            
            smpl_joints_3d[3] = hip_center + spine_vector * 0.3   # spine1
            smpl_joints_3d[6] = hip_center + spine_vector * 0.6   # spine2  
            smpl_joints_3d[9] = hip_center + spine_vector * 0.9   # spine3
            
            # Arms and legs
            smpl_joints_3d[4] = landmarks_3d[25]   # left_knee
            smpl_joints_3d[5] = landmarks_3d[26]   # right_knee
            smpl_joints_3d[7] = landmarks_3d[27]   # left_ankle
            smpl_joints_3d[8] = landmarks_3d[28]   # right_ankle
            
            smpl_joints_3d[16] = landmarks_3d[11]  # left_shoulder
            smpl_joints_3d[17] = landmarks_3d[12]  # right_shoulder
            smpl_joints_3d[18] = landmarks_3d[13]  # left_elbow
            smpl_joints_3d[19] = landmarks_3d[14]  # right_elbow
            smpl_joints_3d[20] = landmarks_3d[15]  # left_wrist
            smpl_joints_3d[21] = landmarks_3d[16]  # right_wrist
            
            # Head/neck
            smpl_joints_3d[15] = landmarks_3d[0]   # head (nose)
            smpl_joints_3d[12] = shoulder_center + (landmarks_3d[0] - shoulder_center) * 0.3  # neck
        
        return smpl_joints_3d


class SimpleSMPLFitter:
    """Simplified SMPL-X fitting without complex dependencies"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if SMPLX_AVAILABLE and model_path:
            try:
                self.smpl_model = smplx.SMPLX(
                    model_path=model_path,
                    gender='neutral',
                    use_face_contour=False,
                    use_hands=False
                ).to(self.device)
                self.smpl_available = True
                print(f"✓ SMPL-X model loaded from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load SMPL-X: {e}")
                self.smpl_available = False
        else:
            self.smpl_available = False
            print("Using simplified mesh representation")
        
        self.converter = MediaPipeToSMPLConverter()
    
    def create_simple_mesh(self, joints_3d):
        """Create a simple stick figure mesh from joint positions"""
        if joints_3d is None:
            return None
            
        # Define connections (stick figure)
        connections = [
            # Torso
            (0, 1), (0, 2), (1, 2), (0, 3), (3, 6), (6, 9),  # spine chain
            
            # Left arm
            (9, 16), (16, 18), (18, 20),
            
            # Right arm  
            (9, 17), (17, 19), (19, 21),
            
            # Left leg
            (1, 4), (4, 7), (7, 10),
            
            # Right leg
            (2, 5), (5, 8), (8, 11),
            
            # Head/neck
            (9, 12), (12, 15)
        ]
        
        # Create mesh data
        vertices = joints_3d
        faces = []
        
        # Create simple tubes for each connection
        tube_segments = 8
        for i, (start_idx, end_idx) in enumerate(connections):
            if start_idx < len(vertices) and end_idx < len(vertices):
                start_pos = vertices[start_idx]
                end_pos = vertices[end_idx]
                
                # Simple line representation for now
                # In a full implementation, create cylindrical tubes
                pass
        
        return {
            'vertices': vertices,
            'connections': connections,
            'faces': faces
        }
    
    def fit_to_landmarks(self, mp_landmarks, frame_idx=0):
        """Fit SMPL-X model to MediaPipe landmarks"""
        
        # Convert MediaPipe to joint positions
        joints_3d = self.converter.convert_landmarks(mp_landmarks)
        
        if joints_3d is None:
            return None
        
        if self.smpl_available:
            # Use actual SMPL-X fitting
            return self._fit_smplx_model(joints_3d)
        else:
            # Use simple mesh representation
            return self.create_simple_mesh(joints_3d)
    
    def _fit_smplx_model(self, target_joints):
        """Fit SMPL-X model using optimization"""
        
        # Initialize SMPL-X parameters
        batch_size = 1
        body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
        global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True) 
        transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        # Optimizer
        params = [body_pose, global_orient, transl, betas]
        optimizer = optim.Adam(params, lr=0.01)
        
        target_joints_tensor = torch.tensor(target_joints, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Optimization loop
        for i in range(100):  # Reduced iterations for speed
            optimizer.zero_grad()
            
            # Forward pass
            output = self.smpl_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                betas=betas
            )
            
            # Loss calculation
            pred_joints = output.joints[:, :len(self.converter.smpl_joints)]
            joint_loss = torch.nn.functional.mse_loss(pred_joints, target_joints_tensor)
            
            # Regularization
            pose_reg = torch.mean(body_pose**2)
            shape_reg = torch.mean(betas**2)
            
            total_loss = joint_loss + 0.001 * pose_reg + 0.0001 * shape_reg
            
            total_loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Iteration {i}, Loss: {total_loss.item():.6f}")
        
        # Return mesh
        with torch.no_grad():
            final_output = self.smpl_model(
                body_pose=body_pose,
                global_orient=global_orient, 
                transl=transl,
                betas=betas
            )
            
            vertices = final_output.vertices[0].cpu().numpy()
            faces = self.smpl_model.faces
            
            return {
                'vertices': vertices,
                'faces': faces,
                'joints': final_output.joints[0].cpu().numpy()
            }


class MeshVisualizer:
    """Visualize 3D mesh using matplotlib (fallback for PyTorch3D)"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def setup_plot(self):
        """Setup 3D plot"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
    
    def render_mesh(self, mesh_data, title="3D Human Mesh"):
        """Render mesh data"""
        if mesh_data is None:
            print("No mesh data to render")
            return
            
        if self.fig is None:
            self.setup_plot()
        
        self.ax.clear()
        self.ax.set_title(title)
        
        vertices = mesh_data.get('vertices')
        faces = mesh_data.get('faces', [])
        connections = mesh_data.get('connections', [])
        
        if vertices is not None:
            # Plot vertices
            self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           c='red', s=20, alpha=0.8)
            
            # Plot connections (stick figure)
            if connections:
                for start_idx, end_idx in connections:
                    if start_idx < len(vertices) and end_idx < len(vertices):
                        start = vertices[start_idx]
                        end = vertices[end_idx]
                        self.ax.plot([start[0], end[0]], 
                                   [start[1], end[1]], 
                                   [start[2], end[2]], 'b-', linewidth=2)
            
            # Plot faces (if available)
            if len(faces) > 0 and isinstance(faces, np.ndarray):
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                face_vertices = vertices[faces]
                poly3d = Poly3DCollection(face_vertices, alpha=0.3, facecolor='lightblue')
                self.ax.add_collection3d(poly3d)
        
        # Set equal aspect ratio
        max_range = np.array(vertices).max() - np.array(vertices).min()
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.draw()
    
    def save_frame(self, mesh_data, output_path, frame_idx=0):
        """Save single frame"""
        self.render_mesh(mesh_data, f"Frame {frame_idx}")
        plt.savefig(f"{output_path}_frame_{frame_idx:04d}.png", dpi=150, bbox_inches='tight')
    
    def create_video_from_meshes(self, mesh_sequence, output_path="output_mesh.mp4", fps=30):
        """Create video from sequence of meshes"""
        if not mesh_sequence:
            print("No mesh sequence provided")
            return
            
        self.setup_plot()
        
        def animate(frame_idx):
            mesh_data = mesh_sequence[frame_idx]
            self.render_mesh(mesh_data, f"Frame {frame_idx}")
            return self.ax
        
        anim = FuncAnimation(self.fig, animate, frames=len(mesh_sequence), 
                           interval=1000//fps, blit=False, repeat=True)
        
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"✓ Video saved to {output_path}")
        except Exception as e:
            print(f"✗ Failed to save video: {e}")
            print("Saving individual frames instead...")
            
            for i, mesh_data in enumerate(mesh_sequence):
                self.save_frame(mesh_data, output_path.replace('.mp4', ''), i)


class VideoPipeline:
    """Main pipeline for processing video and creating 3D meshes"""
    
    def __init__(self, smpl_model_path=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.fitter = SimpleSMPLFitter(smpl_model_path)
        self.visualizer = MeshVisualizer()
    
    def process_video(self, video_path, output_path="output", max_frames=None):
        """Process video and create 3D mesh sequence"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return None
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if max_frames:
            frame_count = min(frame_count, max_frames)
        
        print(f"Processing {frame_count} frames at {fps} FPS...")
        
        mesh_sequence = []
        processed_frames = []
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Fit mesh
                mesh_data = self.fitter.fit_to_landmarks(
                    results.pose_world_landmarks, 
                    frame_idx
                )
                
                if mesh_data:
                    mesh_sequence.append(mesh_data)
                    processed_frames.append(frame)
                    
                    if frame_idx % 30 == 0:  # Progress update
                        print(f"Processed frame {frame_idx}/{frame_count}")
            else:
                print(f"No pose detected in frame {frame_idx}")
        
        cap.release()
        
        print(f"✓ Processed {len(mesh_sequence)} frames successfully")
        
        # Create outputs
        if mesh_sequence:
            # Save video
            self.visualizer.create_video_from_meshes(
                mesh_sequence, 
                f"{output_path}_mesh.mp4", 
                fps=fps
            )
            
            # Save mesh data
            with open(f"{output_path}_meshes.pkl", "wb") as f:
                pickle.dump(mesh_sequence, f)
            
            # Create overlay video (simplified)
            self._create_overlay_video(processed_frames, mesh_sequence, f"{output_path}_overlay.mp4")
        
        return mesh_sequence
    
    def _create_overlay_video(self, frames, mesh_sequence, output_path):
        """Create video with mesh overlay on original frames"""
        if not frames or not mesh_sequence:
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for i, (frame, mesh_data) in enumerate(zip(frames, mesh_sequence)):
            # Draw simple skeleton overlay
            overlay_frame = self._draw_skeleton_overlay(frame.copy(), mesh_data)
            out.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✓ Overlay video saved to {output_path}")
    
    def _draw_skeleton_overlay(self, frame, mesh_data):
        """Draw skeleton overlay on frame"""
        if mesh_data is None:
            return frame
            
        vertices = mesh_data.get('vertices')
        connections = mesh_data.get('connections', [])
        
        if vertices is None:
            return frame
            
        height, width = frame.shape[:2]
        
        # Simple projection (normalize coordinates)
        projected_points = []
        for vertex in vertices:
            # Simple orthographic projection
            x = int((vertex[0] + 1) * width / 2)
            y = int((vertex[1] + 1) * height / 2)
            projected_points.append((x, y))
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(projected_points) and end_idx < len(projected_points):
                start_point = projected_points[start_idx]
                end_point = projected_points[end_idx]
                
                if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                    0 <= end_point[0] < width and 0 <= end_point[1] < height):
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw joint points
        for point in projected_points:
            if 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)
        
        return frame


def main():
    """Main function to test the pipeline"""
    print("Fallback 3D Human Mesh Pipeline")
    print("=" * 50)
    
    # Check for SMPL models
    models_dir = Path("models/smplx")
    model_path = None
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*.npz"))
        if model_files:
            model_path = str(models_dir)
            print(f"✓ SMPL-X models found in {model_path}")
        else:
            print("✗ No SMPL-X model files found in models/smplx/")
    else:
        print("✗ models/smplx/ directory not found")
        print("Will use simplified mesh representation")
    
    # Initialize pipeline
    pipeline = VideoPipeline(model_path)
    
    # Test with sample video (if available)
    test_videos = ["test.mp4", "sample.mp4", "input.mp4"]
    test_video = None
    
    for video in test_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if test_video:
        print(f"Processing test video: {test_video}")
        mesh_sequence = pipeline.process_video(test_video, "test_output", max_frames=60)
        
        if mesh_sequence:
            print("✓ Test completed successfully!")
            print("Outputs created:")
            print("- test_output_mesh.mp4 (3D mesh animation)")
            print("- test_output_overlay.mp4 (overlay on original video)")
            print("- test_output_meshes.pkl (mesh data)")
        else:
            print("✗ Test failed - no meshes generated")
    else:
        print("No test video found. Pipeline is ready for use.")
        print("Usage: pipeline.process_video('your_video.mp4', 'output_name')")


if __name__ == "__main__":
    main()
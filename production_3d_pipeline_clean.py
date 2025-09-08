#!/usr/bin/env python3
"""
Production 3D Human Mesh Pipeline
Complete implementation with SMPL-X, Open3D, and MediaPipe
Designed for maximum accuracy and professional visualization
"""

import os
import sys
import time
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
import matplotlib
matplotlib.use('Agg')  # Set headless backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Import libraries with proper error handling
try:
    import smplx
    SMPLX_AVAILABLE = True
    print("SMPL-X: Available")
except ImportError:
    SMPLX_AVAILABLE = False
    print("SMPL-X: Not Available")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("Open3D: Available (v{})".format(o3d.__version__))
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D: Not Available")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
    print("Trimesh: Available")
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Trimesh: Not Available")


class PreciseMediaPipeConverter:
    """High-precision converter from MediaPipe landmarks to SMPL-X format"""
    
    def __init__(self):
        # Enhanced mapping with anatomical accuracy
        self.mp_landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # SMPL-X joint hierarchy (first 22 joints for body)
        self.smplx_joint_tree = {
            0: ('pelvis', None),
            1: ('left_hip', 0), 2: ('right_hip', 0), 3: ('spine1', 0),
            4: ('left_knee', 1), 5: ('right_knee', 2), 6: ('spine2', 3),
            7: ('left_ankle', 4), 8: ('right_ankle', 5), 9: ('spine3', 6),
            10: ('left_foot', 7), 11: ('right_foot', 8), 12: ('neck', 9),
            13: ('left_collar', 12), 14: ('right_collar', 12), 15: ('head', 12),
            16: ('left_shoulder', 13), 17: ('right_shoulder', 14),
            18: ('left_elbow', 16), 19: ('right_elbow', 17),
            20: ('left_wrist', 18), 21: ('right_wrist', 19)
        }
        
        # Quality weights for different landmarks
        self.landmark_confidence = {
            11: 1.0,  # left_shoulder - very reliable
            12: 1.0,  # right_shoulder - very reliable
            23: 0.9,  # left_hip - reliable
            24: 0.9,  # right_hip - reliable
            13: 0.8,  # left_elbow
            14: 0.8,  # right_elbow
            25: 0.8,  # left_knee
            26: 0.8,  # right_knee
            15: 0.7,  # left_wrist
            16: 0.7,  # right_wrist
            27: 0.7,  # left_ankle
            28: 0.7,  # right_ankle
            0: 0.6,   # nose/head
        }
    
    def convert_landmarks_to_smplx(self, mp_landmarks):
        """Convert MediaPipe world landmarks to SMPL-X joint positions"""
        if mp_landmarks is None:
            return None, None
            
        # Extract 3D coordinates (MediaPipe world coordinates are in meters)
        mp_points = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks.landmark])
        
        # Initialize SMPL-X joints and confidence weights
        num_joints = len(self.smplx_joint_tree)
        smplx_joints = np.zeros((num_joints, 3))
        joint_weights = np.zeros(num_joints)
        
        # Direct landmark mappings with confidence
        direct_mappings = {
            15: (0, 0.6),    # head from nose
            16: (11, 1.0),   # left_shoulder
            17: (12, 1.0),   # right_shoulder
            18: (13, 0.8),   # left_elbow
            19: (14, 0.8),   # right_elbow
            20: (15, 0.7),   # left_wrist
            21: (16, 0.7),   # right_wrist
            1: (23, 0.9),    # left_hip
            2: (24, 0.9),    # right_hip
            4: (25, 0.8),    # left_knee
            5: (26, 0.8),    # right_knee
            7: (27, 0.7),    # left_ankle
            8: (28, 0.7),    # right_ankle
        }
        
        # Apply direct mappings
        for joint_idx, (mp_idx, confidence) in direct_mappings.items():
            if mp_idx < len(mp_points):
                smplx_joints[joint_idx] = mp_points[mp_idx]
                joint_weights[joint_idx] = confidence
        
        # Calculate derived joints with anatomical constraints
        self._calculate_anatomical_joints(mp_points, smplx_joints, joint_weights)
        
        return smplx_joints, joint_weights
    
    def _calculate_anatomical_joints(self, mp_points, smplx_joints, joint_weights):
        """Calculate anatomically consistent joint positions"""
        
        # Pelvis as center of hips
        if len(mp_points) > 24:
            left_hip = mp_points[23]
            right_hip = mp_points[24]
            smplx_joints[0] = (left_hip + right_hip) / 2  # pelvis
            joint_weights[0] = 0.95
        
        # Spine chain with proper curvature
        if len(mp_points) > 12 and joint_weights[0] > 0:
            shoulder_center = (mp_points[11] + mp_points[12]) / 2
            pelvis = smplx_joints[0]
            spine_vector = shoulder_center - pelvis
            spine_length = np.linalg.norm(spine_vector)
            spine_unit = spine_vector / spine_length if spine_length > 0 else np.array([0, 0, 1])
            
            # Natural spine curvature
            smplx_joints[3] = pelvis + spine_unit * spine_length * 0.2   # spine1
            smplx_joints[6] = pelvis + spine_unit * spine_length * 0.5   # spine2  
            smplx_joints[9] = pelvis + spine_unit * spine_length * 0.8   # spine3
            smplx_joints[12] = pelvis + spine_unit * spine_length * 0.95 # neck
            
            joint_weights[3:13] = 0.7  # spine chain confidence
        
        # Feet positions (anatomically below ankles)
        foot_offset = np.array([0, 0, -0.08])  # 8cm below ankle
        if joint_weights[7] > 0:  # left_ankle
            smplx_joints[10] = smplx_joints[7] + foot_offset  # left_foot
            joint_weights[10] = joint_weights[7] * 0.8
        if joint_weights[8] > 0:  # right_ankle
            smplx_joints[11] = smplx_joints[8] + foot_offset  # right_foot
            joint_weights[11] = joint_weights[8] * 0.8
        
        # Collar bones (clavicles)
        if joint_weights[12] > 0:  # neck exists
            neck = smplx_joints[12]
            if joint_weights[16] > 0:  # left_shoulder
                collar_vector = smplx_joints[16] - neck
                smplx_joints[13] = neck + collar_vector * 0.4  # left_collar
                joint_weights[13] = 0.6
            if joint_weights[17] > 0:  # right_shoulder
                collar_vector = smplx_joints[17] - neck
                smplx_joints[14] = neck + collar_vector * 0.4  # right_collar
                joint_weights[14] = 0.6


class HighAccuracySMPLXFitter:
    """High-accuracy SMPL-X mesh fitting with advanced optimization"""
    
    def __init__(self, model_path, device='cpu', gender='neutral'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.gender = gender
        
        # Load SMPL-X model
        if SMPLX_AVAILABLE and self._verify_model_files():
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
                    create_transl=True,
                    batch_size=1
                ).to(self.device)
                self.model_ready = True
                print(f"OK SMPL-X Model: Loaded successfully ({gender})")
            except Exception as e:
                print(f"ERROR SMPL-X Model: Load failed - {e}")
                self.model_ready = False
        else:
            self.model_ready = False
            print("ERROR SMPL-X Model: Files not found")
        
        self.converter = PreciseMediaPipeConverter()
        
        # Temporal smoothing
        self.param_history = []
        self.max_history = 5
        self.temporal_alpha = 0.3
        
    def _verify_model_files(self):
        """Verify all required SMPL-X model files exist"""
        required_files = [f"SMPLX_{self.gender.upper()}.npz"]
        
        for file in required_files:
            if not (self.model_path / file).exists():
                return False
        return True
    
    def fit_mesh_to_landmarks(self, target_joints, joint_weights=None, iterations=250):
        """Fit SMPL-X mesh using advanced optimization techniques"""
        
        if not self.model_ready:
            return self._create_wireframe_mesh(target_joints)
        
        batch_size = 1
        
        # Initialize parameters with smart defaults
        if len(self.param_history) > 0:
            # Temporal initialization
            last_params = self.param_history[-1]
            body_pose = last_params['body_pose'].clone().detach().requires_grad_(True)
            global_orient = last_params['global_orient'].clone().detach().requires_grad_(True)
            transl = last_params['transl'].clone().detach().requires_grad_(True)
            betas = last_params['betas'].clone().detach().requires_grad_(True)
        else:
            # Cold start initialization
            body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
            global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
            transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
            betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        # Target preparation
        target_tensor = torch.tensor(target_joints, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if joint_weights is not None:
            weights_tensor = torch.tensor(joint_weights, dtype=torch.float32, device=self.device)
        else:
            weights_tensor = torch.ones(len(target_joints), device=self.device)
        
        # Multi-stage optimization
        optimization_stages = [
            {'lr': 0.1, 'iterations': 80, 'focus': 'global'},      # Global pose and translation
            {'lr': 0.05, 'iterations': 100, 'focus': 'pose'},      # Fine-tune body pose
            {'lr': 0.01, 'iterations': 70, 'focus': 'refinement'}  # Final refinement
        ]
        
        best_loss = float('inf')
        best_params = None
        
        print(f"  Fitting SMPL-X mesh ({iterations} total iterations)...")
        
        for stage_idx, stage in enumerate(optimization_stages):
            # Stage-specific optimizer
            if stage['focus'] == 'global':
                optimizer = optim.Adam([global_orient, transl], lr=stage['lr'])
                stage_params = [body_pose, global_orient, transl, betas]
            elif stage['focus'] == 'pose':
                optimizer = optim.Adam([body_pose, betas], lr=stage['lr'])
                stage_params = [body_pose, global_orient, transl, betas]
            else:  # refinement
                optimizer = optim.AdamW(stage_params, lr=stage['lr'], weight_decay=1e-4)
            
            # Stage optimization loop
            for i in range(stage['iterations']):
                optimizer.zero_grad()
                
                # Forward pass
                smpl_output = self.smplx_model(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                    betas=betas
                )
                
                # Joint loss with confidence weighting
                pred_joints = smpl_output.joints[:, :len(target_joints)]
                joint_diff = pred_joints - target_tensor
                weighted_diff = joint_diff * weights_tensor.view(1, -1, 1)
                joint_loss = torch.mean(torch.sum(weighted_diff**2, dim=-1))
                
                # Regularization terms
                pose_reg = torch.mean(body_pose**2) * 0.0001
                shape_reg = torch.mean(betas**2) * 0.00001
                
                # Temporal consistency
                temporal_loss = 0.0
                if len(self.param_history) > 0:
                    prev_params = self.param_history[-1]
                    temporal_loss = (
                        torch.mean((body_pose - prev_params['body_pose'])**2) * self.temporal_alpha +
                        torch.mean((betas - prev_params['betas'])**2) * self.temporal_alpha * 0.1
                    )
                
                # Total loss
                total_loss = joint_loss + pose_reg + shape_reg + temporal_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(stage_params, max_norm=1.0)
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
                
                # Progress reporting
                if i % 25 == 0:
                    print(f"    Stage {stage_idx+1}, Iter {i:2d}: Loss={total_loss.item():.6f}")
        
        # Update temporal history
        self.param_history.append(best_params)
        if len(self.param_history) > self.max_history:
            self.param_history.pop(0)
        
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
                'fitting_error': best_loss,
                'vertex_count': len(vertices),
                'face_count': len(faces)
            }
            
            print(f"  OK Mesh fitted: {len(vertices)} vertices, {len(faces)} faces, error={best_loss:.6f}")
            return mesh_result
    
    def _create_wireframe_mesh(self, joints):
        """Create wireframe representation when SMPL-X is unavailable"""
        connections = [
            (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
            (12, 16), (12, 17), (16, 18), (17, 19), (18, 20), (19, 21),
            (13, 16), (14, 17)
        ]
        
        return {
            'vertices': joints,
            'faces': np.array([]),
            'joints': joints,
            'connections': connections,
            'is_wireframe': True
        }


class ProfessionalVisualizer:
    """Professional-grade 3D visualization using Open3D and matplotlib"""
    
    def __init__(self, use_open3d=True, theme='dark'):
        self.use_open3d = use_open3d and OPEN3D_AVAILABLE
        self.theme = theme
        
        # Visual settings
        self.mesh_color = np.array([0.8, 0.9, 1.0])      # Light blue
        self.joint_color = np.array([1.0, 0.3, 0.3])     # Red
        self.skeleton_color = np.array([0.2, 1.0, 0.2])  # Green
        
        if theme == 'dark':
            plt.style.use('dark_background')
            self.bg_color = np.array([0.1, 0.1, 0.1])
        else:
            plt.style.use('default')
            self.bg_color = np.array([1.0, 1.0, 1.0])
        
        print(f"OK Visualizer: {'Open3D' if self.use_open3d else 'Matplotlib'} renderer")
    
    def render_single_mesh(self, mesh_data, title="3D Human Mesh", save_path=None, show_joints=True):
        """Render single mesh with professional quality"""
        
        if mesh_data is None:
            print("No mesh data to render")
            return None
        
        # Try Open3D first, fallback to matplotlib if headless issues
        if self.use_open3d and not mesh_data.get('is_wireframe', False):
            try:
                return self._render_with_open3d(mesh_data, title, save_path, show_joints)
            except Exception as e:
                print(f"Open3D visualization failed: {e}")
                print("Using matplotlib fallback...")
                return self._render_with_matplotlib(mesh_data, title, save_path, show_joints)
        else:
            return self._render_with_matplotlib(mesh_data, title, save_path, show_joints)
    
    def _render_with_open3d(self, mesh_data, title, save_path, show_joints):
        """High-quality Open3D rendering"""
        vertices = mesh_data['vertices']
        faces = mesh_data.get('faces', [])
        joints = mesh_data.get('joints', vertices[:22] if len(vertices) > 22 else vertices)
        
        # Create main mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        geometries = []
        
        if len(faces) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(self.mesh_color)
            geometries.append(mesh)
        
        # Add joints if requested
        if show_joints and len(joints) > 0:
            joint_cloud = o3d.geometry.PointCloud()
            joint_cloud.points = o3d.utility.Vector3dVector(joints)
            joint_cloud.paint_uniform_color(self.joint_color)
            
            # Make joints larger
            joint_spheres = []
            for point in joints:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(point)
                sphere.paint_uniform_color(self.joint_color)
                joint_spheres.append(sphere)
            geometries.extend(joint_spheres)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(coord_frame)
        
        if save_path:
            # Save high-quality screenshot (headless-compatible)
            try:
                vis = o3d.visualization.Visualizer()
                # Try to create window with headless fallback
                try:
                    vis.create_window(window_name=title, width=1920, height=1080, visible=False)
                except:
                    # Fallback for headless environment
                    vis.create_window(window_name=title, width=1920, height=1080)
                
                for geom in geometries:
                    vis.add_geometry(geom)
                
                # Set optimal viewpoint (with headless safety)
                ctr = vis.get_view_control()
                if ctr is not None:  # Safety check for headless environment
                    try:
                        ctr.set_zoom(0.8)
                        ctr.set_front([0.0, 0.0, 1.0])
                        ctr.set_lookat([0.0, 0.0, 0.0])
                        ctr.set_up([0.0, 1.0, 0.0])
                    except Exception as e:
                        print(f"Warning: View control failed: {e}")
                        # Use fallback matplotlib rendering
                        vis.destroy_window()
                        return self._render_with_matplotlib(mesh_data, title, save_path, show_joints)
                
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(save_path)
                vis.destroy_window()
                
                print(f"OK High-quality render saved: {save_path}")
                return save_path
                
            except Exception as e:
                print(f"Open3D rendering failed: {e}")
                print("Falling back to matplotlib rendering...")
                # Fallback to matplotlib
                return self._render_with_matplotlib(mesh_data, title, save_path, show_joints)
        else:
            # Interactive visualization
            o3d.visualization.draw_geometries(
                geometries,
                window_name=title,
                width=1200,
                height=900
            )
            return None
    
    def _render_with_matplotlib(self, mesh_data, title, save_path, show_joints):
        """Professional matplotlib rendering"""
        fig = plt.figure(figsize=(16, 12), facecolor=self.bg_color)
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh_data['vertices']
        faces = mesh_data.get('faces', [])
        connections = mesh_data.get('connections', [])
        joints = mesh_data.get('joints', vertices)
        
        # Plot mesh faces
        if len(faces) > 0:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            mesh_faces = vertices[faces]
            collection = Poly3DCollection(mesh_faces, alpha=0.7, facecolor=self.mesh_color,
                                        edgecolor='none', linewidth=0)
            ax.add_collection3d(collection)
            
            # Add wireframe for detail
            wireframe = Poly3DCollection(mesh_faces, alpha=0.1, facecolor='none',
                                       edgecolor='cyan', linewidth=0.3)
            ax.add_collection3d(wireframe)
        
        # Plot skeleton connections
        if len(connections) > 0:
            for start_idx, end_idx in connections:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    start, end = vertices[start_idx], vertices[end_idx]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                           color=self.skeleton_color, linewidth=4, alpha=0.8)
        
        # Plot joints
        if show_joints and len(joints) > 0:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                      c=[self.joint_color], s=100, alpha=0.9, depthshade=True)
        
        # Professional styling
        ax.set_title(title, fontsize=20, color='white', pad=30)
        ax.set_xlabel('X (meters)', fontsize=14, color='white')
        ax.set_ylabel('Y (meters)', fontsize=14, color='white')
        ax.set_zlabel('Z (meters)', fontsize=14, color='white')
        
        # Set equal aspect ratio
        all_points = np.vstack([vertices, joints]) if len(joints) > 0 else vertices
        center = np.mean(all_points, axis=0)
        ranges = np.ptp(all_points, axis=0)
        max_range = np.max(ranges) * 0.6
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # Optimize viewing angle
        ax.view_init(elev=15, azim=45)
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
            print(f"OK Professional render saved: {save_path}")
            plt.close(fig)
            return save_path
        
        return fig, ax
    
    def create_professional_video(self, mesh_sequence, output_path, fps=30, quality='ultra'):
        """Create professional-quality video from mesh sequence"""
        
        if not mesh_sequence:
            print("No mesh sequence provided")
            return
        
        print(f"Creating professional video ({quality} quality)...")
        print(f"Frames: {len(mesh_sequence)}, FPS: {fps}")
        
        if self.use_open3d and quality in ['ultra', 'high']:
            self._create_open3d_video(mesh_sequence, output_path, fps, quality)
        else:
            self._create_matplotlib_video(mesh_sequence, output_path, fps, quality)
    
    def _create_open3d_video(self, mesh_sequence, output_path, fps, quality):
        """Ultra-high quality video using Open3D"""
        
        frames_dir = Path(output_path).parent / "temp_frames"
        frames_dir.mkdir(exist_ok=True)
        
        print("Rendering individual frames with Open3D...")
        
        for i, mesh_data in enumerate(mesh_sequence):
            frame_path = frames_dir / f"frame_{i:06d}.png"
            self._render_with_open3d(mesh_data, f"Frame {i+1:04d}", str(frame_path), show_joints=True)
            
            if i % 10 == 0:
                print(f"  Rendered {i+1}/{len(mesh_sequence)} frames")
        
        # Convert to video using ffmpeg
        self._frames_to_video(frames_dir, output_path, fps, quality)
        
        # Cleanup
        import shutil
        shutil.rmtree(frames_dir)
        print("OK Temporary frames cleaned up")
    
    def _create_matplotlib_video(self, mesh_sequence, output_path, fps, quality):
        """High-quality video using matplotlib animation"""
        
        fig = plt.figure(figsize=(16, 12), facecolor=self.bg_color)
        ax = fig.add_subplot(111, projection='3d')
        
        def animate_frame(frame_idx):
            ax.clear()
            mesh_data = mesh_sequence[frame_idx]
            self._render_with_matplotlib(mesh_data, f"Frame {frame_idx+1:04d}", None, True)
        
        print("Creating matplotlib animation...")
        anim = FuncAnimation(fig, animate_frame, frames=len(mesh_sequence),
                           interval=1000//fps, blit=False, repeat=False)
        
        # High-quality export settings
        writer_kwargs = {
            'fps': fps,
            'bitrate': 8000 if quality == 'ultra' else 5000 if quality == 'high' else 2000,
            'extra_args': ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']
        }
        
        try:
            anim.save(output_path, writer='ffmpeg', **writer_kwargs)
            print(f"OK Professional video saved: {output_path}")
        except Exception as e:
            print(f"ERROR Video creation failed: {e}")
        
        plt.close(fig)
    
    def _frames_to_video(self, frames_dir, output_path, fps, quality):
        """Convert frame sequence to video using ffmpeg"""
        
        quality_settings = {
            'ultra': ['-crf', '15', '-preset', 'slow'],
            'high': ['-crf', '18', '-preset', 'medium'],
            'medium': ['-crf', '23', '-preset', 'fast']
        }
        
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p'
        ] + quality_settings.get(quality, quality_settings['medium']) + [str(output_path)]
        
        try:
            import subprocess
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"OK Professional video created: {output_path}")
        except Exception as e:
            print(f"ERROR Video encoding failed: {e}")


class MasterPipeline:
    """Master pipeline orchestrating the complete 3D human mesh workflow"""
    
    def __init__(self, smplx_path="models/smplx", device='auto', gender='neutral'):
        
        print("ROCKET Initializing Master 3D Human Mesh Pipeline")
        print("=" * 70)
        
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.smplx_path = Path(smplx_path)
        self.gender = gender
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        # Initialize core components
        self.mesh_fitter = HighAccuracySMPLXFitter(self.smplx_path, self.device, gender)
        self.visualizer = ProfessionalVisualizer(use_open3d=True, theme='dark')
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'meshes_generated': 0,
            'average_fitting_error': 0.0,
            'processing_time': 0.0
        }
        
        print(f"OK Device: {self.device}")
        print(f"OK SMPL-X Path: {self.smplx_path}")
        print(f"OK Gender: {gender}")
        print("OK Master Pipeline Ready!")
    
    def execute_full_pipeline(self, video_path, output_dir="production_output", 
                            max_frames=None, frame_skip=1, quality='ultra'):
        """Execute complete pipeline from video to final visualization"""
        
        import time
        start_time = time.time()
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not video_path.exists():
            print(f"X Video file not found: {video_path}")
            return None
        
        print(f"\nMOVIE PROCESSING: {video_path}")
        print("=" * 70)
        
        # Video analysis
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"X Failed to open video")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"CHART Video Analysis:")
        print(f"   Total Frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Frame Skip: {frame_skip}")
        print(f"   Quality: {quality}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
        
        # Process video frames
        print(f"\nGEAR  PROCESSING FRAMES...")
        print("-" * 70)
        
        mesh_sequence = []
        frame_idx = 0
        successful_frames = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            progress = (frame_idx + 1) / total_frames * 100
            print(f"Frame {frame_idx+1:4d}/{total_frames} ({progress:5.1f}%)")
            
            # MediaPipe pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Convert landmarks to SMPL-X format
                converter = PreciseMediaPipeConverter()
                joints, weights = converter.convert_landmarks_to_smplx(results.pose_world_landmarks)
                
                if joints is not None:
                    # Fit SMPL-X mesh
                    mesh_data = self.mesh_fitter.fit_mesh_to_landmarks(joints, weights)
                    
                    if mesh_data and not mesh_data.get('is_wireframe', False):
                        mesh_sequence.append(mesh_data)
                        successful_frames += 1
                        
                        # Update statistics
                        if 'fitting_error' in mesh_data:
                            self.stats['average_fitting_error'] += mesh_data['fitting_error']
                        
                        print(f"  OK Mesh generated: {mesh_data['vertex_count']} vertices")
                        
                        # Save sample frames
                        if successful_frames in [1, successful_frames//4, successful_frames//2, successful_frames*3//4]:
                            sample_path = output_dir / f"sample_frame_{successful_frames:04d}.png"
                            self.visualizer.render_single_mesh(mesh_data, f"Sample Frame {successful_frames}",
                                                             str(sample_path), show_joints=True)
                    else:
                        print(f"  ERROR Mesh fitting failed")
                else:
                    print(f"  ERROR Landmark conversion failed")
            else:
                print(f"  ERROR No pose detected")
            
            frame_idx += 1
        
        cap.release()
        
        # Final statistics
        processing_time = time.time() - start_time
        self.stats['frames_processed'] = frame_idx
        self.stats['meshes_generated'] = successful_frames
        self.stats['processing_time'] = processing_time
        
        if successful_frames > 0:
            self.stats['average_fitting_error'] /= successful_frames
        
        print(f"\nGRAPH PROCESSING COMPLETE!")
        print("-" * 70)
        print(f"   Processed Frames: {frame_idx}")
        print(f"   Successful Meshes: {successful_frames}")
        print(f"   Success Rate: {successful_frames/frame_idx*100:.1f}%")
        print(f"   Average Fitting Error: {self.stats['average_fitting_error']:.6f}")
        print(f"   Processing Time: {processing_time:.1f} seconds")
        print(f"   FPS: {frame_idx/processing_time:.2f}")
        
        if mesh_sequence:
            print(f"\nART GENERATING OUTPUTS...")
            print("-" * 70)
            
            # Save mesh data with metadata
            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            pkl_data = {
                'mesh_sequence': mesh_sequence,
                'metadata': {
                    'fps': fps,
                    'total_frames': len(mesh_sequence),
                    'video_duration_seconds': len(mesh_sequence) / fps,
                    'frame_skip': frame_skip,
                    'video_filename': video_path.name,
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'video_resolution': f"{width}x{height}",
                    'original_total_frames': total_frames
                }
            }
            with open(mesh_file, 'wb') as f:
                pickle.dump(pkl_data, f)
            print(f"OK Mesh data with metadata: {mesh_file}")
            
            # Save statistics
            stats_file = output_dir / f"{video_path.stem}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"OK Statistics: {stats_file}")
            
            # Create final high-quality visualization
            final_mesh_img = output_dir / f"{video_path.stem}_final_mesh.png"
            self.visualizer.render_single_mesh(mesh_sequence[-1], "Final 3D Human Mesh", 
                                             str(final_mesh_img), show_joints=True)
            print(f"OK Final mesh: {final_mesh_img}")
            
            # Create professional video
            output_video = output_dir / f"{video_path.stem}_3d_animation.mp4"
            self.visualizer.create_professional_video(mesh_sequence, str(output_video), 
                                                     fps=fps//frame_skip, quality=quality)
            print(f"OK 3D Animation: {output_video}")
            
            print(f"\nPARTY SUCCESS! All outputs generated in: {output_dir}")
            
            return {
                'mesh_sequence': mesh_sequence,
                'mesh_file': mesh_file,
                'video_file': output_video,
                'stats': self.stats,
                'output_dir': output_dir
            }
        else:
            print(f"\nX FAILED: No meshes were generated")
            return None


def main():
    """Main execution function"""
    print("ROCKET PRODUCTION 3D HUMAN MESH PIPELINE")
    print("=" * 80)
    
    # Verify SMPL-X models
    models_dir = Path("models/smplx")
    if not models_dir.exists() or not any(models_dir.glob("*.npz")):
        print("X SMPL-X models not found!")
        print("Please download models from: https://smpl-x.is.tue.mpg.de/")
        return
    
    print("OK SMPL-X models found")
    
    # Initialize master pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = MasterPipeline(
        smplx_path="models/smplx",
        device=device,
        gender='neutral'
    )
    
    # Find test video
    test_videos = [
        "test.mp4", "sample.mp4", "input.mp4", 
        "video.mp4", "demo.mp4", "example.mp4"
    ]
    
    test_video = None
    for video in test_videos:
        if Path(video).exists():
            test_video = video
            break
    
    if test_video:
        print(f"\nMOVIE Found test video: {test_video}")
        
        # Execute full pipeline
        results = pipeline.execute_full_pipeline(
            test_video,
            output_dir="final_production_output",
            max_frames=150,  # ~5 seconds at 30fps
            frame_skip=2,    # Process every 2nd frame
            quality='ultra'  # Ultra-high quality output
        )
        
        if results:
            print(f"\nTROPHY MISSION ACCOMPLISHED!")
            print(f"Check '{results['output_dir']}' for all results:")
            print(f"  MOVIE 3D Animation: {results['video_file'].name}")
            print(f"  CHART Mesh Data: {results['mesh_file'].name}")
            print(f"  GRAPH Statistics: Generated {results['stats']['meshes_generated']} meshes")
        else:
            print(f"\nBOOM MISSION FAILED!")
            
    else:
        print(f"\nFOLDER No test video found!")
        print(f"Available test files: {', '.join(test_videos)}")
        print(f"Place a video file in the current directory and run again.")


if __name__ == "__main__":
    main()
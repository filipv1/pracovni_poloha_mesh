#!/usr/bin/env python3
"""
Production 3D Human Mesh Pipeline - PARALLEL VERSION
=====================================================

Complete implementation with SMPL-X, Open3D, and MediaPipe, refactored for
high-performance parallel processing.

Architecture:
-------------
This script uses a two-phase approach for maximum efficiency:

1.  **Phase 1: Sequential Landmark Detection with MediaPipe Smoothing:**
    A fast, sequential pass over the video leverages MediaPipe's internal
    temporal smoothing (`smooth_landmarks=True`) to produce a clean and
    temporally consistent sequence of 3D landmarks. This phase is extremely
    fast and provides a high-quality input for the next stage.

2.  **Phase 2: Massively Parallel SMPL-X Fitting:**
    The computationally expensive SMPL-X fitting process is performed for all
    frames simultaneously in a single, large batch on the GPU. This is enabled
    by a refactored, stateless fitter that operates on the entire sequence
    of smoothed landmarks at once.

This design achieves significant speedups over a frame-by-frame approach
while maintaining high-quality, smooth, and consistent output.
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
matplotlib.use('Agg')  # RUNPOD SAFE: Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
import argparse

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
    OPEN3D_AVAILABLE = False  # RUNPOD SAFE: Force disable GUI
    print("Open3D: Available but DISABLED for RunPod safety")
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
    """
    High-precision converter from MediaPipe landmarks to SMPL-X format.
    (Unchanged from original implementation)
    """
    
    def __init__(self):
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
        
        self.smplx_joint_tree = {
            0: ('pelvis', None), 1: ('left_hip', 0), 2: ('right_hip', 0), 3: ('spine1', 0),
            4: ('left_knee', 1), 5: ('right_knee', 2), 6: ('spine2', 3),
            7: ('left_ankle', 4), 8: ('right_ankle', 5), 9: ('spine3', 6),
            10: ('left_foot', 7), 11: ('right_foot', 8), 12: ('neck', 9),
            13: ('left_collar', 12), 14: ('right_collar', 12), 15: ('head', 12),
            16: ('left_shoulder', 13), 17: ('right_shoulder', 14),
            18: ('left_elbow', 16), 19: ('right_elbow', 17),
            20: ('left_wrist', 18), 21: ('right_wrist', 19)
        }
        
    def convert_landmarks_to_smplx(self, mp_landmarks):
        if mp_landmarks is None:
            return None, None
            
        mp_points = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks.landmark])
        visibility = np.array([lm.visibility if hasattr(lm, 'visibility') else 1.0 for lm in mp_landmarks.landmark])
        
        num_joints = len(self.smplx_joint_tree)
        smplx_joints = np.zeros((num_joints, 3))
        joint_weights = np.zeros(num_joints)
        
        direct_mappings = {
            16: (11, 1.0), 17: (12, 1.0), 18: (13, 0.8), 19: (14, 0.8),
            20: (15, 0.7), 21: (16, 0.7), 1: (23, 0.9), 2: (24, 0.9),
            4: (25, 0.8), 5: (26, 0.8), 7: (27, 0.7), 8: (28, 0.7),
        }
        
        for joint_idx, (mp_idx, confidence) in direct_mappings.items():
            if mp_idx < len(mp_points):
                smplx_joints[joint_idx] = mp_points[mp_idx]
                joint_weights[joint_idx] = confidence * visibility[mp_idx]
        
        self._calculate_anatomical_joints(mp_points, visibility, smplx_joints, joint_weights)
        return smplx_joints, joint_weights
    
    def _calculate_anatomical_joints(self, mp_points, visibility, smplx_joints, joint_weights):
        if len(mp_points) > 24 and visibility[23] > 0.5 and visibility[24] > 0.5:
            left_hip, right_hip = mp_points[23], mp_points[24]
            smplx_joints[0] = (left_hip + right_hip) / 2
            joint_weights[0] = 0.95
        
        if len(mp_points) > 12 and joint_weights[0] > 0 and visibility[11] > 0.5 and visibility[12] > 0.5:
            shoulder_center = (mp_points[11] + mp_points[12]) / 2
            pelvis = smplx_joints[0]
            spine_vector = shoulder_center - pelvis
            spine_length = np.linalg.norm(spine_vector)
            spine_unit = spine_vector / spine_length if spine_length > 0 else np.array([0, 0, 1])
            
            smplx_joints[3] = pelvis + spine_unit * spine_length * 0.2
            smplx_joints[6] = pelvis + spine_unit * spine_length * 0.5
            smplx_joints[9] = pelvis + spine_unit * spine_length * 0.8
            joint_weights[3:10] = 0.7
        
        self._calculate_improved_head_neck(mp_points, visibility, smplx_joints, joint_weights)
        
        foot_offset = np.array([0, 0, -0.08])
        if joint_weights[7] > 0:
            smplx_joints[10] = smplx_joints[7] + foot_offset
            joint_weights[10] = joint_weights[7] * 0.8
        if joint_weights[8] > 0:
            smplx_joints[11] = smplx_joints[8] + foot_offset
            joint_weights[11] = joint_weights[8] * 0.8
            
        if joint_weights[12] > 0:
            neck = smplx_joints[12]
            if joint_weights[16] > 0:
                smplx_joints[13] = neck + (smplx_joints[16] - neck) * 0.4
                joint_weights[13] = 0.6
            if joint_weights[17] > 0:
                smplx_joints[14] = neck + (smplx_joints[17] - neck) * 0.4
                joint_weights[14] = 0.6

    def _calculate_improved_head_neck(self, mp_points, visibility, smplx_joints, joint_weights):
        if len(mp_points) < 13: return

        nose = mp_points[0]
        has_left_ear = len(mp_points) > 7 and visibility[7] > 0.3
        has_right_ear = len(mp_points) > 8 and visibility[8] > 0.3

        if not has_left_ear and not has_right_ear:
            smplx_joints[15] = nose; joint_weights[15] = 0.5
            if joint_weights[16] > 0 and joint_weights[17] > 0:
                shoulder_center = (smplx_joints[16] + smplx_joints[17]) / 2
                smplx_joints[12] = shoulder_center + (nose - shoulder_center) * 0.3
                joint_weights[12] = 0.7
            return

        if has_left_ear and has_right_ear:
            left_ear, right_ear = mp_points[7], mp_points[8]
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = (visibility[7] + visibility[8]) / 2
        elif has_left_ear:
            left_ear = mp_points[7]
            ear_to_nose = nose - left_ear
            right_ear = nose + np.array([-ear_to_nose[0], ear_to_nose[1], ear_to_nose[2]])
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = visibility[7] * 0.7
        else: # has_right_ear
            right_ear = mp_points[8]
            ear_to_nose = nose - right_ear
            left_ear = nose + np.array([-ear_to_nose[0], ear_to_nose[1], ear_to_nose[2]])
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = visibility[8] * 0.7

        forward_vector = nose - ear_center
        forward_dist = np.linalg.norm(forward_vector)
        forward_unit = forward_vector / forward_dist if forward_dist > 0 else np.array([0, 0, 1])

        ear_vector = right_ear - left_ear
        up_vector = np.cross(ear_vector, forward_vector)
        up_norm = np.linalg.norm(up_vector)
        up_vector = up_vector / up_norm if up_norm > 0 else np.array([0, 1, 0])

        skull_top = ear_center + up_vector * (forward_dist * 1.3)
        head_center = skull_top + forward_unit * (forward_dist * 0.2)
        
        smplx_joints[15] = head_center
        joint_weights[15] = min(visibility[0], ear_confidence) * 0.85
        
        if joint_weights[16] > 0 and joint_weights[17] > 0:
            shoulder_center = (smplx_joints[16] + smplx_joints[17]) / 2
            smplx_joints[12] = shoulder_center + (ear_center - shoulder_center) * 0.3
            joint_weights[12] = 0.85


class HighAccuracySMPLXFitter:
    """
    High-accuracy, STATELESS, BATCH-ENABLED SMPL-X mesh fitter.
    This version is refactored to process an entire batch of frames at once,
    removing all temporal, frame-by-frame logic.
    """
    
    def __init__(self, model_path, device='cpu', gender='neutral', batch_size=1):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.gender = gender
        
        if SMPLX_AVAILABLE and self._verify_model_files():
            try:
                self.smplx_model = smplx.SMPLX(
                    model_path=str(self.model_path), gender=gender, use_face_contour=False,
                    use_hands=False, num_betas=10, num_expression_coeffs=0,
                    create_global_orient=True, create_body_pose=True,
                    create_transl=True, batch_size=batch_size
                ).to(self.device)
                self.model_ready = True
                print(f"OK SMPL-X Model: Loaded successfully (gender={gender}, batch_size={batch_size})")
            except Exception as e:
                print(f"ERROR SMPL-X Model: Load failed - {e}")
                self.model_ready = False
        else:
            self.model_ready = False
            print("ERROR SMPL-X Model: Files not found")
        
    def _verify_model_files(self):
        return (self.model_path / f"SMPLX_{self.gender.upper()}.npz").exists()
    
    def fit_mesh_to_landmarks(self, target_joints_batch, joint_weights_batch=None, iterations=250):
        if not self.model_ready:
            return [self._create_wireframe_mesh(joints) for joints in target_joints_batch.cpu().numpy()]

        batch_size = target_joints_batch.shape[0]
        
        body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
        global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        target_tensor = target_joints_batch.to(dtype=torch.float32, device=self.device)
        
        if joint_weights_batch is not None:
            weights_tensor = joint_weights_batch.to(dtype=torch.float32, device=self.device)
        else:
            weights_tensor = torch.ones_like(target_tensor[..., 0], device=self.device)
        
        optimization_stages = [
            {'lr': 0.1, 'iterations': 80, 'params': [global_orient, transl, betas]},
            {'lr': 0.05, 'iterations': 100, 'params': [body_pose, global_orient]},
            {'lr': 0.01, 'iterations': 70, 'params': [body_pose, betas, transl]}
        ]
        
        print(f"  Fitting SMPL-X mesh for a batch of {batch_size} frames...")
        
        all_params = [body_pose, global_orient, transl, betas]
        
        for stage_idx, stage in enumerate(optimization_stages):
            optimizer = optim.Adam(stage['params'], lr=stage['lr'])
            
            for i in range(stage['iterations']):
                optimizer.zero_grad()
                
                smpl_output = self.smplx_model(body_pose=body_pose, global_orient=global_orient, transl=transl, betas=betas)
                
                pred_joints = smpl_output.joints[:, :target_tensor.shape[1]]
                joint_diff = pred_joints - target_tensor
                weighted_diff = joint_diff * weights_tensor.unsqueeze(-1)
                joint_loss = torch.mean(weighted_diff**2)
                
                pose_reg = torch.mean(body_pose**2) * 0.0001
                shape_reg = torch.mean(betas**2) * 0.00001
                
                total_loss = joint_loss + pose_reg + shape_reg
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                if i % 40 == 0:
                    print(f"    Stage {stage_idx+1}, Iter {i:3d}: Loss={total_loss.item():.6f}")

        print("  OK Batch fitting complete.")
        
        with torch.no_grad():
            final_params = {'body_pose': body_pose, 'global_orient': global_orient, 'transl': transl, 'betas': betas}
            final_output = self.smplx_model(**final_params)
            
            vertices_batch = final_output.vertices.cpu().numpy()
            faces = self.smplx_model.faces
            joints_batch = final_output.joints.cpu().numpy()
            
            vertices_batch[:, :, 1] *= -1
            joints_batch[:, :, 1] *= -1
            
            mesh_results_list = []
            for i in range(batch_size):
                mesh_result = {
                    'vertices': vertices_batch[i],
                    'faces': faces,
                    'joints': joints_batch[i],
                    'smplx_params': {k: v[i].cpu().numpy() for k, v in final_params.items()},
                    'vertex_count': len(vertices_batch[i]),
                    'face_count': len(faces)
                }
                mesh_results_list.append(mesh_result)
            
            return mesh_results_list

    def _create_wireframe_mesh(self, joints):
        connections = [(0,1),(0,2),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(12,16),(12,17),(16,18),(17,19),(18,20),(19,21)]
        return {'vertices': joints, 'faces': np.array([]), 'joints': joints, 'connections': connections, 'is_wireframe': True}


class ProfessionalVisualizer:
    """
    Professional-grade 3D visualization using Open3D and matplotlib.
    (Unchanged from original implementation)
    """
    def __init__(self, use_open3d=True, theme='dark'):
        self.use_open3d = use_open3d and OPEN3D_AVAILABLE
        self.theme = theme
        self.mesh_color = np.array([0.8, 0.9, 1.0])
        self.joint_color = np.array([1.0, 0.3, 0.3])
        self.skeleton_color = np.array([0.2, 1.0, 0.2])
        plt.style.use('dark_background' if theme == 'dark' else 'default')
        self.bg_color = np.array([0.1, 0.1, 0.1]) if theme == 'dark' else np.array([1.0, 1.0, 1.0])
        print(f"OK Visualizer: {'Open3D' if self.use_open3d else 'Matplotlib'} renderer")

    def render_single_mesh(self, mesh_data, title="3D Human Mesh", save_path=None, show_joints=True):
        if mesh_data is None: return
        if self.use_open3d and not mesh_data.get('is_wireframe', False):
            return self._render_with_open3d(mesh_data, title, save_path, show_joints)
        else:
            return self._render_with_matplotlib(mesh_data, title, save_path, show_joints)

    def _render_with_open3d(self, mesh_data, title, save_path, show_joints):
        pass

    def _render_with_matplotlib(self, mesh_data, title, save_path, show_joints):
        fig = plt.figure(figsize=(16, 12), facecolor=self.bg_color)
        ax = fig.add_subplot(111, projection='3d')
        vertices = mesh_data['vertices']
        faces = mesh_data.get('faces', [])
        connections = mesh_data.get('connections', [])
        joints = mesh_data.get('joints', vertices)
        
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=self.mesh_color, s=3, alpha=0.9, depthshade=True)
        
        if len(connections) > 0:
            for start_idx, end_idx in connections:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    start, end = vertices[start_idx], vertices[end_idx]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=self.skeleton_color, linewidth=4, alpha=0.8)
        
        if show_joints and len(joints) > 0:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=[self.joint_color], s=100, alpha=0.9, depthshade=True)
        
        ax.set_title(title, fontsize=20, color='white', pad=30)
        all_points = np.vstack([vertices, joints]) if len(joints) > 0 else vertices
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.ptp(all_points, axis=0)) * 0.6
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.view_init(elev=15, azim=45)
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.bg_color)
            print(f"OK Matplotlib render saved: {save_path}")
            plt.close(fig)
            return save_path
        return fig, ax

    def create_professional_video(self, mesh_sequence, output_path, fps=30, quality='high'):
        print(f"Creating video ({len(mesh_sequence)} frames)...")
        if not mesh_sequence: return
        
        first_valid_mesh = next((mesh for mesh in mesh_sequence if mesh is not None), None)
        if not first_valid_mesh:
            print("ERROR: No valid meshes in sequence to create video.")
            return

        fig, ax = self._render_with_matplotlib(first_valid_mesh, "Frame 0", None, True)

        def animate_frame(frame_idx):
            ax.clear()
            mesh_data = mesh_sequence[frame_idx]
            if mesh_data:
                self._render_with_matplotlib(mesh_data, f"Frame {frame_idx+1:04d}", None, True)
            else:
                ax.text2D(0.5, 0.5, 'No Pose Detected', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=20)
                ax.set_title(f"Frame {frame_idx+1:04d}", fontsize=20, color='white', pad=30)

        anim = FuncAnimation(fig, animate_frame, frames=len(mesh_sequence), interval=1000//fps, blit=False)
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print(f"OK Professional video saved: {output_path}")
        except Exception as e:
            print(f"ERROR FFmpeg failed: {e}. Saving as GIF.")
            try:
                anim.save(output_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
                print(f"OK GIF saved: {output_path.replace('.mp4', '.gif')}")
            except Exception as e2:
                print(f"ERROR Pillow fallback failed: {e2}")
        plt.close(fig)


class MasterPipeline:
    """
    Master pipeline orchestrating the new two-phase parallel workflow.
    """
    
    def __init__(self, smplx_path="models/smplx", device='auto', gender='neutral'):
        print("Initializing Master 3D Human Mesh Pipeline (PARALLEL)")
        print("=" * 70)
        
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.smplx_path = Path(smplx_path)
        self.gender = gender
        
        self.mp_pose = mp.solutions.pose
        self.visualizer = ProfessionalVisualizer(use_open3d=False, theme='dark')
        
        self.stats = {'frames_processed': 0, 'meshes_generated': 0, 'processing_time': 0.0}
        
        print(f"Device: {self.device}")
        print(f"SMPL-X Path: {self.smplx_path}")
        print(f"Gender: {self.gender}")
        print("Master Pipeline Ready.")

    def _phase1_detect_and_smooth_landmarks(self, video_path, max_frames=None, frame_skip=1):
        """
        Phase 1: Sequentially process video to get a smooth landmark sequence
        using MediaPipe's internal temporal filtering.
        """
        print()
        print("PHASE 1: Detecting and Smoothing Landmarks with MediaPipe...")
        print("-" * 70)

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            smooth_landmarks=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames: total_frames = min(total_frames, max_frames)

        all_landmarks = []
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            all_landmarks.append(results.pose_world_landmarks)
            
            progress = (frame_idx + 1) / total_frames * 100
            print(f"  Frame {frame_idx+1:4d}/{total_frames} ({progress:5.1f}%) - Pose: {'OK' if results.pose_world_landmarks else 'FAIL'}")
            frame_idx += 1
        
        cap.release()
        pose.close()
        print(f"Phase 1 complete. Extracted landmarks for {len(all_landmarks)} frames.")
        return all_landmarks

    def _phase2_fit_parallel_mesh(self, all_landmarks):
        """
        Phase 2: Take the smooth landmark sequence and fit the SMPL-X model
        to all valid frames in a single, massive parallel batch.
        """
        print()
        print("PHASE 2: Fitting SMPL-X Model in Parallel Batch...")
        print("-" * 70)

        converter = PreciseMediaPipeConverter()
        
        valid_landmarks_with_indices = [(i, lm) for i, lm in enumerate(all_landmarks) if lm is not None]
        if not valid_landmarks_with_indices:
            print("  ERROR: No valid landmarks found to fit.")
            return [None] * len(all_landmarks)
            
        original_indices, valid_landmarks = zip(*valid_landmarks_with_indices)
        
        print(f"  Preparing batch of {len(valid_landmarks)} valid frames for fitting...")
        joints_list = []
        weights_list = []
        for lm in valid_landmarks:
            joints, weights = converter.convert_landmarks_to_smplx(lm)
            if joints is not None:
                joints_list.append(joints)
                weights_list.append(weights)

        if not joints_list:
            print("  ERROR: Landmark conversion failed for all valid frames.")
            return [None] * len(all_landmarks)

        target_joints_batch = torch.from_numpy(np.stack(joints_list)).to(self.device)
        joint_weights_batch = torch.from_numpy(np.stack(weights_list)).to(self.device)
        
        batch_size = len(joints_list)
        fitter = HighAccuracySMPLXFitter(self.smplx_path, self.device, self.gender, batch_size=batch_size)
        
        list_of_mesh_results = fitter.fit_mesh_to_landmarks(target_joints_batch, joint_weights_batch)
        
        mesh_sequence = [None] * len(all_landmarks)
        for i, result in zip(original_indices, list_of_mesh_results):
            mesh_sequence[i] = result
        
        print(f"Phase 2 complete. Generated {len(list_of_mesh_results)} meshes.")
        return mesh_sequence

    def execute_parallel_pipeline(self, video_path, output_dir="production_output_p", 
                                  max_frames=None, frame_skip=1, quality='high'):
        start_time = time.time()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}")
            return None
        
        all_landmarks = self._phase1_detect_and_smooth_landmarks(video_path, max_frames, frame_skip)
        
        mesh_sequence = self._phase2_fit_parallel_mesh(all_landmarks)
        
        successful_frames = sum(1 for m in mesh_sequence if m is not None)
        
        processing_time = time.time() - start_time
        self.stats['frames_processed'] = len(all_landmarks)
        self.stats['meshes_generated'] = successful_frames
        self.stats['processing_time'] = processing_time
        
        print()
        print("PROCESSING COMPLETE!")
        print("-" * 70)
        print(f"   Total Frames Processed: {len(all_landmarks)}")
        print(f"   Successful Meshes: {successful_frames}")
        if len(all_landmarks) > 0:
            print(f"   Success Rate: {successful_frames/len(all_landmarks)*100:.1f}%")
        print(f"   Total Processing Time: {processing_time:.1f} seconds")
        if processing_time > 0:
            print(f"   Overall FPS: {len(all_landmarks)/processing_time:.2f}")
        
        if successful_frames > 0:
            print()
            print("GENERATING OUTPUTS...")
            print("-" * 70)
            
            cap = cv2.VideoCapture(str(video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            pkl_data = {
                'mesh_sequence': mesh_sequence,
                'metadata': {'fps': fps, 'frame_skip': frame_skip, 'video_filename': video_path.name}
            }
            with open(mesh_file, 'wb') as f:
                pickle.dump(pkl_data, f)
            print(f"Mesh data saved: {mesh_file}")
            
            stats_file = output_dir / f"{video_path.stem}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"Statistics saved: {stats_file}")

            # Video generation removed to save time (30+ seconds)
            print()
            print(f"SUCCESS! All outputs generated in: {output_dir}")
            return {'mesh_sequence': mesh_sequence, 'mesh_file': mesh_file, 'stats': self.stats, 'output_dir': output_dir}
        else:
            print()
            print("FAILED: No meshes were generated")
            return None


def main():
    """Main execution function for the parallel pipeline."""
    parser = argparse.ArgumentParser(description="Run the Parallel 3D Human Mesh Pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, nargs='?', default="production_output_p", help="Directory to save outputs.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every N-th frame.")
    parser.add_argument("--quality", type=str, default="high", choices=['medium', 'high', 'ultra'], help="Quality of the output video.")
    parser.add_argument("--gender", type=str, default="neutral", choices=['neutral', 'male', 'female'], help="Gender for the SMPL-X model.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device ('cuda', 'cpu', or 'auto').")
    args = parser.parse_args()

    print("PRODUCTION 3D HUMAN MESH PIPELINE (PARALLEL)")
    print("=" * 80)
    
    models_dir = Path("models/smplx")
    if not models_dir.exists() or not any(models_dir.glob("*.npz")):
        print("ERROR: SMPL-X models not found! Please download from https://smpl-x.is.tue.mpg.de/")
        return
    
    video_file = Path(args.video_path)
    if not video_file.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        return

    pipeline = MasterPipeline(smplx_path=models_dir, device=args.device, gender=args.gender)
    
    pipeline.execute_parallel_pipeline(
        args.video_path, output_dir=args.output_dir, max_frames=args.max_frames,
        frame_skip=args.frame_skip, quality=args.quality
    )

if __name__ == "__main__":
    main()

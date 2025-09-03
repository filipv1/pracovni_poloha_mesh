#!/usr/bin/env python3
"""
Production 3D Human Mesh Pipeline
Complete implementation with SMPL-X, Open3D, and MediaPipe
Designed for maximum accuracy and professional visualization
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
import matplotlib
matplotlib.use('Agg')  # RUNPOD SAFE: Headless backend
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
        
        # Extract visibility scores
        visibility = np.array([lm.visibility if hasattr(lm, 'visibility') else 1.0 
                              for lm in mp_landmarks.landmark])
        
        # Initialize SMPL-X joints and confidence weights
        num_joints = len(self.smplx_joint_tree)
        smplx_joints = np.zeros((num_joints, 3))
        joint_weights = np.zeros(num_joints)
        
        # Direct landmark mappings with confidence (excluding head - will be improved)
        direct_mappings = {
            # 15: (0, 0.6),    # head from nose - REMOVED, using improved estimation
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
        self._calculate_anatomical_joints(mp_points, visibility, smplx_joints, joint_weights)
        
        return smplx_joints, joint_weights
    
    def _calculate_anatomical_joints(self, mp_points, visibility, smplx_joints, joint_weights):
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
            # Note: neck will be improved below
            
            joint_weights[3:10] = 0.7  # spine chain confidence (not neck)
        
        # IMPROVED HEAD AND NECK ESTIMATION
        self._calculate_improved_head_neck(mp_points, visibility, smplx_joints, joint_weights)
        
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
    
    def _calculate_improved_head_neck(self, mp_points, visibility, smplx_joints, joint_weights):
        """Calculate improved head and neck positions using ear landmarks"""
        
        # Check if we have necessary landmarks
        if len(mp_points) < 13:
            # Fallback to nose-based estimation
            if len(mp_points) > 0:
                smplx_joints[15] = mp_points[0]  # head from nose
                joint_weights[15] = 0.6
                # Simple neck estimation
                if joint_weights[16] > 0 and joint_weights[17] > 0:
                    shoulder_center = (smplx_joints[16] + smplx_joints[17]) / 2
                    smplx_joints[12] = shoulder_center + (mp_points[0] - shoulder_center) * 0.3
                    joint_weights[12] = 0.7
            return
        
        # Extract key landmarks
        nose = mp_points[0]
        
        # Check ear availability (indices 7 and 8)
        has_left_ear = len(mp_points) > 7 and visibility[7] > 0.3
        has_right_ear = len(mp_points) > 8 and visibility[8] > 0.3
        
        if not has_left_ear and not has_right_ear:
            # No ears visible - fallback to nose with lower confidence
            smplx_joints[15] = nose
            joint_weights[15] = 0.5
            # Simple neck
            if joint_weights[16] > 0 and joint_weights[17] > 0:
                shoulder_center = (smplx_joints[16] + smplx_joints[17]) / 2
                smplx_joints[12] = shoulder_center + (nose - shoulder_center) * 0.3
                joint_weights[12] = 0.7
            return
        
        # Get ear positions
        if has_left_ear and has_right_ear:
            # Both ears visible - best case
            left_ear = mp_points[7]
            right_ear = mp_points[8]
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = (visibility[7] + visibility[8]) / 2
        elif has_left_ear:
            # Only left ear visible - mirror it
            left_ear = mp_points[7]
            ear_to_nose = nose - left_ear
            right_ear = nose + np.array([-ear_to_nose[0], ear_to_nose[1], ear_to_nose[2]])
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = visibility[7] * 0.7
        else:
            # Only right ear visible - mirror it
            right_ear = mp_points[8]
            ear_to_nose = nose - right_ear
            left_ear = nose + np.array([-ear_to_nose[0], ear_to_nose[1], ear_to_nose[2]])
            ear_center = (left_ear + right_ear) / 2
            ear_confidence = visibility[8] * 0.7
        
        # Calculate head orientation
        forward_vector = nose - ear_center
        forward_dist = np.linalg.norm(forward_vector)
        
        if forward_dist > 0:
            forward_unit = forward_vector / forward_dist
        else:
            forward_unit = np.array([0, 0, 1])
        
        # Calculate up vector
        ear_vector = right_ear - left_ear
        ear_dist = np.linalg.norm(ear_vector)
        
        if ear_dist > 0:
            up_vector = np.cross(ear_vector, forward_vector)
            up_norm = np.linalg.norm(up_vector)
            if up_norm > 0:
                up_vector = up_vector / up_norm
            else:
                up_vector = np.array([0, 1, 0])
        else:
            up_vector = np.array([0, 1, 0])
        
        # Anthropometric ratios
        skull_height_ratio = 1.3  # skull height = 1.3 * nose-ear distance
        
        # Estimate skull top
        skull_top = ear_center + up_vector * (forward_dist * skull_height_ratio)
        
        # SMPL-X head joint (center of mass)
        head_center = skull_top + forward_unit * (forward_dist * 0.2)
        
        # Set improved head position
        smplx_joints[15] = head_center
        joint_weights[15] = min(visibility[0], ear_confidence) * 0.85
        
        # Improve neck position
        if joint_weights[16] > 0 and joint_weights[17] > 0:
            shoulder_center = (smplx_joints[16] + smplx_joints[17]) / 2
            improved_neck = shoulder_center + (ear_center - shoulder_center) * 0.3
            smplx_joints[12] = improved_neck
            joint_weights[12] = 0.85


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
        
        # Simple processing - emulate arm_meshes.pkl quality with batch speed
        self.param_history = []
        self.max_history = 5
        self.temporal_alpha = 0.3  # Keep for individual processing compatibility
        
        # DISABLE complex predictions - keep it simple like arm_meshes.pkl
        self.limb_predictor = None
        print("INFO: Using simple processing for natural movement (arm_meshes.pkl style)")
        
        # Batch processing support
        self.supports_batch = True
        self.batch_models_cache = {}  # Cache for batch SMPL-X models
        
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
            
            # Fix orientation - SMPL-X Y-axis points down by default
            # Flip Y-axis to make mesh stand upright
            vertices[:, 1] *= -1
            joints[:, 1] *= -1
            
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
    
    def fit_mesh_to_landmarks_batch(self, landmarks_batch, weights_batch=None, iterations=250):
        """Batch fitting of SMPL-X meshes for multiple frames simultaneously
        
        Args:
            landmarks_batch: List of landmark arrays [(33, 3), ...]
            weights_batch: List of weight arrays (optional)
            iterations: Number of optimization iterations
            
        Returns:
            List of mesh_data dictionaries (same format as single fitting)
        """
        if not self.model_ready or len(landmarks_batch) == 0:
            # Fallback to individual processing
            return [self.fit_mesh_to_landmarks(landmarks, weights) 
                    for landmarks, weights in zip(landmarks_batch, weights_batch or [None] * len(landmarks_batch))]
        
        batch_size = len(landmarks_batch)
        
        # Temporal smoothing quality warning/optimization
        if batch_size > self.max_recommended_batch_size:
            print(f"  WARNING: Large batch size ({batch_size}) may reduce temporal smoothing quality")
            print(f"  Recommended: Use --batch-size {self.max_recommended_batch_size} or smaller for optimal results")
        elif batch_size > 64:
            print(f"  INFO: Large batch size ({batch_size}) - using advanced hierarchical temporal smoothing")
        
        print(f"  Batch fitting {batch_size} frames with hierarchical temporal smoothing...")
        
        # Get or create cached batch SMPL-X model for this batch size
        if batch_size not in self.batch_models_cache:
            try:
                self.batch_models_cache[batch_size] = smplx.SMPLX(
                    model_path=str(self.model_path),
                    gender=self.gender,
                    use_face_contour=False,
                    use_hands=False,
                    num_betas=10,
                    num_expression_coeffs=0,
                    create_global_orient=True,
                    create_body_pose=True,
                    create_transl=True,
                    batch_size=batch_size  # Use actual batch_size
                ).to(self.device)
                print(f"  Created batch SMPL-X model for batch_size={batch_size}")
            except Exception as e:
                print(f"  WARNING: Batch model creation failed ({e}), falling back to individual processing")
                return [self.fit_mesh_to_landmarks(landmarks, weights) 
                        for landmarks, weights in zip(landmarks_batch, weights_batch or [None] * len(landmarks_batch))]
        
        batch_smplx_model = self.batch_models_cache[batch_size]
        
        # Prepare batch tensors
        batch_landmarks = []
        batch_weights = []
        
        for i, landmarks in enumerate(landmarks_batch):
            if landmarks is not None and len(landmarks) > 0:
                batch_landmarks.append(torch.tensor(landmarks, dtype=torch.float32, device=self.device))
                
                if weights_batch and i < len(weights_batch) and weights_batch[i] is not None:
                    batch_weights.append(torch.tensor(weights_batch[i], dtype=torch.float32, device=self.device))
                else:
                    batch_weights.append(torch.ones(len(landmarks), device=self.device))
            else:
                # Handle invalid landmarks
                batch_landmarks.append(torch.zeros((33, 3), device=self.device))
                batch_weights.append(torch.ones(33, device=self.device) * 0.1)
        
        if not batch_landmarks:
            return []
        
        # Stack into batch tensors
        target_batch = torch.stack(batch_landmarks)  # (batch_size, 33, 3)
        weights_batch_tensor = torch.stack(batch_weights)  # (batch_size, 33)
        
        # Initialize parameters for batch
        body_pose = torch.zeros((batch_size, 63), device=self.device, requires_grad=True)
        global_orient = torch.zeros((batch_size, 3), device=self.device, requires_grad=True) 
        transl = torch.zeros((batch_size, 3), device=self.device, requires_grad=True)
        betas = torch.zeros((batch_size, 10), device=self.device, requires_grad=True)
        
        # Smart initialization from history
        if len(self.param_history) > 0:
            last_params = self.param_history[-1]
            # Repeat last parameters for all batch items
            body_pose.data = last_params['body_pose'].repeat(batch_size, 1)
            global_orient.data = last_params['global_orient'].repeat(batch_size, 1)
            transl.data = last_params['transl'].repeat(batch_size, 1) 
            betas.data = last_params['betas'].repeat(batch_size, 1)
        
        # Multi-stage batch optimization
        optimization_stages = [
            {'lr': 0.1, 'iterations': 80, 'focus': 'global'},
            {'lr': 0.05, 'iterations': 100, 'focus': 'pose'},  
            {'lr': 0.01, 'iterations': 70, 'focus': 'refinement'}
        ]
        
        best_loss = float('inf')
        best_params = None
        
        for stage_idx, stage in enumerate(optimization_stages):
            # Stage-specific optimizer  
            if stage['focus'] == 'global':
                optimizer = optim.Adam([global_orient, transl], lr=stage['lr'])
            elif stage['focus'] == 'pose':
                optimizer = optim.Adam([body_pose, betas], lr=stage['lr'])
            else:  # refinement
                optimizer = optim.AdamW([body_pose, global_orient, transl, betas], 
                                      lr=stage['lr'], weight_decay=1e-4)
            
            for i in range(stage['iterations']):
                optimizer.zero_grad()
                
                # Forward pass - batch SMPL-X with correct batch model
                smpl_output = batch_smplx_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=transl,
                    betas=betas
                )
                
                # Use only the first N joints to match target_batch size
                num_target_joints = target_batch.shape[1]  # Should be 22 for SMPL-X converted landmarks
                predicted_joints = smpl_output.joints[:, :num_target_joints, :]  # (batch_size, N, 3)
                
                # Batch loss computation
                joint_diff = (predicted_joints - target_batch) * weights_batch_tensor.unsqueeze(-1)
                joint_loss = torch.mean(joint_diff ** 2)
                
                # Regularization terms (reduced to match individual processing)
                pose_reg = torch.mean(body_pose ** 2) * 0.0001  # Match individual processing
                shape_reg = torch.mean(betas ** 2) * 0.00001    # Match individual processing
                
                # LIGHT TEMPORAL SMOOTHING - Exact arm_meshes.pkl style
                # arm_meshes.pkl used individual processing WITH temporal smoothing
                temporal_loss = 0.0
                
                # Restore arm_meshes.pkl temporal smoothing (individual processing style)
                if len(self.param_history) > 0:
                    prev_params = self.param_history[-1]
                    
                    # Apply temporal smoothing only to first frame of batch (inter-batch consistency)
                    # This emulates individual processing where each frame is smoothed with previous
                    temporal_loss = (
                        torch.mean((body_pose[0:1] - prev_params['body_pose']) ** 2) * self.temporal_alpha +
                        torch.mean((betas[0:1] - prev_params['betas']) ** 2) * self.temporal_alpha * 0.1
                    )
                    
                    # For other frames in batch, no temporal smoothing (independent processing)
                    # This matches arm_meshes.pkl where each frame was processed individually
                
                total_loss = joint_loss + pose_reg + shape_reg + temporal_loss
                
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_params = {
                        'global_orient': global_orient.clone(),
                        'body_pose': body_pose.clone(), 
                        'transl': transl.clone(),
                        'betas': betas.clone()
                    }
        
        # Generate individual mesh results  
        mesh_results = []
        with torch.no_grad():
            final_output = batch_smplx_model(**best_params)
            
            for i in range(batch_size):
                vertices = final_output.vertices[i].cpu().numpy()
                faces = batch_smplx_model.faces
                joints = final_output.joints[i].cpu().numpy()
                
                # Fix orientation (same as single version)
                vertices[:, 1] *= -1
                joints[:, 1] *= -1
                
                mesh_result = {
                    'vertices': vertices,
                    'faces': faces, 
                    'joints': joints,
                    'smplx_params': {k: v[i].cpu().numpy() for k, v in best_params.items()},
                    'fitting_error': best_loss / batch_size,  # Approximate individual error
                    'vertex_count': len(vertices),
                    'face_count': len(faces)
                }
                
                mesh_results.append(mesh_result)
        
        # Update temporal history with last frame
        if batch_size > 0:
            last_frame_params = {k: v[-1:].clone() for k, v in best_params.items()}
            self.param_history.append(last_frame_params)
            if len(self.param_history) > self.max_history:
                self.param_history.pop(0)
        
        # Keep batch model in cache for reuse (don't delete)
        
        print(f"  OK Batch fitted {batch_size} meshes, avg error={best_loss:.6f}")
        return mesh_results
    
    def clear_batch_cache(self):
        """Clear cached batch SMPL-X models to free GPU memory"""
        for model in self.batch_models_cache.values():
            del model
        self.batch_models_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Cleared batch model cache")


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
    
    # PNG rendering methods removed for performance - only video generation remains
    
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
            'bitrate': 8000 if quality == 'ultra' else 5000 if quality == 'high' else 2000
        }
        
        # Add extra_args only if ffmpeg is available
        try:
            import matplotlib.animation as animation
            if hasattr(animation, 'FFMpegWriter'):
                writer_kwargs['extra_args'] = ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']
        except:
            pass
        
        try:
            anim.save(output_path, writer='ffmpeg', **writer_kwargs)
            print(f"OK Professional video saved: {output_path}")
        except Exception as e:
            print(f"ERROR FFmpeg failed: {e}")
            
            # Fallback to pillow with basic settings only
            try:
                print("Trying Pillow fallback...")
                basic_kwargs = {'fps': writer_kwargs['fps']}  # Only FPS for Pillow
                anim.save(output_path, writer='pillow', **basic_kwargs)
                print(f"OK Pillow video saved: {output_path}")
            except Exception as e2:
                print(f"ERROR Pillow failed: {e2}")
                print("Video creation skipped - PNG frames available instead")
        
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
                            max_frames=None, frame_skip=1, quality='ultra', batch_size=16):
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
        print(f"   Batch Size: {batch_size}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
        
        # BATCH PROCESSING: Extract all landmarks first, then batch process SMPL-X
        print(f"\nGEAR  PHASE 1: EXTRACT LANDMARKS...")
        print("-" * 70)
        
        landmarks_batch = []
        weights_batch = []
        frame_indices = []
        converter = PreciseMediaPipeConverter()
        frame_idx = 0
        
        # Phase 1: Extract all MediaPipe landmarks
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            progress = (frame_idx + 1) / total_frames * 100
            if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                print(f"Extracting landmarks {frame_idx+1:4d}/{total_frames} ({progress:5.1f}%)")
            
            # MediaPipe pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Simple processing like arm_meshes.pkl - no complex predictions
                joints, weights = converter.convert_landmarks_to_smplx(results.pose_world_landmarks)
                
                if joints is not None:
                    landmarks_batch.append(joints)
                    weights_batch.append(weights)
                    frame_indices.append(frame_idx)
                else:
                    # Add None for failed conversion (will be handled in batch processing)
                    landmarks_batch.append(None)
                    weights_batch.append(None)
                    frame_indices.append(frame_idx)
            else:
                # Add None for failed detection
                landmarks_batch.append(None)
                weights_batch.append(None)
                frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\nGEAR  PHASE 2: BATCH SMPL-X FITTING...")
        print("-" * 70)
        
        # Phase 2: Batch process SMPL-X fitting
        mesh_sequence = []
        successful_frames = 0
        total_batches = len(landmarks_batch)
        
        if total_batches == 0:
            print("ERROR: No frames to process!")
            return None
        
        # Process in batches
        for batch_start in range(0, total_batches, batch_size):
            batch_end = min(batch_start + batch_size, total_batches)
            current_batch_size = batch_end - batch_start
            
            batch_landmarks = landmarks_batch[batch_start:batch_end]
            batch_weights = weights_batch[batch_start:batch_end]
            batch_frame_indices = frame_indices[batch_start:batch_end]
            
            # Filter out None values for batch processing
            valid_landmarks = []
            valid_weights = []
            valid_indices = []
            none_positions = []  # Track positions of None values
            
            for i, (landmarks, weights, frame_idx) in enumerate(zip(batch_landmarks, batch_weights, batch_frame_indices)):
                if landmarks is not None:
                    valid_landmarks.append(landmarks)
                    valid_weights.append(weights)
                    valid_indices.append(frame_idx)
                else:
                    none_positions.append(i)
            
            batch_progress = ((batch_start + current_batch_size) / total_batches) * 100
            print(f"Processing batch {batch_start//batch_size + 1}: frames {batch_start+1}-{batch_end} ({batch_progress:.1f}%)")
            
            # Batch fit SMPL-X meshes
            if valid_landmarks:
                batch_meshes = self.mesh_fitter.fit_mesh_to_landmarks_batch(valid_landmarks, valid_weights)
                
                # Insert results back in correct positions
                mesh_results = []
                valid_idx = 0
                
                for i in range(current_batch_size):
                    if i in none_positions:
                        mesh_results.append(None)  # Failed frame
                    else:
                        if valid_idx < len(batch_meshes):
                            mesh_data = batch_meshes[valid_idx]
                            if mesh_data and not mesh_data.get('is_wireframe', False):
                                mesh_results.append(mesh_data)
                                successful_frames += 1
                                
                                # Update statistics
                                if 'fitting_error' in mesh_data:
                                    self.stats['average_fitting_error'] += mesh_data['fitting_error']
                                    
                                # PNG generation removed for performance
                            else:
                                mesh_results.append(None)
                        else:
                            mesh_results.append(None)
                        valid_idx += 1
                
                # Add successful meshes to sequence
                for mesh_data in mesh_results:
                    if mesh_data is not None:
                        mesh_sequence.append(mesh_data)
        
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
            
            # Save mesh data
            mesh_file = output_dir / f"{video_path.stem}_meshes.pkl"
            with open(mesh_file, 'wb') as f:
                pickle.dump(mesh_sequence, f)
            print(f"OK Mesh data: {mesh_file}")
            
            # Save statistics
            stats_file = output_dir / f"{video_path.stem}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"OK Statistics: {stats_file}")
            
            # Final mesh PNG generation removed for performance
            
            # Create professional video
            output_video = output_dir / f"{video_path.stem}_3d_animation.mp4"
            # Calculate effective FPS (ensure at least 1 FPS)
            effective_fps = max(1, fps // frame_skip) if frame_skip > 1 else fps
            self.visualizer.create_professional_video(mesh_sequence, str(output_video), 
                                                     fps=effective_fps, quality=quality)
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
    import sys
    
    # Parse command line arguments
    video_path = None
    output_dir = "production_output"
    max_frames = None
    frame_skip = 1
    quality = 'ultra'
    batch_size = 16  # Default batch size for GPU processing
    
    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--max-frames':
            if i + 1 < len(sys.argv):
                max_frames = int(sys.argv[i + 1])
                i += 2
            else:
                print("ERROR: --max-frames requires a value")
                return
        elif arg == '--frame-skip':
            if i + 1 < len(sys.argv):
                frame_skip = int(sys.argv[i + 1])
                i += 2
            else:
                print("ERROR: --frame-skip requires a value")
                return
        elif arg == '--quality':
            if i + 1 < len(sys.argv):
                quality = sys.argv[i + 1]
                i += 2
            else:
                print("ERROR: --quality requires a value")
                return
        elif arg == '--batch-size':
            if i + 1 < len(sys.argv):
                batch_size = int(sys.argv[i + 1])
                i += 2
            else:
                print("ERROR: --batch-size requires a value")
                return
        elif not arg.startswith('--'):
            if video_path is None:
                video_path = arg
            elif output_dir == "production_output":  # Default not yet changed
                output_dir = arg
            i += 1
        else:
            print(f"WARNING: Unknown argument: {arg}")
            i += 1
    
    print("ROCKET PRODUCTION 3D HUMAN MESH PIPELINE")
    print("=" * 80)
    
    if not video_path:
        print("ERROR: No video file specified")
        print("Usage: python run_production_simple.py <video_file> [output_dir] [--max-frames N] [--frame-skip N] [--quality ultra/high/medium] [--batch-size N]")
        return
    
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
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    print(f"\nMOVIE Processing video: {video_path}")
    if max_frames:
        print(f"LIMIT Max frames: {max_frames}")
    print(f"OUTPUT Output directory: {output_dir}")
    
    # Execute full pipeline with parsed arguments
    results = pipeline.execute_full_pipeline(
        video_path,
        output_dir=output_dir,
        max_frames=max_frames,
        frame_skip=frame_skip,
        quality=quality,
        batch_size=batch_size
    )
    
    if results:
        print(f"\nTROPHY MISSION ACCOMPLISHED!")
        print(f"Check '{results['output_dir']}' for all results:")
        if 'video_file' in results:
            print(f"  MOVIE 3D Animation: {results['video_file'].name}")
        if 'mesh_file' in results:
            print(f"  CHART Mesh Data: {results['mesh_file'].name}")
        if 'stats' in results:
            print(f"  GRAPH Statistics: Generated {results['stats']['meshes_generated']} meshes")
    else:
        print(f"\nBOOM MISSION FAILED!")
        print(f"Check console output for errors.")


if __name__ == "__main__":
    main()
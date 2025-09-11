#!/usr/bin/env python3
"""
Parallel Production 3D Human Mesh Pipeline - NO TEMPORAL SMOOTHING
Modified from run_production_simple.py to remove temporal dependencies
Designed for maximum parallelization, post-processing smoothing will be applied later
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
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

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


class ParallelSMPLXFitter:
    """MODIFIED: SMPL-X fitter WITHOUT temporal smoothing for parallel processing"""
    
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
                print(f"OK PARALLEL SMPL-X Model: Loaded successfully ({gender})")
            except Exception as e:
                print(f"ERROR SMPL-X Model: Load failed - {e}")
                self.model_ready = False
        else:
            self.model_ready = False
            print("ERROR SMPL-X Model: Files not found")
        
        self.converter = PreciseMediaPipeConverter()
        
        # NO TEMPORAL SMOOTHING - these are disabled for parallel processing
        # self.param_history = []  # REMOVED
        # self.max_history = 5     # REMOVED  
        # self.temporal_alpha = 0.3 # REMOVED
        
    def _verify_model_files(self):
        """Verify all required SMPL-X model files exist"""
        required_files = [f"SMPLX_{self.gender.upper()}.npz"]
        
        for file in required_files:
            if not (self.model_path / file).exists():
                return False
        return True
    
    def fit_mesh_to_landmarks(self, target_joints, joint_weights=None, iterations=250, frame_id=None):
        """Fit SMPL-X mesh using advanced optimization WITHOUT temporal smoothing"""
        
        if not self.model_ready:
            return self._create_wireframe_mesh(target_joints)
        
        batch_size = 1
        
        # ALWAYS COLD START - no temporal initialization for parallel processing
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
        
        # Multi-stage optimization (same as original)
        optimization_stages = [
            {'lr': 0.1, 'iterations': 80, 'focus': 'global'},      # Global pose and translation
            {'lr': 0.05, 'iterations': 100, 'focus': 'pose'},      # Fine-tune body pose
            {'lr': 0.01, 'iterations': 70, 'focus': 'refinement'}  # Final refinement
        ]
        
        best_loss = float('inf')
        best_params = None
        
        frame_info = f"Frame {frame_id}" if frame_id is not None else "Single frame"
        print(f"  Fitting SMPL-X mesh {frame_info} ({iterations} total iterations, NO TEMPORAL SMOOTHING)...")
        
        for stage_idx, stage in enumerate(optimization_stages):
            # Stage-specific optimizer (same as original)
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
                
                # Joint loss with confidence weighting (same as original)
                pred_joints = smpl_output.joints[:, :len(target_joints)]
                joint_diff = pred_joints - target_tensor
                weighted_diff = joint_diff * weights_tensor.view(1, -1, 1)
                joint_loss = torch.mean(torch.sum(weighted_diff**2, dim=-1))
                
                # Regularization terms (same as original)
                pose_reg = torch.mean(body_pose**2) * 0.0001
                shape_reg = torch.mean(betas**2) * 0.00001
                
                # NO TEMPORAL CONSISTENCY LOSS - this is the key change!
                # temporal_loss = 0.0 (always zero)
                
                # Total loss WITHOUT temporal component
                total_loss = joint_loss + pose_reg + shape_reg
                
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
                
                # Progress reporting (reduced frequency for parallel processing)
                if i % 50 == 0:
                    print(f"    Stage {stage_idx+1}, Iter {i:2d}: Loss={total_loss.item():.6f}")
        
        # NO TEMPORAL HISTORY UPDATE - this is removed for parallel processing
        
        # Generate final mesh (same as original)
        with torch.no_grad():
            final_output = self.smplx_model(**best_params)
            
            vertices = final_output.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces
            joints = final_output.joints[0].cpu().numpy()
            
            # Fix orientation - SMPL-X Y-axis points down by default
            vertices[:, 1] *= -1
            joints[:, 1] *= -1
            
            mesh_result = {
                'vertices': vertices,
                'faces': faces,
                'joints': joints,
                'smplx_params': {k: v.cpu().numpy() for k, v in best_params.items()},
                'fitting_error': best_loss,
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'frame_id': frame_id  # Add frame ID for parallel processing
            }
            
            print(f"  OK Mesh fitted {frame_info}: {len(vertices)} vertices, {len(faces)} faces, error={best_loss:.6f}")
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


def process_single_frame(args):
    """Process single frame for parallel processing"""
    frame_idx, frame_data, smplx_path, device, gender = args
    
    try:
        # Initialize fitter for this process
        fitter = ParallelSMPLXFitter(smplx_path, device, gender)
        converter = PreciseMediaPipeConverter()
        
        # Extract frame and landmarks
        frame, mp_landmarks = frame_data
        
        if mp_landmarks:
            # Convert landmarks to SMPL-X format
            joints, weights = converter.convert_landmarks_to_smplx(mp_landmarks)
            
            if joints is not None:
                # Fit SMPL-X mesh WITHOUT temporal smoothing
                mesh_data = fitter.fit_mesh_to_landmarks(joints, weights, frame_id=frame_idx)
                
                if mesh_data and not mesh_data.get('is_wireframe', False):
                    return frame_idx, mesh_data
        
        return frame_idx, None
        
    except Exception as e:
        print(f"ERROR processing frame {frame_idx}: {e}")
        return frame_idx, None


class ParallelMasterPipeline:
    """Master pipeline for PARALLEL processing WITHOUT temporal smoothing"""
    
    def __init__(self, smplx_path="models/smplx", device='auto', gender='neutral', max_workers=None):
        
        print("🚀 Initializing PARALLEL Master 3D Human Mesh Pipeline (NO TEMPORAL SMOOTHING)")
        print("=" * 80)
        
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.smplx_path = Path(smplx_path)
        self.gender = gender
        
        # Parallel processing configuration with RunPod-safe limits
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            # Conservative worker limits for stability
            if cpu_count >= 24:  # High-end systems like RunPod RTX 4090
                self.max_workers = min(8, cpu_count // 4)  # Cap at 8 workers for stability
            elif cpu_count >= 12:  # Mid-range systems
                self.max_workers = min(6, cpu_count // 2)
            else:  # Low-end systems
                self.max_workers = max(1, cpu_count - 1)
        else:
            self.max_workers = max_workers
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
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
        print(f"OK Max Workers: {self.max_workers}")
        print("🚀 PARALLEL Master Pipeline Ready!")
    
    def extract_all_landmarks(self, video_path):
        """Extract MediaPipe landmarks from all frames"""
        
        print(f"\n🔍 EXTRACTING LANDMARKS FROM ALL FRAMES: {video_path}")
        print("-" * 70)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video")
            return None
        
        landmarks_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Store frame and landmarks
            landmarks_data.append((frame, results.pose_world_landmarks))
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"  Extracted landmarks from frame {frame_idx}")
        
        cap.release()
        print(f"✅ Extracted landmarks from {len(landmarks_data)} frames")
        return landmarks_data
    
    def execute_parallel_pipeline(self, video_path, output_dir="parallel_output", 
                                 max_frames=None, frame_skip=1, timeout_per_frame=300):
        """Execute PARALLEL pipeline WITHOUT temporal smoothing"""
        
        start_time = time.time()
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return None
        
        print(f"\n🎬 PARALLEL PROCESSING: {video_path}")
        print("=" * 70)
        
        # Extract all landmarks first
        landmarks_data = self.extract_all_landmarks(video_path)
        if not landmarks_data:
            return None
        
        # Apply frame limits and skipping
        if max_frames:
            landmarks_data = landmarks_data[:max_frames]
        
        # Apply frame skipping
        if frame_skip > 1:
            landmarks_data = landmarks_data[::frame_skip]
        
        print(f"\n⚡ PARALLEL MESH FITTING ({self.max_workers} workers)")
        print("-" * 70)
        print(f"   Total Frames: {len(landmarks_data)}")
        print(f"   Frame Skip: {frame_skip}")
        print(f"   Workers: {self.max_workers}")
        
        # Prepare arguments for parallel processing
        process_args = [
            (frame_idx, frame_data, str(self.smplx_path), self.device, self.gender)
            for frame_idx, frame_data in enumerate(landmarks_data)
            if frame_data[1] is not None  # Only process frames with landmarks
        ]
        
        print(f"   Frames with landmarks: {len(process_args)}")
        
        # Parallel processing of all frames
        mesh_sequence = []
        successful_frames = 0
        
        print(f"🚀 Starting parallel processing with {self.max_workers} workers...")
        print(f"   CPU cores available: {multiprocessing.cpu_count()}")
        print(f"   Memory safety: Workers limited for stability")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_frame = {executor.submit(process_single_frame, arg): arg[0] 
                              for arg in process_args}
            
            print(f"   Submitted {len(future_to_frame)} tasks to worker pool")
            
            # Collect results as they complete
            results = {}
            completed_count = 0
            total_tasks = len(future_to_frame)
            
            for future in future_to_frame:
                frame_idx = future_to_frame[future]
                try:
                    frame_idx, mesh_data = future.result(timeout=timeout_per_frame)  # Configurable timeout per frame
                    results[frame_idx] = mesh_data
                    completed_count += 1
                    
                    if mesh_data is not None:
                        successful_frames += 1
                        if completed_count % 10 == 0 or completed_count == total_tasks:
                            print(f"  ✅ Frame {frame_idx}: {mesh_data['vertex_count']} vertices")
                    else:
                        if completed_count % 10 == 0 or completed_count == total_tasks:
                            print(f"  ❌ Frame {frame_idx}: Failed")
                            
                    # Progress reporting every 10 completed frames
                    if completed_count % 10 == 0 or completed_count == total_tasks:
                        progress = (completed_count / total_tasks) * 100
                        print(f"   Progress: {completed_count}/{total_tasks} ({progress:.1f}%) - {successful_frames} successful")
                        
                except Exception as e:
                    print(f"  ❌ Frame {frame_idx}: Exception - {e}")
                    results[frame_idx] = None
                    completed_count += 1
                    
                    # Progress reporting for exceptions too
                    if completed_count % 10 == 0 or completed_count == total_tasks:
                        progress = (completed_count / total_tasks) * 100
                        print(f"   Progress: {completed_count}/{total_tasks} ({progress:.1f}%) - {successful_frames} successful")
        
        # Sort results by frame index and collect successful meshes
        sorted_results = [results[i] for i in sorted(results.keys())]
        mesh_sequence = [mesh for mesh in sorted_results if mesh is not None]
        
        # Final statistics
        processing_time = time.time() - start_time
        self.stats['frames_processed'] = len(landmarks_data)
        self.stats['meshes_generated'] = successful_frames
        self.stats['processing_time'] = processing_time
        
        if successful_frames > 0:
            total_error = sum(mesh['fitting_error'] for mesh in mesh_sequence)
            self.stats['average_fitting_error'] = total_error / successful_frames
        
        print(f"\n📊 PARALLEL PROCESSING COMPLETE!")
        print("-" * 70)
        print(f"   Processed Frames: {len(landmarks_data)}")
        print(f"   Successful Meshes: {successful_frames}")
        print(f"   Success Rate: {successful_frames/len(landmarks_data)*100:.1f}%")
        print(f"   Average Fitting Error: {self.stats['average_fitting_error']:.6f}")
        print(f"   Processing Time: {processing_time:.1f} seconds")
        print(f"   FPS: {len(landmarks_data)/processing_time:.2f}")
        print(f"   Speedup vs Serial: ~{self.max_workers:.1f}x (theoretical)")
        
        if mesh_sequence:
            print(f"\n💾 SAVING OUTPUTS...")
            print("-" * 70)
            
            # Save mesh data with metadata
            mesh_file = output_dir / f"{video_path.stem}_parallel_no_smoothing_meshes.pkl"
            pkl_data = {
                'mesh_sequence': mesh_sequence,
                'metadata': {
                    'processing_method': 'parallel_no_temporal_smoothing',
                    'max_workers': self.max_workers,
                    'total_frames': len(mesh_sequence),
                    'frame_skip': frame_skip,
                    'video_filename': video_path.name,
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'requires_post_processing_smoothing': True,
                    'original_temporal_alpha': 0.3,  # For reference in post-processing
                    'original_temporal_weights': {
                        'body_pose': 0.3,
                        'betas': 0.03,
                        'global_orient': 0.0,  # Was not in original temporal loss
                        'transl': 0.0          # Was not in original temporal loss  
                    }
                }
            }
            with open(mesh_file, 'wb') as f:
                pickle.dump(pkl_data, f)
            print(f"✅ Parallel mesh data saved: {mesh_file}")
            print(f"⚠️  NOTE: This PKL requires post-processing smoothing!")
            
            # Save statistics
            stats_file = output_dir / f"{video_path.stem}_parallel_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"✅ Statistics saved: {stats_file}")
            
            print(f"\n🎉 SUCCESS! Parallel processing complete in: {output_dir}")
            print(f"⚡ Next step: Apply post-processing smoothing to PKL file")
            
            return {
                'mesh_sequence': mesh_sequence,
                'mesh_file': mesh_file,
                'stats': self.stats,
                'output_dir': output_dir,
                'requires_smoothing': True
            }
        else:
            print(f"\n❌ FAILED: No meshes were generated")
            return None


def main():
    """Main execution function for parallel processing"""
    import sys
    
    # Parse command line arguments
    video_path = None
    output_dir = "parallel_no_smoothing_output"
    max_frames = None
    frame_skip = 1
    max_workers = None
    
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
        elif arg == '--max-workers':
            if i + 1 < len(sys.argv):
                max_workers = int(sys.argv[i + 1])
                i += 2
            else:
                print("ERROR: --max-workers requires a value")
                return
        elif not arg.startswith('--'):
            if video_path is None:
                video_path = arg
            elif output_dir == "parallel_no_smoothing_output":  # Default not yet changed
                output_dir = arg
            i += 1
        else:
            print(f"WARNING: Unknown argument: {arg}")
            i += 1
    
    print("🚀 PARALLEL PRODUCTION 3D HUMAN MESH PIPELINE (NO TEMPORAL SMOOTHING)")
    print("=" * 80)
    
    if not video_path:
        print("ERROR: No video file specified")
        print("Usage: python run_production_parallel_no_smoothing.py <video_file> [output_dir] [--max-frames N] [--frame-skip N] [--max-workers N]")
        return
    
    # Verify SMPL-X models
    models_dir = Path("models/smplx")
    if not models_dir.exists() or not any(models_dir.glob("*.npz")):
        print("❌ SMPL-X models not found!")
        print("Please download models from: https://smpl-x.is.tue.mpg.de/")
        return
    
    print("✅ SMPL-X models found")
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    # Initialize parallel pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = ParallelMasterPipeline(
        smplx_path="models/smplx",
        device=device,
        gender='neutral',
        max_workers=max_workers
    )
    
    print(f"\n🎬 Processing video: {video_path}")
    if max_frames:
        print(f"📊 Max frames: {max_frames}")
    print(f"📁 Output directory: {output_dir}")
    
    # Execute parallel pipeline
    results = pipeline.execute_parallel_pipeline(
        video_path,
        output_dir=output_dir,
        max_frames=max_frames,
        frame_skip=frame_skip
    )
    
    if results:
        print(f"\n🏆 PARALLEL PROCESSING ACCOMPLISHED!")
        print(f"Check '{results['output_dir']}' for results:")
        print(f"  📊 Parallel Mesh Data: {results['mesh_file'].name}")
        print(f"  📈 Statistics: Generated {results['stats']['meshes_generated']} meshes")
        print(f"  ⚠️  WARNING: PKL requires post-processing smoothing!")
    else:
        print(f"\n💥 PARALLEL PROCESSING FAILED!")


if __name__ == "__main__":
    main()
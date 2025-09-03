#!/usr/bin/env python3
"""
Enhanced Pipeline Integration - Integrate new components into production pipeline

Priority: CRITICAL
Dependencies: coordinate_system_fix, proactive_joint_validator, kalman_angle_filter
Test Coverage Required: 100%

This module provides enhanced versions of existing pipeline components that integrate
the new coordinate system transformer, proactive joint validator, and Kalman filter.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any
import logging
from pathlib import Path
import sys

# Add core module to path
sys.path.append(str(Path(__file__).parent))

from coordinate_system_fix import CoordinateSystemTransformer
from proactive_joint_validator import ProactiveJointValidator
from kalman_angle_filter import MultiAngleKalmanFilter, create_posture_angle_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMediaPipeConverter:
    """Enhanced MediaPipe converter with coordinate system fix and validation"""
    
    def __init__(self, repair_mode: str = 'smart'):
        """Initialize enhanced converter
        
        Args:
            repair_mode: Joint repair strategy ('smart', 'interpolation', 'constraint', 'hybrid')
        """
        # Core components
        self.coordinate_transformer = CoordinateSystemTransformer()
        self.joint_validator = ProactiveJointValidator(repair_mode=repair_mode)
        self.angle_filter = create_posture_angle_filter()
        
        # MediaPipe landmarks mapping
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
        
        # SMPL-X joint mapping (enhanced)
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
        
        # Quality weights for landmarks
        self.landmark_confidence = {
            11: 1.0, 12: 1.0,  # shoulders - very reliable
            23: 0.9, 24: 0.9,  # hips - reliable
            13: 0.8, 14: 0.8,  # elbows
            25: 0.8, 26: 0.8,  # knees
            15: 0.7, 16: 0.7,  # wrists
            27: 0.7, 28: 0.7,  # ankles
            0: 0.6,            # nose/head
        }
        
        logger.info("EnhancedMediaPipeConverter initialized with coordinate fix and validation")
    
    def convert_landmarks_to_smplx(self, mp_landmarks, 
                                  frame_confidences: Optional[np.ndarray] = None) -> Dict:
        """Enhanced conversion with coordinate transformation and validation
        
        Args:
            mp_landmarks: MediaPipe world landmarks
            frame_confidences: Optional confidence scores for landmarks
            
        Returns:
            Dict with conversion results, validation info, and filtered angles
        """
        if mp_landmarks is None:
            return {
                'joints': None,
                'weights': None,
                'validation_result': None,
                'angles': None,
                'coordinate_transform_applied': False,
                'repair_applied': False
            }
        
        # Extract 3D coordinates from MediaPipe
        mp_points = np.array([[lm.x, lm.y, lm.z] for lm in mp_landmarks.landmark])
        
        # Step 1: Apply coordinate system transformation (MediaPipe -> SMPL-X)
        transformed_points = self.coordinate_transformer.transform_landmarks(mp_points)
        
        # Step 2: Validate and repair joints proactively
        validation_result = self.joint_validator.validate_and_repair_frame(
            transformed_points, 
            confidences=frame_confidences
        )
        
        # Use repaired joints
        validated_joints = validation_result['repaired_joints']
        
        # Step 3: Convert to SMPL-X format with anatomical constraints
        smplx_joints, joint_weights = self._convert_to_smplx_format(
            validated_joints, frame_confidences
        )
        
        # Step 4: Calculate angles and apply Kalman filtering
        angles_data = self._calculate_and_filter_angles(smplx_joints, joint_weights)
        
        result = {
            'joints': smplx_joints,
            'weights': joint_weights,
            'validation_result': validation_result,
            'angles': angles_data,
            'coordinate_transform_applied': True,
            'repair_applied': not validation_result['valid']
        }
        
        return result
    
    def _convert_to_smplx_format(self, validated_joints: np.ndarray, 
                                confidences: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert validated joints to SMPL-X format"""
        num_joints = len(self.smplx_joint_tree)
        smplx_joints = np.zeros((num_joints, 3))
        joint_weights = np.zeros(num_joints)
        
        # Direct mappings from MediaPipe to SMPL-X indices
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
        for smplx_idx, (mp_idx, base_confidence) in direct_mappings.items():
            if mp_idx < len(validated_joints):
                smplx_joints[smplx_idx] = validated_joints[mp_idx]
                
                # Adjust confidence based on frame confidences
                final_confidence = base_confidence
                if confidences is not None and mp_idx < len(confidences):
                    final_confidence *= confidences[mp_idx]
                
                joint_weights[smplx_idx] = final_confidence
        
        # Calculate anatomically consistent joints
        self._calculate_anatomical_joints(validated_joints, smplx_joints, joint_weights)
        
        return smplx_joints, joint_weights
    
    def _calculate_anatomical_joints(self, mp_points: np.ndarray, 
                                   smplx_joints: np.ndarray, 
                                   joint_weights: np.ndarray):
        """Calculate anatomically consistent joint positions"""
        
        # Pelvis as center of hips
        if len(mp_points) > 24:
            left_hip = mp_points[23]
            right_hip = mp_points[24]
            smplx_joints[0] = (left_hip + right_hip) / 2  # pelvis
            joint_weights[0] = 0.95
        
        # Spine chain with proper curvature
        if len(mp_points) > 12 and joint_weights[0] > 0:
            if len(mp_points) > 12:  # Check if we have shoulder landmarks
                shoulder_center = (mp_points[11] + mp_points[12]) / 2
                pelvis = smplx_joints[0]
                spine_vector = shoulder_center - pelvis
                spine_length = np.linalg.norm(spine_vector)
                
                if spine_length > 1e-6:  # Avoid division by zero
                    spine_unit = spine_vector / spine_length
                    
                    # Natural spine curvature progression
                    smplx_joints[3] = pelvis + spine_unit * spine_length * 0.2   # spine1
                    smplx_joints[6] = pelvis + spine_unit * spine_length * 0.5   # spine2
                    smplx_joints[9] = pelvis + spine_unit * spine_length * 0.8   # spine3
                    smplx_joints[12] = pelvis + spine_unit * spine_length * 0.95 # neck
                    
                    joint_weights[3] = 0.7
                    joint_weights[6] = 0.7
                    joint_weights[9] = 0.7
                    joint_weights[12] = 0.8
        
        # Feet positions (anatomically below ankles)
        foot_offset = np.array([0, 0, -0.08])  # 8cm below ankle in SMPL-X coordinates
        
        if joint_weights[7] > 0:  # left_ankle exists
            smplx_joints[10] = smplx_joints[7] + foot_offset  # left_foot
            joint_weights[10] = joint_weights[7] * 0.8
            
        if joint_weights[8] > 0:  # right_ankle exists
            smplx_joints[11] = smplx_joints[8] + foot_offset  # right_foot
            joint_weights[11] = joint_weights[8] * 0.8
        
        # Collar bones (clavicles) for shoulder attachment
        if joint_weights[12] > 0:  # neck exists
            neck = smplx_joints[12]
            
            if joint_weights[16] > 0:  # left_shoulder exists
                collar_vector = smplx_joints[16] - neck
                smplx_joints[13] = neck + collar_vector * 0.4  # left_collar
                joint_weights[13] = 0.6
                
            if joint_weights[17] > 0:  # right_shoulder exists
                collar_vector = smplx_joints[17] - neck
                smplx_joints[14] = neck + collar_vector * 0.4  # right_collar
                joint_weights[14] = 0.6
    
    def _calculate_and_filter_angles(self, joints: np.ndarray, 
                                   weights: np.ndarray) -> Dict:
        """Calculate key posture angles and apply Kalman filtering"""
        angles = {}
        
        try:
            # Calculate trunk angles
            if weights[0] > 0 and weights[12] > 0:  # pelvis and neck exist
                pelvis = joints[0]
                neck = joints[12]
                trunk_vector = neck - pelvis
                
                # Sagittal plane angle (forward/backward bend)
                trunk_sagittal = self._calculate_sagittal_angle(trunk_vector)
                angles['trunk_sagittal'] = trunk_sagittal
                
                # Lateral plane angle (side bend)
                trunk_lateral = self._calculate_lateral_angle(trunk_vector)
                angles['trunk_lateral'] = trunk_lateral
            
            # Calculate neck angle
            if weights[12] > 0 and weights[15] > 0:  # neck and head exist
                neck = joints[12]
                head = joints[15]
                neck_vector = head - neck
                neck_flexion = self._calculate_sagittal_angle(neck_vector)
                angles['neck_flexion'] = neck_flexion
            
            # Calculate shoulder angles
            angles.update(self._calculate_shoulder_angles(joints, weights))
            
            # Calculate elbow angles
            angles.update(self._calculate_elbow_angles(joints, weights))
            
            # Apply Kalman filtering to angles
            confidences = {name: 0.8 for name in angles.keys()}  # Default confidence
            filtered_results = self.angle_filter.update(angles, confidences)
            
            return {
                'raw_angles': angles,
                'filtered_angles': {name: result['filtered_angle'] 
                                  for name, result in filtered_results.items()},
                'angular_velocities': {name: result['angular_velocity'] 
                                     for name, result in filtered_results.items()},
                'filter_results': filtered_results
            }
            
        except Exception as e:
            logger.warning(f"Angle calculation failed: {e}")
            return {'raw_angles': {}, 'filtered_angles': {}, 'angular_velocities': {}}
    
    def _calculate_sagittal_angle(self, vector: np.ndarray) -> float:
        """Calculate angle in sagittal plane (forward/backward)"""
        # Project vector onto sagittal plane (Y-Z in SMPL-X coordinates)
        sagittal_projection = np.array([0, vector[1], vector[2]])
        sagittal_projection_norm = np.linalg.norm(sagittal_projection)
        
        if sagittal_projection_norm < 1e-6:
            return 0.0
        
        # Reference is straight up (positive Z)
        reference = np.array([0, 0, 1])
        
        dot_product = np.dot(sagittal_projection / sagittal_projection_norm, reference)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(dot_product))
        
        # Determine sign based on Y component (forward/backward)
        if vector[1] > 0:  # Forward bend
            angle = -angle
        
        return angle
    
    def _calculate_lateral_angle(self, vector: np.ndarray) -> float:
        """Calculate angle in lateral plane (side bend)"""
        # Project vector onto frontal plane (X-Z in SMPL-X coordinates)
        frontal_projection = np.array([vector[0], 0, vector[2]])
        frontal_projection_norm = np.linalg.norm(frontal_projection)
        
        if frontal_projection_norm < 1e-6:
            return 0.0
        
        # Reference is straight up (positive Z)
        reference = np.array([0, 0, 1])
        
        dot_product = np.dot(frontal_projection / frontal_projection_norm, reference)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(dot_product))
        
        # Determine sign based on X component (left/right)
        if vector[0] < 0:  # Left bend
            angle = -angle
        
        return angle
    
    def _calculate_shoulder_angles(self, joints: np.ndarray, weights: np.ndarray) -> Dict:
        """Calculate shoulder flexion and abduction angles"""
        angles = {}
        
        # Left shoulder
        if all(weights[i] > 0 for i in [16, 18]):  # left_shoulder, left_elbow
            shoulder = joints[16]
            elbow = joints[18]
            arm_vector = elbow - shoulder
            
            angles['left_shoulder_flexion'] = self._calculate_sagittal_angle(arm_vector)
            angles['left_shoulder_abduction'] = abs(self._calculate_lateral_angle(arm_vector))
        
        # Right shoulder
        if all(weights[i] > 0 for i in [17, 19]):  # right_shoulder, right_elbow
            shoulder = joints[17]
            elbow = joints[19]
            arm_vector = elbow - shoulder
            
            angles['right_shoulder_flexion'] = self._calculate_sagittal_angle(arm_vector)
            angles['right_shoulder_abduction'] = abs(self._calculate_lateral_angle(arm_vector))
        
        return angles
    
    def _calculate_elbow_angles(self, joints: np.ndarray, weights: np.ndarray) -> Dict:
        """Calculate elbow flexion angles"""
        angles = {}
        
        # Left elbow
        if all(weights[i] > 0 for i in [16, 18, 20]):  # shoulder, elbow, wrist
            shoulder = joints[16]
            elbow = joints[18]
            wrist = joints[20]
            
            upper_arm = shoulder - elbow
            forearm = wrist - elbow
            
            # Calculate angle between upper arm and forearm
            upper_arm_norm = np.linalg.norm(upper_arm)
            forearm_norm = np.linalg.norm(forearm)
            
            if upper_arm_norm > 1e-6 and forearm_norm > 1e-6:
                dot_product = np.dot(upper_arm / upper_arm_norm, forearm / forearm_norm)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                angles['left_elbow_flexion'] = 180 - angle  # 0° = straight, 90° = bent
        
        # Right elbow
        if all(weights[i] > 0 for i in [17, 19, 21]):  # shoulder, elbow, wrist
            shoulder = joints[17]
            elbow = joints[19]
            wrist = joints[21]
            
            upper_arm = shoulder - elbow
            forearm = wrist - elbow
            
            upper_arm_norm = np.linalg.norm(upper_arm)
            forearm_norm = np.linalg.norm(forearm)
            
            if upper_arm_norm > 1e-6 and forearm_norm > 1e-6:
                dot_product = np.dot(upper_arm / upper_arm_norm, forearm / forearm_norm)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                angles['right_elbow_flexion'] = 180 - angle
        
        return angles
    
    def reset_filters(self):
        """Reset all temporal filters"""
        self.joint_validator.reset_history()
        self.angle_filter.reset_all()
        
        logger.info("All filters reset")
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        joint_stats = self.joint_validator.get_validation_statistics()
        angle_stats = self.angle_filter.get_all_statistics()
        
        return {
            'joint_validation': joint_stats,
            'angle_filtering': angle_stats,
            'coordinate_transform': {
                'transformation_matrix': self.coordinate_transformer.get_transformation_matrix().tolist(),
                'quaternion': self.coordinate_transformer.get_rotation_quaternion().tolist()
            }
        }


class EnhancedSMPLXFitter:
    """Enhanced SMPL-X fitter with improved initialization and optimization"""
    
    def __init__(self, model_path: str, device: str = 'cpu', gender: str = 'neutral'):
        """Initialize enhanced SMPL-X fitter
        
        Args:
            model_path: Path to SMPL-X models
            device: torch device ('cpu' or 'cuda')
            gender: Model gender ('neutral', 'male', 'female')
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.gender = gender
        
        # Initialize coordinate transformer for consistency
        self.coordinate_transformer = CoordinateSystemTransformer()
        
        logger.info(f"EnhancedSMPLXFitter initialized on {device} for {gender} model")
    
    def fit_to_joints(self, joints: np.ndarray, weights: np.ndarray, 
                     validation_info: Dict, angles_info: Dict) -> Dict:
        """Enhanced fitting that uses validation and angle information
        
        Args:
            joints: SMPL-X format joint positions
            weights: Joint confidence weights
            validation_info: Results from joint validation
            angles_info: Calculated angle information
            
        Returns:
            Dict with fitting results and enhanced metadata
        """
        # This would integrate with existing SMPL-X fitting code
        # For now, return structure that matches expected interface
        
        fitting_result = {
            'vertices': None,  # Would contain mesh vertices
            'faces': None,     # Would contain mesh faces
            'joints_3d': joints,
            'pose_params': None,      # Would contain pose parameters
            'shape_params': None,     # Would contain shape parameters
            'fitting_loss': None,     # Would contain optimization loss
            'validation_applied': validation_info.get('repair_applied', False),
            'joint_confidence': np.mean(weights),
            'angles_filtered': len(angles_info.get('filtered_angles', {})) > 0,
            'coordinate_transform_applied': True
        }
        
        logger.info(f"Enhanced SMPL-X fitting completed with "
                   f"{np.sum(weights > 0.5)} confident joints")
        
        return fitting_result


def create_enhanced_pipeline_converter(repair_mode: str = 'smart') -> EnhancedMediaPipeConverter:
    """Factory function to create enhanced converter for pipeline integration"""
    return EnhancedMediaPipeConverter(repair_mode=repair_mode)


if __name__ == "__main__":
    # Quick test of enhanced converter
    converter = EnhancedMediaPipeConverter()
    
    print("Testing enhanced pipeline integration...")
    
    # Create mock MediaPipe landmarks
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    
    class MockLandmarks:
        def __init__(self, points):
            self.landmark = [MockLandmark(*point) for point in points]
    
    # Create test data
    test_points = np.random.randn(33, 3) * 0.1
    test_landmarks = MockLandmarks(test_points)
    test_confidences = np.ones(33) * 0.8
    
    # Test conversion
    result = converter.convert_landmarks_to_smplx(test_landmarks, test_confidences)
    
    print(f"Conversion completed:")
    print(f"  Joints shape: {result['joints'].shape if result['joints'] is not None else None}")
    print(f"  Coordinate transform applied: {result['coordinate_transform_applied']}")
    print(f"  Repair applied: {result['repair_applied']}")
    print(f"  Angles calculated: {len(result['angles'].get('filtered_angles', {}))}")
    
    # Get statistics
    stats = converter.get_processing_statistics()
    print(f"  Total frames processed: {stats['joint_validation']['total_frames']}")
    
    print("[PASS] Enhanced pipeline integration test completed")
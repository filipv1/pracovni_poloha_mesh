#!/usr/bin/env python3
"""
Proactive Joint Validator - Validate and fix joint data BEFORE SMPL-X fitting

Priority: CRITICAL
Dependencies: numpy, scipy
Test Coverage Required: 100%

This module implements proactive validation and repair of joint positions
to prevent SMPL-X fitting failures and improve pose accuracy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProactiveJointValidator:
    """Proactive validation and repair of 3D joint positions
    
    This validator checks joint positions for biomechanical constraints,
    temporal consistency, and anatomical plausibility BEFORE SMPL-X fitting,
    eliminating the need for post-processing repairs.
    """
    
    # MediaPipe landmark indices for key joints
    JOINT_INDICES = {
        'nose': 0,
        'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8,
        'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_pinky': 17, 'left_index': 18, 'left_thumb': 19,
        'right_pinky': 20, 'right_index': 21, 'right_thumb': 22,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Biomechanical constraints (in meters, for normalized coordinates scale by body_scale)
    BONE_LENGTH_CONSTRAINTS = {
        # Upper body
        ('left_shoulder', 'left_elbow'): (0.25, 0.40),     # Humerus
        ('right_shoulder', 'right_elbow'): (0.25, 0.40),   
        ('left_elbow', 'left_wrist'): (0.20, 0.35),        # Forearm
        ('right_elbow', 'right_wrist'): (0.20, 0.35),      
        ('left_shoulder', 'right_shoulder'): (0.25, 0.50), # Shoulder width
        
        # Lower body
        ('left_hip', 'left_knee'): (0.35, 0.55),           # Femur
        ('right_hip', 'right_knee'): (0.35, 0.55),         
        ('left_knee', 'left_ankle'): (0.30, 0.50),         # Tibia
        ('right_knee', 'right_ankle'): (0.30, 0.50),       
        ('left_hip', 'right_hip'): (0.20, 0.40),           # Hip width
        
        # Torso
        ('left_shoulder', 'left_hip'): (0.40, 0.70),       # Torso length
        ('right_shoulder', 'right_hip'): (0.40, 0.70),     
    }
    
    # Joint angle constraints (degrees)
    ANGLE_CONSTRAINTS = {
        # Elbow: should not hyper-extend
        ('left_shoulder', 'left_elbow', 'left_wrist'): (0, 160),
        ('right_shoulder', 'right_elbow', 'right_wrist'): (0, 160),
        
        # Knee: should not hyper-extend
        ('left_hip', 'left_knee', 'left_ankle'): (0, 160),
        ('right_hip', 'right_knee', 'right_ankle'): (0, 160),
    }
    
    def __init__(self, confidence_threshold: float = 0.5,
                 temporal_window: int = 5,
                 repair_mode: str = 'smart'):
        """Initialize proactive joint validator
        
        Args:
            confidence_threshold: Minimum confidence for joint validation
            temporal_window: Number of frames for temporal consistency
            repair_mode: 'smart', 'interpolation', 'constraint' or 'hybrid'
        """
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.repair_mode = repair_mode
        
        # Temporal history for consistency checking
        self.joint_history = []
        self.confidence_history = []
        
        # Statistics tracking
        self.validation_stats = {
            'total_frames': 0,
            'invalid_frames': 0,
            'repaired_joints': 0,
            'biomechanical_violations': 0,
            'temporal_inconsistencies': 0,
            'confidence_issues': 0
        }
        
        logger.info(f"ProactiveJointValidator initialized with {repair_mode} repair mode")
    
    def validate_and_repair_frame(self, joints_3d: np.ndarray, 
                                 confidences: Optional[np.ndarray] = None) -> Dict:
        """Validate and repair a single frame of joint data
        
        Args:
            joints_3d: (33, 3) array of 3D joint positions
            confidences: Optional (33,) array of confidence scores
            
        Returns:
            Dict with validation results and repaired joints
        """
        if joints_3d is None:
            raise ValueError("joints_3d cannot be None")
        
        joints_3d = np.asarray(joints_3d, dtype=np.float32)
        if joints_3d.shape != (33, 3):
            raise ValueError(f"Expected joints shape (33, 3), got {joints_3d.shape}")
        
        self.validation_stats['total_frames'] += 1
        
        # Initialize result structure
        result = {
            'valid': True,
            'repaired_joints': joints_3d.copy(),
            'confidence_mask': np.ones(33, dtype=bool) if confidences is None else confidences > self.confidence_threshold,
            'violations': [],
            'repair_log': []
        }
        
        # 1. Confidence-based validation
        if confidences is not None:
            low_conf_joints = self._validate_confidence(confidences, result)
            if len(low_conf_joints) > 0:
                result['valid'] = False
                self.validation_stats['confidence_issues'] += len(low_conf_joints)
        
        # 2. Biomechanical constraint validation
        biomech_violations = self._validate_biomechanical_constraints(result['repaired_joints'], result)
        if len(biomech_violations) > 0:
            result['valid'] = False
            self.validation_stats['biomechanical_violations'] += len(biomech_violations)
        
        # 3. Temporal consistency validation
        if len(self.joint_history) > 0:
            temporal_issues = self._validate_temporal_consistency(result['repaired_joints'], result)
            if len(temporal_issues) > 0:
                result['valid'] = False
                self.validation_stats['temporal_inconsistencies'] += len(temporal_issues)
        
        # 4. Apply repairs if needed
        if not result['valid']:
            self.validation_stats['invalid_frames'] += 1
            repaired_joints = self._apply_repairs(result)
            result['repaired_joints'] = repaired_joints
            
            # Validate repairs
            validation_after_repair = self._quick_validate(repaired_joints)
            result['repair_successful'] = validation_after_repair['valid']
        
        # 5. Update temporal history
        self._update_temporal_history(result['repaired_joints'], confidences)
        
        return result
    
    def _validate_confidence(self, confidences: np.ndarray, result: Dict) -> List[int]:
        """Validate confidence scores and identify low-confidence joints"""
        low_conf_joints = []
        
        for i, conf in enumerate(confidences):
            if conf < self.confidence_threshold:
                low_conf_joints.append(i)
                result['violations'].append({
                    'type': 'low_confidence',
                    'joint_idx': i,
                    'confidence': conf,
                    'threshold': self.confidence_threshold
                })
        
        return low_conf_joints
    
    def _validate_biomechanical_constraints(self, joints_3d: np.ndarray, result: Dict) -> List[Dict]:
        """Validate biomechanical constraints (bone lengths, joint angles)"""
        violations = []
        
        # Estimate body scale from shoulder width
        body_scale = self._estimate_body_scale(joints_3d)
        
        # Check bone length constraints
        for (joint1_name, joint2_name), (min_len, max_len) in self.BONE_LENGTH_CONSTRAINTS.items():
            if joint1_name not in self.JOINT_INDICES or joint2_name not in self.JOINT_INDICES:
                continue
                
            idx1 = self.JOINT_INDICES[joint1_name]
            idx2 = self.JOINT_INDICES[joint2_name]
            
            bone_length = np.linalg.norm(joints_3d[idx1] - joints_3d[idx2])
            expected_min = min_len * body_scale
            expected_max = max_len * body_scale
            
            if bone_length < expected_min or bone_length > expected_max:
                violations.append({
                    'type': 'bone_length',
                    'joints': (joint1_name, joint2_name),
                    'indices': (idx1, idx2),
                    'actual_length': bone_length,
                    'expected_range': (expected_min, expected_max),
                    'severity': min(abs(bone_length - expected_min) / expected_min,
                                  abs(bone_length - expected_max) / expected_max)
                })
        
        # Check joint angle constraints
        for (j1_name, j2_name, j3_name), (min_angle, max_angle) in self.ANGLE_CONSTRAINTS.items():
            if any(name not in self.JOINT_INDICES for name in [j1_name, j2_name, j3_name]):
                continue
                
            idx1 = self.JOINT_INDICES[j1_name]
            idx2 = self.JOINT_INDICES[j2_name]
            idx3 = self.JOINT_INDICES[j3_name]
            
            angle = self._calculate_joint_angle(joints_3d[idx1], joints_3d[idx2], joints_3d[idx3])
            
            if angle < min_angle or angle > max_angle:
                violations.append({
                    'type': 'joint_angle',
                    'joints': (j1_name, j2_name, j3_name),
                    'indices': (idx1, idx2, idx3),
                    'actual_angle': angle,
                    'expected_range': (min_angle, max_angle),
                    'severity': min(abs(angle - min_angle) / min_angle if min_angle > 0 else abs(angle - min_angle),
                                  abs(angle - max_angle) / max_angle)
                })
        
        result['violations'].extend(violations)
        return violations
    
    def _validate_temporal_consistency(self, joints_3d: np.ndarray, result: Dict) -> List[Dict]:
        """Validate temporal consistency with previous frames"""
        if len(self.joint_history) == 0:
            return []
        
        violations = []
        prev_joints = self.joint_history[-1]
        
        # Calculate joint velocities (position changes)
        joint_velocities = np.linalg.norm(joints_3d - prev_joints, axis=1)
        
        # Detect unrealistic movements (threshold based on body scale)
        body_scale = self._estimate_body_scale(joints_3d)
        velocity_threshold = body_scale * 0.3  # 30% of body scale per frame
        
        for i, velocity in enumerate(joint_velocities):
            if velocity > velocity_threshold:
                violations.append({
                    'type': 'temporal_inconsistency',
                    'joint_idx': i,
                    'velocity': velocity,
                    'threshold': velocity_threshold,
                    'severity': velocity / velocity_threshold
                })
        
        result['violations'].extend(violations)
        return violations
    
    def _apply_repairs(self, result: Dict) -> np.ndarray:
        """Apply repairs based on detected violations"""
        repaired_joints = result['repaired_joints'].copy()
        
        if self.repair_mode == 'smart':
            repaired_joints = self._smart_repair(repaired_joints, result['violations'])
        elif self.repair_mode == 'interpolation':
            repaired_joints = self._interpolation_repair(repaired_joints, result['violations'])
        elif self.repair_mode == 'constraint':
            repaired_joints = self._constraint_repair(repaired_joints, result['violations'])
        elif self.repair_mode == 'hybrid':
            repaired_joints = self._hybrid_repair(repaired_joints, result['violations'])
        
        return repaired_joints
    
    def _smart_repair(self, joints_3d: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """Smart repair using multiple strategies based on violation type"""
        repaired = joints_3d.copy()
        
        for violation in violations:
            if violation['type'] == 'low_confidence':
                # Use temporal interpolation for low confidence joints
                joint_idx = violation['joint_idx']
                if len(self.joint_history) > 0:
                    # Simple linear interpolation from previous frame
                    repaired[joint_idx] = self.joint_history[-1][joint_idx]
                else:
                    # No history available, set to zero or use anatomical prior
                    repaired[joint_idx] = np.array([0.0, 0.0, 0.0])
                    
            elif violation['type'] == 'bone_length':
                # Adjust bone length to fit constraints
                idx1, idx2 = violation['indices']
                expected_min, expected_max = violation['expected_range']
                current_length = violation['actual_length']
                
                # Target length (midpoint of valid range)
                target_length = (expected_min + expected_max) / 2
                
                # Adjust the more distal joint (higher index)
                if current_length > 1e-8:  # Avoid division by zero
                    if idx1 < idx2:
                        direction = (repaired[idx2] - repaired[idx1]) / current_length
                        repaired[idx2] = repaired[idx1] + direction * target_length
                    else:
                        direction = (repaired[idx1] - repaired[idx2]) / current_length
                        repaired[idx1] = repaired[idx2] + direction * target_length
                else:
                    # If joints are at same position, create minimal separation
                    if idx1 < idx2:
                        repaired[idx2] = repaired[idx1] + np.array([0, target_length, 0])
                    else:
                        repaired[idx1] = repaired[idx2] + np.array([0, target_length, 0])
                    
            elif violation['type'] == 'joint_angle':
                # Adjust joint angle to fit constraints
                idx1, idx2, idx3 = violation['indices']
                min_angle, max_angle = violation['expected_range']
                current_angle = violation['actual_angle']
                
                # Target angle
                target_angle = np.clip(current_angle, min_angle, max_angle)
                if target_angle == current_angle:
                    target_angle = (min_angle + max_angle) / 2
                
                # Adjust the end joint (idx3) to achieve target angle
                repaired = self._adjust_joint_angle(repaired, idx1, idx2, idx3, target_angle)
                
            elif violation['type'] == 'temporal_inconsistency':
                # Use temporal smoothing
                joint_idx = violation['joint_idx']
                if len(self.joint_history) >= 2:
                    # Use weighted average with previous frames
                    weights = np.array([0.1, 0.3, 0.6])  # More weight to current frame
                    history_len = min(len(self.joint_history), len(weights) - 1)
                    
                    weighted_pos = repaired[joint_idx] * weights[-1]
                    for i in range(history_len):
                        weighted_pos += self.joint_history[-(i+1)][joint_idx] * weights[-(i+2)]
                    
                    repaired[joint_idx] = weighted_pos / np.sum(weights[:history_len+1])
        
        return repaired
    
    def _interpolation_repair(self, joints_3d: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """Repair using temporal interpolation"""
        if len(self.joint_history) == 0:
            return joints_3d
        
        repaired = joints_3d.copy()
        
        # Identify joints that need repair
        repair_joints = set()
        for violation in violations:
            if 'joint_idx' in violation:
                repair_joints.add(violation['joint_idx'])
            elif 'indices' in violation:
                repair_joints.update(violation['indices'])
        
        # Apply interpolation
        for joint_idx in repair_joints:
            if len(self.joint_history) == 1:
                repaired[joint_idx] = self.joint_history[0][joint_idx]
            else:
                # Linear interpolation from last two frames
                prev1 = self.joint_history[-1][joint_idx]
                prev2 = self.joint_history[-2][joint_idx]
                velocity = prev1 - prev2
                repaired[joint_idx] = prev1 + velocity * 0.5  # Damped prediction
        
        return repaired
    
    def _constraint_repair(self, joints_3d: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """Repair using constraint optimization"""
        repaired = joints_3d.copy()
        
        # Define optimization problem
        def objective(x):
            # Minimize change from original pose
            joints_reshaped = x.reshape(-1, 3)
            return np.sum((joints_reshaped - joints_3d.astype(np.float64)) ** 2)
        
        def constraint_func(x):
            joints_reshaped = x.reshape(-1, 3)
            constraints = []
            
            # Add bone length constraints
            body_scale = self._estimate_body_scale(joints_reshaped)
            for (joint1_name, joint2_name), (min_len, max_len) in self.BONE_LENGTH_CONSTRAINTS.items():
                if joint1_name not in self.JOINT_INDICES or joint2_name not in self.JOINT_INDICES:
                    continue
                    
                idx1 = self.JOINT_INDICES[joint1_name]
                idx2 = self.JOINT_INDICES[joint2_name]
                
                bone_length = np.linalg.norm(joints_reshaped[idx1] - joints_reshaped[idx2])
                expected_min = min_len * body_scale
                expected_max = max_len * body_scale
                
                # Add constraint: expected_min <= bone_length <= expected_max
                constraints.append(bone_length - expected_min)  # >= 0
                constraints.append(expected_max - bone_length)  # >= 0
            
            return np.array(constraints)
        
        try:
            # Solve constrained optimization
            result = minimize(
                objective,
                joints_3d.astype(np.float64).flatten(),
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                repaired = result.x.reshape(-1, 3)
            
        except Exception as e:
            logger.warning(f"Constraint optimization failed: {e}")
        
        return repaired
    
    def _hybrid_repair(self, joints_3d: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """Hybrid repair combining multiple strategies"""
        # Start with smart repair
        repaired = self._smart_repair(joints_3d, violations)
        
        # If still invalid, try constraint repair
        quick_validation = self._quick_validate(repaired)
        if not quick_validation['valid']:
            repaired = self._constraint_repair(repaired, violations)
        
        return repaired
    
    def _estimate_body_scale(self, joints_3d: np.ndarray) -> float:
        """Estimate body scale from joint positions"""
        # Use shoulder width as reference
        left_shoulder = joints_3d[self.JOINT_INDICES['left_shoulder']]
        right_shoulder = joints_3d[self.JOINT_INDICES['right_shoulder']]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Average shoulder width is about 0.35m, use as scale reference
        return max(shoulder_width / 0.35, 0.5)  # Minimum scale to prevent division by zero
    
    def _calculate_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at joint p2 between vectors p2->p1 and p2->p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Calculate angle
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        
        return np.degrees(angle_rad)
    
    def _adjust_joint_angle(self, joints_3d: np.ndarray, idx1: int, idx2: int, idx3: int, target_angle: float) -> np.ndarray:
        """Adjust joint at idx2 to achieve target angle"""
        repaired = joints_3d.copy()
        
        p1, p2, p3 = joints_3d[idx1], joints_3d[idx2], joints_3d[idx3]
        
        # Calculate current vectors
        v1 = p1 - p2  # Vector from joint to first point
        v2 = p3 - p2  # Vector from joint to second point
        
        # Calculate rotation needed
        current_angle = self._calculate_joint_angle(p1, p2, p3)
        angle_diff = np.radians(target_angle - current_angle)
        
        # Find rotation axis (perpendicular to both vectors)
        rotation_axis = np.cross(v1, v2)
        if np.linalg.norm(rotation_axis) < 1e-8:
            return repaired  # Vectors are parallel, can't adjust
        
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Apply rotation to v2 (adjust point p3)
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_rotvec(angle_diff * rotation_axis)
        v2_rotated = rotation.apply(v2)
        
        # Update joint position
        repaired[idx3] = p2 + v2_rotated
        
        return repaired
    
    def _quick_validate(self, joints_3d: np.ndarray) -> Dict:
        """Quick validation check without full analysis"""
        # Check for NaN or infinite values
        if np.any(np.isnan(joints_3d)) or np.any(np.isinf(joints_3d)):
            return {'valid': False, 'reason': 'invalid_values'}
        
        # Check basic biomechanical constraints
        body_scale = self._estimate_body_scale(joints_3d)
        
        # Quick bone length check on major bones
        major_bones = [
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee')
        ]
        
        for joint1_name, joint2_name in major_bones:
            if joint1_name not in self.JOINT_INDICES or joint2_name not in self.JOINT_INDICES:
                continue
                
            idx1 = self.JOINT_INDICES[joint1_name]
            idx2 = self.JOINT_INDICES[joint2_name]
            
            if (joint1_name, joint2_name) in self.BONE_LENGTH_CONSTRAINTS:
                min_len, max_len = self.BONE_LENGTH_CONSTRAINTS[(joint1_name, joint2_name)]
                bone_length = np.linalg.norm(joints_3d[idx1] - joints_3d[idx2])
                expected_min = min_len * body_scale
                expected_max = max_len * body_scale
                
                if bone_length < expected_min * 0.5 or bone_length > expected_max * 2.0:
                    return {'valid': False, 'reason': 'bone_length_violation'}
        
        return {'valid': True}
    
    def _update_temporal_history(self, joints_3d: np.ndarray, confidences: Optional[np.ndarray]):
        """Update temporal history for consistency checking"""
        self.joint_history.append(joints_3d.copy())
        if confidences is not None:
            self.confidence_history.append(confidences.copy())
        
        # Maintain fixed window size
        if len(self.joint_history) > self.temporal_window:
            self.joint_history.pop(0)
        if len(self.confidence_history) > self.temporal_window:
            self.confidence_history.pop(0)
    
    def get_validation_statistics(self) -> Dict:
        """Get validation and repair statistics"""
        stats = self.validation_stats.copy()
        
        if stats['total_frames'] > 0:
            stats['invalid_percentage'] = (stats['invalid_frames'] / stats['total_frames']) * 100
            stats['average_repairs_per_frame'] = stats['repaired_joints'] / stats['total_frames']
        
        return stats
    
    def reset_history(self):
        """Reset temporal history"""
        self.joint_history.clear()
        self.confidence_history.clear()
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_frames': 0,
            'invalid_frames': 0,
            'repaired_joints': 0,
            'biomechanical_violations': 0,
            'temporal_inconsistencies': 0,
            'confidence_issues': 0
        }


def create_test_pose_with_violations() -> Tuple[np.ndarray, np.ndarray]:
    """Create test pose with known violations for validation testing"""
    # Start with valid pose
    joints = np.zeros((33, 3), dtype=np.float32)
    
    # Basic standing pose
    joints[0] = [0, 0.8, 0]    # nose
    joints[11] = [-0.2, 0.6, 0]  # left shoulder
    joints[12] = [0.2, 0.6, 0]   # right shoulder
    joints[13] = [-0.25, 0.3, 0] # left elbow
    joints[14] = [0.25, 0.3, 0]  # right elbow
    joints[15] = [-0.3, 0, 0]    # left wrist
    joints[16] = [0.3, 0, 0]     # right wrist
    joints[23] = [-0.15, 0, 0]   # left hip
    joints[24] = [0.15, 0, 0]    # right hip
    joints[25] = [-0.2, -0.4, 0] # left knee
    joints[26] = [0.2, -0.4, 0]  # right knee
    joints[27] = [-0.25, -0.8, 0]  # left ankle
    joints[28] = [0.25, -0.8, 0]   # right ankle
    
    # Introduce violations
    # 1. Make left arm too long (bone length violation)
    joints[15] = [-0.6, 0, 0]  # Move left wrist too far
    
    # 2. Create impossible elbow angle
    joints[14] = [0.15, 0.7, 0]  # Move right elbow to create hyper-extension
    
    # Confidence scores (some low confidence joints)
    confidences = np.ones(33, dtype=np.float32) * 0.8
    confidences[15] = 0.3  # Low confidence for problematic left wrist
    confidences[14] = 0.4  # Low confidence for problematic right elbow
    
    return joints, confidences


if __name__ == "__main__":
    # Quick test
    validator = ProactiveJointValidator(repair_mode='smart')
    test_joints, test_confidences = create_test_pose_with_violations()
    
    print("Testing proactive joint validator...")
    
    # Validate and repair
    result = validator.validate_and_repair_frame(test_joints, test_confidences)
    
    print(f"Original pose valid: {result['valid']}")
    print(f"Number of violations: {len(result['violations'])}")
    print(f"Repair successful: {result.get('repair_successful', 'N/A')}")
    
    if result['violations']:
        print("\nDetected violations:")
        for i, violation in enumerate(result['violations'][:3]):  # Show first 3
            print(f"  {i+1}. {violation['type']}: {violation}")
    
    # Get statistics
    stats = validator.get_validation_statistics()
    print(f"\nValidation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("✓ Proactive joint validator test completed")
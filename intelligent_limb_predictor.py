#!/usr/bin/env python3
"""
Intelligent Limb Predictor
Predicts missing or low-confidence limb positions based on body mechanics
Alternative to global temporal smoothing - only fixes problematic frames
"""

import numpy as np
import torch

class IntelligentLimbPredictor:
    """Predicts limb positions when MediaPipe detection fails or has low confidence"""
    
    def __init__(self):
        self.limb_history = []  # Rolling window of recent limb positions
        self.max_history = 3
        self.confidence_threshold = 0.3
        
    def predict_missing_limbs(self, landmarks, confidences):
        """
        Predict missing limb positions based on:
        1. Body mechanics (shoulder-elbow-wrist relationships)
        2. Recent history (short-term prediction)
        3. Symmetry (left-right limb mirroring)
        
        Args:
            landmarks: MediaPipe landmarks array (33, 3)
            confidences: Confidence scores for each landmark
            
        Returns:
            corrected_landmarks: Landmarks with predicted missing limbs
            correction_mask: Boolean mask showing which joints were predicted
        """
        
        if landmarks is None or len(landmarks) < 33:
            return landmarks, np.zeros(33, dtype=bool)
            
        corrected_landmarks = landmarks.copy()
        correction_mask = np.zeros(33, dtype=bool)
        
        # Key limb joint indices (MediaPipe)
        limb_chains = {
            'left_arm': [11, 13, 15],   # shoulder, elbow, wrist
            'right_arm': [12, 14, 16],  # shoulder, elbow, wrist  
            'left_leg': [23, 25, 27],   # hip, knee, ankle
            'right_leg': [24, 26, 28]   # hip, knee, ankle
        }
        
        for limb_name, joint_indices in limb_chains.items():
            corrected, corrected_indices = self._predict_limb_chain(
                corrected_landmarks, confidences, joint_indices, limb_name
            )
            corrected_landmarks = corrected
            correction_mask[corrected_indices] = True
            
        # Update history for future predictions
        self._update_history(corrected_landmarks)
        
        return corrected_landmarks, correction_mask
    
    def _predict_limb_chain(self, landmarks, confidences, joint_indices, limb_name):
        """Predict positions for a specific limb chain (e.g., shoulder->elbow->wrist)"""
        
        corrected_landmarks = landmarks.copy()
        corrected_indices = []
        
        # Check which joints in the chain have low confidence
        low_confidence_joints = []
        for i, joint_idx in enumerate(joint_indices):
            if confidences[joint_idx] < self.confidence_threshold:
                low_confidence_joints.append((i, joint_idx))
        
        if not low_confidence_joints:
            return corrected_landmarks, corrected_indices
            
        # Strategy 1: Interpolation for middle joints (elbow, knee)
        for i, joint_idx in low_confidence_joints:
            if i == 1 and len(joint_indices) == 3:  # Middle joint (elbow/knee)
                # Interpolate between start and end joints
                start_joint = landmarks[joint_indices[0]]  # shoulder/hip
                end_joint = landmarks[joint_indices[2]]    # wrist/ankle
                
                if confidences[joint_indices[0]] > self.confidence_threshold and \
                   confidences[joint_indices[2]] > self.confidence_threshold:
                    # Simple interpolation
                    predicted_pos = (start_joint + end_joint) / 2.0
                    # Add natural bend (slightly away from straight line)
                    bend_direction = self._get_natural_bend_direction(limb_name)
                    predicted_pos += bend_direction * 0.05
                    
                    corrected_landmarks[joint_idx] = predicted_pos
                    corrected_indices.append(joint_idx)
        
        # Strategy 2: History-based prediction
        if len(self.limb_history) >= 2:
            for i, joint_idx in low_confidence_joints:
                if joint_idx not in corrected_indices:
                    predicted_pos = self._predict_from_history(joint_idx)
                    if predicted_pos is not None:
                        corrected_landmarks[joint_idx] = predicted_pos
                        corrected_indices.append(joint_idx)
        
        # Strategy 3: Symmetry-based prediction (left-right mirroring)
        mirror_mapping = {
            11: 12, 12: 11,  # shoulders
            13: 14, 14: 13,  # elbows  
            15: 16, 16: 15,  # wrists
            23: 24, 24: 23,  # hips
            25: 26, 26: 25,  # knees
            27: 28, 28: 27   # ankles
        }
        
        for i, joint_idx in low_confidence_joints:
            if joint_idx not in corrected_indices and joint_idx in mirror_mapping:
                mirror_idx = mirror_mapping[joint_idx]
                if confidences[mirror_idx] > self.confidence_threshold:
                    # Mirror the position (flip X coordinate)
                    mirror_pos = landmarks[mirror_idx].copy()
                    mirror_pos[0] *= -1  # Flip X
                    corrected_landmarks[joint_idx] = mirror_pos
                    corrected_indices.append(joint_idx)
        
        return corrected_landmarks, corrected_indices
    
    def _get_natural_bend_direction(self, limb_name):
        """Get natural bending direction for different limbs"""
        bend_directions = {
            'left_arm': np.array([0, 0, 0.1]),    # Bend forward slightly
            'right_arm': np.array([0, 0, 0.1]),   # Bend forward slightly
            'left_leg': np.array([0, 0, -0.1]),   # Bend backward (knee)
            'right_leg': np.array([0, 0, -0.1])   # Bend backward (knee)
        }
        return bend_directions.get(limb_name, np.array([0, 0, 0]))
    
    def _predict_from_history(self, joint_idx):
        """Predict joint position based on recent history (linear extrapolation)"""
        if len(self.limb_history) < 2:
            return None
            
        # Get recent positions for this joint
        recent_positions = []
        for frame_landmarks in self.limb_history:
            if len(frame_landmarks) > joint_idx:
                recent_positions.append(frame_landmarks[joint_idx])
        
        if len(recent_positions) < 2:
            return None
            
        # Simple linear extrapolation
        last_pos = recent_positions[-1]
        second_last_pos = recent_positions[-2]
        velocity = last_pos - second_last_pos
        
        # Predict next position with damping to prevent drift
        predicted_pos = last_pos + velocity * 0.5  # 50% damping
        
        return predicted_pos
    
    def _update_history(self, landmarks):
        """Update rolling history of limb positions"""
        self.limb_history.append(landmarks.copy())
        
        # Keep only recent history
        if len(self.limb_history) > self.max_history:
            self.limb_history.pop(0)
    
    def reset_history(self):
        """Reset history (call between different videos)"""
        self.limb_history = []

# Usage example:
if __name__ == "__main__":
    predictor = IntelligentLimbPredictor()
    
    # Example usage
    landmarks = np.random.random((33, 3))
    confidences = np.random.random(33)
    confidences[13] = 0.1  # Low confidence elbow
    
    corrected, mask = predictor.predict_missing_limbs(landmarks, confidences)
    print(f"Corrected {np.sum(mask)} joints: {np.where(mask)[0]}")
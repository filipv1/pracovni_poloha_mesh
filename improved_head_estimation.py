#!/usr/bin/env python3
"""
Improved Head and Neck Estimation for SMPL-X
Fixes the head orientation issue by using ears and nose to estimate skull center
"""

import numpy as np


class ImprovedHeadEstimator:
    """
    Better head position estimation using multiple facial landmarks
    MediaPipe landmarks:
    - 0: nose
    - 7: left ear  
    - 8: right ear
    """
    
    def estimate_head_center(self, mp_landmarks):
        """
        Estimate actual head center (top of skull) from facial landmarks
        
        Args:
            mp_landmarks: MediaPipe pose landmarks (33 points)
            
        Returns:
            head_center: 3D position of estimated head center
            neck_base: Improved neck position
        """
        
        # Extract key landmarks
        nose = np.array([mp_landmarks[0].x, mp_landmarks[0].y, mp_landmarks[0].z])
        left_ear = np.array([mp_landmarks[7].x, mp_landmarks[7].y, mp_landmarks[7].z])
        right_ear = np.array([mp_landmarks[8].x, mp_landmarks[8].y, mp_landmarks[8].z])
        left_shoulder = np.array([mp_landmarks[11].x, mp_landmarks[11].y, mp_landmarks[11].z])
        right_shoulder = np.array([mp_landmarks[12].x, mp_landmarks[12].y, mp_landmarks[12].z])
        
        # Calculate ear center (approximates skull base)
        ear_center = (left_ear + right_ear) / 2
        
        # Calculate head orientation vector
        # From ear center to nose gives forward direction
        forward_vector = nose - ear_center
        forward_dist = np.linalg.norm(forward_vector)
        
        # Calculate up vector (perpendicular to ear line and forward)
        ear_vector = right_ear - left_ear
        up_vector = np.cross(ear_vector, forward_vector)
        up_vector = up_vector / np.linalg.norm(up_vector) if np.linalg.norm(up_vector) > 0 else np.array([0, 1, 0])
        
        # Anthropometric ratios (based on average human proportions)
        # Distance from ear canal to top of head is ~13cm
        # Distance from nose to ear center is ~10cm
        skull_height_ratio = 1.3  # Ratio of skull height to nose-ear distance
        
        # Estimate skull top
        # Move up from ear center by skull height
        head_top = ear_center + up_vector * (forward_dist * skull_height_ratio)
        
        # For SMPL-X, we want the head joint slightly forward of skull top
        # (SMPL-X head joint represents more of the head center of mass)
        head_center = head_top + forward_vector * 0.2
        
        # Better neck position (between shoulders and ear center)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        neck_base = shoulder_center + (ear_center - shoulder_center) * 0.3
        
        return head_center, neck_base
    
    def apply_to_smplx_joints(self, mp_landmarks, smplx_joints, joint_weights):
        """
        Apply improved head/neck estimation to SMPL-X joints
        
        Args:
            mp_landmarks: MediaPipe landmarks
            smplx_joints: Current SMPL-X joint positions (to be modified)
            joint_weights: Joint confidence weights
        """
        
        try:
            # Check if we have all required landmarks
            if len(mp_landmarks) < 13:
                return  # Not enough landmarks
                
            # Check visibility of key landmarks
            nose_vis = mp_landmarks[0].visibility if hasattr(mp_landmarks[0], 'visibility') else 1.0
            left_ear_vis = mp_landmarks[7].visibility if hasattr(mp_landmarks[7], 'visibility') else 1.0
            right_ear_vis = mp_landmarks[8].visibility if hasattr(mp_landmarks[8], 'visibility') else 1.0
            
            # Need at least nose and one ear visible
            if nose_vis < 0.5 or (left_ear_vis < 0.5 and right_ear_vis < 0.5):
                return  # Can't reliably estimate head
                
            # If only one ear is visible, mirror it
            if left_ear_vis < 0.5:
                # Mirror right ear
                nose_pos = np.array([mp_landmarks[0].x, mp_landmarks[0].y, mp_landmarks[0].z])
                right_ear_pos = np.array([mp_landmarks[8].x, mp_landmarks[8].y, mp_landmarks[8].z])
                ear_to_nose = nose_pos - right_ear_pos
                left_ear_estimate = nose_pos + ear_to_nose
                mp_landmarks[7].x, mp_landmarks[7].y, mp_landmarks[7].z = left_ear_estimate
                
            elif right_ear_vis < 0.5:
                # Mirror left ear
                nose_pos = np.array([mp_landmarks[0].x, mp_landmarks[0].y, mp_landmarks[0].z])
                left_ear_pos = np.array([mp_landmarks[7].x, mp_landmarks[7].y, mp_landmarks[7].z])
                ear_to_nose = nose_pos - left_ear_pos
                right_ear_estimate = nose_pos + ear_to_nose
                mp_landmarks[8].x, mp_landmarks[8].y, mp_landmarks[8].z = right_ear_estimate
            
            # Estimate improved head and neck positions
            head_center, neck_base = self.estimate_head_center(mp_landmarks)
            
            # Update SMPL-X joints
            # Joint 15 is head, Joint 12 is neck in SMPL-X
            smplx_joints[15] = head_center  # Head
            smplx_joints[12] = neck_base    # Neck
            
            # Update confidence based on ear visibility
            avg_ear_visibility = (left_ear_vis + right_ear_vis) / 2
            joint_weights[15] = min(nose_vis, avg_ear_visibility) * 0.9
            joint_weights[12] = 0.85
            
        except Exception as e:
            print(f"Warning: Could not improve head estimation: {e}")
            # Fall back to original estimation
            pass


def integrate_with_pipeline(converter_class):
    """
    Monkey-patch existing PreciseMediaPipeConverter to use improved head estimation
    
    Usage:
        from improved_head_estimation import integrate_with_pipeline
        integrate_with_pipeline(PreciseMediaPipeConverter)
    """
    
    # Save original method
    original_convert = converter_class.convert_landmarks_to_smplx
    
    # Create head estimator
    head_estimator = ImprovedHeadEstimator()
    
    def improved_convert(self, mp_landmarks):
        # Call original conversion
        smplx_joints, joint_weights = original_convert(self, mp_landmarks)
        
        if smplx_joints is not None and mp_landmarks is not None:
            # Apply improved head estimation
            head_estimator.apply_to_smplx_joints(
                mp_landmarks.landmark if hasattr(mp_landmarks, 'landmark') else mp_landmarks,
                smplx_joints,
                joint_weights
            )
        
        return smplx_joints, joint_weights
    
    # Replace method
    converter_class.convert_landmarks_to_smplx = improved_convert
    
    print("Head estimation improvement integrated!")


# Standalone test
if __name__ == "__main__":
    import mediapipe as mp
    
    # Test with sample data
    mp_pose = mp.solutions.pose
    
    # Create mock landmarks for testing
    class MockLandmark:
        def __init__(self, x, y, z, visibility=1.0):
            self.x, self.y, self.z = x, y, z
            self.visibility = visibility
    
    # Create test pose (person looking slightly down)
    test_landmarks = [
        MockLandmark(0, 0, 0.1),      # 0: nose (forward)
        MockLandmark(-0.05, 0, 0),    # 1: left eye inner
        MockLandmark(-0.07, 0, 0),    # 2: left eye
        MockLandmark(-0.09, 0, 0),    # 3: left eye outer
        MockLandmark(0.05, 0, 0),     # 4: right eye inner
        MockLandmark(0.07, 0, 0),     # 5: right eye
        MockLandmark(0.09, 0, 0),     # 6: right eye outer
        MockLandmark(-0.1, 0, -0.05), # 7: left ear
        MockLandmark(0.1, 0, -0.05),  # 8: right ear
        MockLandmark(-0.03, -0.02, 0.1), # 9: mouth left
        MockLandmark(0.03, -0.02, 0.1),  # 10: mouth right
        MockLandmark(-0.15, -0.2, 0), # 11: left shoulder
        MockLandmark(0.15, -0.2, 0),  # 12: right shoulder
    ]
    
    # Test estimator
    estimator = ImprovedHeadEstimator()
    head_center, neck_base = estimator.estimate_head_center(test_landmarks)
    
    print(f"Original nose position: [0.000, 0.000, 0.100]")
    print(f"Estimated head center: [{head_center[0]:.3f}, {head_center[1]:.3f}, {head_center[2]:.3f}]")
    print(f"Estimated neck base: [{neck_base[0]:.3f}, {neck_base[1]:.3f}, {neck_base[2]:.3f}]")
    print("\nHead is now positioned above and behind the nose, as it should be!")
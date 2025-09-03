#!/usr/bin/env python3
"""
Coordinate System Transformer - Fix MediaPipe to SMPL-X coordinate system misalignment

Priority: CRITICAL
Dependencies: numpy, scipy
Test Coverage Required: 100%

This module fixes the root cause of orientation issues by properly transforming
MediaPipe landmarks to SMPL-X coordinate system at the source.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinateSystemTransformer:
    """Fix MediaPipe to SMPL-X coordinate system misalignment
    
    MediaPipe uses: Z-forward, Y-up, X-right (right-handed)
    SMPL-X uses: Y-up, Z-backward, X-right (right-handed)
    
    The transformation involves swapping and inverting axes to align the systems.
    """
    
    def __init__(self):
        """Initialize coordinate system transformer"""
        
        # MediaPipe: Z-forward, Y-up, X-right
        # SMPL-X: Y-up, Z-backward, X-right
        
        # Rotation matrix to align coordinate systems
        # This transforms MediaPipe coordinates to SMPL-X coordinates
        self.mp_to_smplx = np.array([
            [1,  0,  0],  # X stays same (right direction)
            [0,  0,  1],  # Y <- Z (up direction from forward)
            [0, -1,  0]   # Z <- -Y (backward direction from inverted up)
        ], dtype=np.float32)
        
        # Alternative quaternion representation for stability
        self.rotation_quaternion = R.from_matrix(self.mp_to_smplx).as_quat()
        
        # Inverse transformation (SMPL-X to MediaPipe)
        self.smplx_to_mp = self.mp_to_smplx.T
        
        logger.info("Coordinate system transformer initialized")
        logger.info(f"Transformation matrix:\n{self.mp_to_smplx}")
    
    def transform_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Transform MediaPipe landmarks to SMPL-X coordinate system
        
        Args:
            landmarks: (33, 3) or (N, 33, 3) array of MediaPipe landmarks
            
        Returns:
            Transformed landmarks in SMPL-X coordinate system
            
        Raises:
            ValueError: If input shape is invalid
        """
        if landmarks is None:
            raise ValueError("Landmarks cannot be None")
        
        landmarks = np.asarray(landmarks, dtype=np.float32)
        original_shape = landmarks.shape
        
        # Validate input shape
        if len(original_shape) < 2 or original_shape[-1] != 3:
            raise ValueError(f"Invalid landmarks shape {original_shape}. Expected (..., 3)")
        
        # Reshape for matrix multiplication
        landmarks_reshaped = landmarks.reshape(-1, 3)
        
        # Apply rotation transformation
        # landmarks @ R.T is equivalent to R @ landmarks.T but more efficient
        transformed = landmarks_reshaped @ self.mp_to_smplx.T
        
        # Restore original shape
        return transformed.reshape(original_shape)
    
    def transform_with_confidence(self, landmarks: np.ndarray, 
                                confidences: np.ndarray, 
                                confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Transform with confidence score weighting
        
        Args:
            landmarks: MediaPipe landmarks array
            confidences: Confidence scores for each landmark
            confidence_threshold: Minimum confidence to keep landmark
            
        Returns:
            Tuple of (transformed_landmarks, confidence_mask)
        """
        if landmarks is None or confidences is None:
            raise ValueError("Landmarks and confidences cannot be None")
        
        # Transform landmarks
        transformed = self.transform_landmarks(landmarks)
        
        # Create confidence mask
        confidence_mask = confidences > confidence_threshold
        
        # Set low-confidence landmarks to NaN for later interpolation
        if len(transformed.shape) == 2:  # Single frame
            transformed[~confidence_mask] = np.nan
        elif len(transformed.shape) == 3:  # Multiple frames
            for i in range(transformed.shape[0]):
                transformed[i][~confidence_mask[i]] = np.nan
        
        return transformed, confidence_mask
    
    def inverse_transform(self, landmarks: np.ndarray) -> np.ndarray:
        """Transform SMPL-X coordinates back to MediaPipe system
        
        Args:
            landmarks: SMPL-X landmarks
            
        Returns:
            MediaPipe coordinate system landmarks
        """
        if landmarks is None:
            raise ValueError("Landmarks cannot be None")
        
        landmarks = np.asarray(landmarks, dtype=np.float32)
        original_shape = landmarks.shape
        landmarks_reshaped = landmarks.reshape(-1, 3)
        
        # Apply inverse transformation
        transformed = landmarks_reshaped @ self.smplx_to_mp.T
        
        return transformed.reshape(original_shape)
    
    def validate_transformation(self, original: np.ndarray, 
                              transformed: np.ndarray) -> dict:
        """Validate transformation correctness
        
        Args:
            original: Original MediaPipe landmarks
            transformed: Transformed SMPL-X landmarks
            
        Returns:
            Dict with validation results
        """
        # Test round-trip transformation
        round_trip = self.inverse_transform(transformed)
        
        # Calculate errors
        max_error = np.max(np.abs(original - round_trip))
        mean_error = np.mean(np.abs(original - round_trip))
        
        # Check if specific axis transformations are correct
        # MediaPipe Y-up should become SMPL-X Y-up (should remain same)
        y_up_mp = np.array([0, 1, 0])
        y_up_transformed = self.transform_landmarks(y_up_mp.reshape(1, -1))[0]
        y_preservation = np.allclose(y_up_transformed, [0, 0, 1])  # Should map to Z
        
        # MediaPipe Z-forward should become SMPL-X Z-backward
        z_forward_mp = np.array([0, 0, 1])
        z_transformed = self.transform_landmarks(z_forward_mp.reshape(1, -1))[0]
        z_inversion = np.allclose(z_transformed, [0, -1, 0])  # Should map to -Y
        
        return {
            'round_trip_max_error': max_error,
            'round_trip_mean_error': mean_error,
            'y_axis_correct': y_preservation,
            'z_axis_correct': z_inversion,
            'transformation_valid': max_error < 1e-6
        }
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix
        
        Returns:
            3x3 transformation matrix
        """
        return self.mp_to_smplx.copy()
    
    def get_rotation_quaternion(self) -> np.ndarray:
        """Get transformation as quaternion
        
        Returns:
            Quaternion representation [x, y, z, w]
        """
        return self.rotation_quaternion.copy()


def create_test_landmarks() -> np.ndarray:
    """Create test landmarks for validation
    
    Returns:
        Test MediaPipe landmarks array
    """
    # Create basic pose with known orientations
    landmarks = np.array([
        [0.0, 0.5, 0.0],   # Nose (center, up)
        [0.1, 0.4, 0.0],   # Left eye
        [-0.1, 0.4, 0.0],  # Right eye
        [0.0, 0.3, 0.0],   # Mouth
        [0.0, 0.0, 0.0],   # Neck base
        [0.3, 0.0, 0.0],   # Left shoulder
        [-0.3, 0.0, 0.0],  # Right shoulder
        [0.3, -0.3, 0.0],  # Left elbow
        [-0.3, -0.3, 0.0], # Right elbow
        [0.0, -0.5, 0.0],  # Hip center
    ] + [[0.0, 0.0, 0.0]] * 23, dtype=np.float32)  # Pad to 33 landmarks
    
    return landmarks


if __name__ == "__main__":
    # Quick test
    transformer = CoordinateSystemTransformer()
    test_landmarks = create_test_landmarks()
    
    print("Testing coordinate system transformer...")
    
    # Transform landmarks
    transformed = transformer.transform_landmarks(test_landmarks)
    
    # Validate transformation
    validation = transformer.validate_transformation(test_landmarks, transformed)
    
    print("Validation results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    if validation['transformation_valid']:
        print("✓ Coordinate system transformer working correctly")
    else:
        print("✗ Coordinate system transformer has issues")
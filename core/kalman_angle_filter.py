#!/usr/bin/env python3
"""
Kalman Angle Filter - Advanced temporal filtering for angle measurements

Priority: HIGH
Dependencies: numpy, scipy
Test Coverage Required: 100%

This module implements a Kalman filter specifically optimized for angle measurements
with proper handling of circular statistics and temporal smoothing.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy.stats import vonmises
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanAngleFilter:
    """Kalman filter optimized for angle measurements
    
    Features:
    - Circular statistics for proper angle handling
    - Adaptive noise estimation
    - Multi-angle tracking with cross-correlation
    - Outlier detection and rejection
    - Confidence-weighted updates
    """
    
    def __init__(self, 
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 initial_uncertainty: float = 10.0,
                 angle_type: str = 'degrees',
                 outlier_threshold: float = 3.0):
        """Initialize Kalman angle filter
        
        Args:
            process_noise: Process noise variance (angle change per frame)
            measurement_noise: Measurement noise variance 
            initial_uncertainty: Initial state uncertainty
            angle_type: 'degrees' or 'radians'
            outlier_threshold: Standard deviations for outlier detection
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_uncertainty = initial_uncertainty
        self.angle_type = angle_type
        self.outlier_threshold = outlier_threshold
        
        # Convert angle bounds based on type
        if angle_type == 'degrees':
            self.angle_bound = 180.0
            self.full_circle = 360.0
        else:
            self.angle_bound = np.pi
            self.full_circle = 2 * np.pi
        
        # Filter state
        self.is_initialized = False
        self.state = None  # [angle, angular_velocity]
        self.covariance = None  # State covariance matrix
        
        # Statistics and adaptation
        self.measurement_history = []
        self.innovation_history = []
        self.confidence_history = []
        
        # Adaptive parameters
        self.adaptive_noise = True
        self.noise_adaptation_window = 10
        
        logger.info(f"KalmanAngleFilter initialized for {angle_type} with process_noise={process_noise}")
    
    def update(self, measurement: float, 
               confidence: Optional[float] = None,
               dt: float = 1.0) -> Dict:
        """Update filter with new angle measurement
        
        Args:
            measurement: New angle measurement
            confidence: Measurement confidence (0-1)
            dt: Time step since last update
            
        Returns:
            Dict with filtered angle and filter statistics
        """
        # Normalize angle to [-bound, bound] range
        measurement = self._normalize_angle(measurement)
        
        if not self.is_initialized:
            return self._initialize_filter(measurement, confidence)
        
        # Prediction step
        predicted_state, predicted_covariance = self._predict(dt)
        
        # Outlier detection
        innovation = self._calculate_innovation(measurement, predicted_state[0])
        is_outlier = self._detect_outlier(innovation, predicted_covariance)
        
        if is_outlier and confidence is not None and confidence < 0.3:
            # Skip update for low-confidence outliers
            result = {
                'filtered_angle': predicted_state[0],
                'angular_velocity': predicted_state[1],
                'innovation': innovation,
                'confidence': confidence or 1.0,
                'outlier_detected': True,
                'measurement_used': False,
                'filter_uncertainty': predicted_covariance[0, 0]
            }
            return result
        
        # Update step
        filtered_state, updated_covariance = self._update_step(
            measurement, predicted_state, predicted_covariance, confidence
        )
        
        # Store filter state
        self.state = filtered_state
        self.covariance = updated_covariance
        
        # Update history for adaptation
        self._update_history(measurement, innovation, confidence)
        
        # Adaptive noise estimation
        if self.adaptive_noise:
            self._adapt_noise_parameters()
        
        result = {
            'filtered_angle': filtered_state[0],
            'angular_velocity': filtered_state[1],
            'innovation': innovation,
            'confidence': confidence or 1.0,
            'outlier_detected': is_outlier,
            'measurement_used': True,
            'filter_uncertainty': updated_covariance[0, 0],
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise
        }
        
        return result
    
    def _initialize_filter(self, measurement: float, confidence: Optional[float]) -> Dict:
        """Initialize filter with first measurement"""
        self.state = np.array([measurement, 0.0])  # [angle, angular_velocity]
        
        # Initial covariance matrix
        initial_var = self.initial_uncertainty ** 2
        self.covariance = np.array([
            [initial_var, 0],
            [0, initial_var]
        ])
        
        self.is_initialized = True
        
        result = {
            'filtered_angle': measurement,
            'angular_velocity': 0.0,
            'innovation': 0.0,
            'confidence': confidence or 1.0,
            'outlier_detected': False,
            'measurement_used': True,
            'filter_uncertainty': initial_var,
            'initialized': True
        }
        
        logger.info(f"Filter initialized with angle: {measurement:.2f} {self.angle_type}")
        return result
    
    def _predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step of Kalman filter"""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Process noise matrix
        Q = np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ]) * self.process_noise
        
        # Predict state
        predicted_state = F @ self.state
        
        # Handle angle wrapping for predicted angle
        predicted_state[0] = self._normalize_angle(predicted_state[0])
        
        # Predict covariance
        predicted_covariance = F @ self.covariance @ F.T + Q
        
        return predicted_state, predicted_covariance
    
    def _update_step(self, measurement: float, 
                    predicted_state: np.ndarray,
                    predicted_covariance: np.ndarray,
                    confidence: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Update step of Kalman filter"""
        # Measurement matrix (we observe angle only)
        H = np.array([[1, 0]])
        
        # Measurement noise (adapt based on confidence)
        R = self._get_measurement_noise(confidence)
        
        # Innovation (measurement residual)
        innovation = self._calculate_innovation(measurement, predicted_state[0])
        
        # Innovation covariance
        S = H @ predicted_covariance @ H.T + R
        
        # Kalman gain
        K = predicted_covariance @ H.T / S
        
        # Update state
        updated_state = predicted_state + K.flatten() * innovation
        
        # Handle angle wrapping for updated angle
        updated_state[0] = self._normalize_angle(updated_state[0])
        
        # Update covariance
        I = np.eye(2)
        updated_covariance = (I - K @ H) @ predicted_covariance
        
        return updated_state, updated_covariance
    
    def _calculate_innovation(self, measurement: float, predicted_angle: float) -> float:
        """Calculate innovation with proper angle wrapping"""
        diff = measurement - predicted_angle
        
        # Handle angle wrapping
        if diff > self.angle_bound:
            diff -= self.full_circle
        elif diff < -self.angle_bound:
            diff += self.full_circle
            
        return diff
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to (-bound, bound] range"""
        while angle > self.angle_bound:
            angle -= self.full_circle
        while angle <= -self.angle_bound:
            angle += self.full_circle
        return angle
    
    def _get_measurement_noise(self, confidence: Optional[float]) -> float:
        """Get measurement noise based on confidence"""
        if confidence is None:
            return self.measurement_noise
        
        # Lower confidence = higher noise
        confidence_factor = max(0.1, confidence)  # Minimum confidence factor
        return self.measurement_noise / confidence_factor
    
    def _detect_outlier(self, innovation: float, predicted_covariance: np.ndarray) -> bool:
        """Detect outlier based on innovation magnitude"""
        if len(self.innovation_history) < 3:
            return False
        
        # Expected innovation variance
        innovation_variance = predicted_covariance[0, 0] + self.measurement_noise
        
        # Check if innovation is unusually large
        threshold = self.outlier_threshold * np.sqrt(innovation_variance)
        
        return abs(innovation) > threshold
    
    def _update_history(self, measurement: float, innovation: float, confidence: Optional[float]):
        """Update measurement and innovation history"""
        self.measurement_history.append(measurement)
        self.innovation_history.append(innovation)
        self.confidence_history.append(confidence or 1.0)
        
        # Maintain fixed window size
        max_history = max(50, self.noise_adaptation_window * 2)
        if len(self.measurement_history) > max_history:
            self.measurement_history.pop(0)
            self.innovation_history.pop(0)
            self.confidence_history.pop(0)
    
    def _adapt_noise_parameters(self):
        """Adapt noise parameters based on recent performance"""
        if len(self.innovation_history) < self.noise_adaptation_window:
            return
        
        # Get recent innovations
        recent_innovations = self.innovation_history[-self.noise_adaptation_window:]
        recent_confidences = self.confidence_history[-self.noise_adaptation_window:]
        
        # Estimate measurement noise from innovation variance
        innovation_var = np.var(recent_innovations)
        
        # Adapt measurement noise
        avg_confidence = np.mean(recent_confidences)
        if avg_confidence > 0.7 and innovation_var > 0:
            # High confidence data - trust the innovation variance estimate
            new_measurement_noise = 0.8 * self.measurement_noise + 0.2 * innovation_var
            self.measurement_noise = np.clip(new_measurement_noise, 0.01, 1.0)
        
        # Adapt process noise based on innovation trends
        if len(recent_innovations) >= 5:
            # If innovations are consistently large, increase process noise
            innovation_trend = np.mean(np.abs(recent_innovations[-5:]))
            if innovation_trend > 2 * np.sqrt(self.measurement_noise):
                self.process_noise = min(self.process_noise * 1.1, 0.1)
            elif innovation_trend < 0.5 * np.sqrt(self.measurement_noise):
                self.process_noise = max(self.process_noise * 0.95, 0.001)
    
    def predict_next(self, dt: float = 1.0) -> Dict:
        """Predict next angle value without updating filter"""
        if not self.is_initialized:
            return {'predicted_angle': None, 'uncertainty': None}
        
        predicted_state, predicted_covariance = self._predict(dt)
        
        return {
            'predicted_angle': predicted_state[0],
            'predicted_velocity': predicted_state[1],
            'uncertainty': predicted_covariance[0, 0]
        }
    
    def get_state(self) -> Optional[Dict]:
        """Get current filter state"""
        if not self.is_initialized:
            return None
        
        return {
            'angle': self.state[0],
            'angular_velocity': self.state[1],
            'uncertainty': self.covariance[0, 0],
            'velocity_uncertainty': self.covariance[1, 1],
            'measurements_processed': len(self.measurement_history),
            'current_process_noise': self.process_noise,
            'current_measurement_noise': self.measurement_noise
        }
    
    def reset(self):
        """Reset filter to uninitialized state"""
        self.is_initialized = False
        self.state = None
        self.covariance = None
        self.measurement_history.clear()
        self.innovation_history.clear()
        self.confidence_history.clear()
        
        # Reset adaptive parameters
        self.process_noise = 0.01
        self.measurement_noise = 0.1
        
        logger.info("Filter reset")
    
    def get_statistics(self) -> Dict:
        """Get filter performance statistics"""
        if len(self.innovation_history) == 0:
            return {}
        
        innovations = np.array(self.innovation_history)
        confidences = np.array(self.confidence_history)
        
        return {
            'mean_innovation': np.mean(innovations),
            'innovation_std': np.std(innovations),
            'mean_abs_innovation': np.mean(np.abs(innovations)),
            'mean_confidence': np.mean(confidences),
            'outlier_rate': np.sum(np.abs(innovations) > self.outlier_threshold * np.std(innovations)) / len(innovations),
            'measurements_processed': len(self.measurement_history),
            'adaptive_process_noise': self.process_noise,
            'adaptive_measurement_noise': self.measurement_noise
        }


class MultiAngleKalmanFilter:
    """Multi-angle Kalman filter for tracking multiple angles simultaneously"""
    
    def __init__(self, 
                 angle_names: List[str],
                 angle_type: str = 'degrees',
                 **filter_kwargs):
        """Initialize multi-angle filter
        
        Args:
            angle_names: List of angle names to track
            angle_type: 'degrees' or 'radians'
            **filter_kwargs: Keyword arguments passed to individual filters
        """
        self.angle_names = angle_names
        self.filters = {}
        
        # Create individual filters
        for name in angle_names:
            self.filters[name] = KalmanAngleFilter(
                angle_type=angle_type,
                **filter_kwargs
            )
        
        logger.info(f"MultiAngleKalmanFilter initialized for {len(angle_names)} angles: {angle_names}")
    
    def update(self, angle_measurements: Dict[str, float],
               confidences: Optional[Dict[str, float]] = None,
               dt: float = 1.0) -> Dict[str, Dict]:
        """Update all filters with angle measurements
        
        Args:
            angle_measurements: Dict mapping angle names to measurements
            confidences: Optional dict mapping angle names to confidences
            dt: Time step since last update
            
        Returns:
            Dict mapping angle names to filter results
        """
        results = {}
        
        for name in self.angle_names:
            if name in angle_measurements:
                confidence = confidences.get(name) if confidences else None
                result = self.filters[name].update(
                    angle_measurements[name],
                    confidence=confidence,
                    dt=dt
                )
                results[name] = result
            else:
                # No measurement available - use prediction
                prediction = self.filters[name].predict_next(dt)
                results[name] = {
                    'filtered_angle': prediction.get('predicted_angle'),
                    'angular_velocity': prediction.get('predicted_velocity'),
                    'innovation': 0.0,
                    'confidence': 0.0,
                    'outlier_detected': False,
                    'measurement_used': False,
                    'filter_uncertainty': prediction.get('uncertainty')
                }
        
        return results
    
    def get_filtered_angles(self) -> Dict[str, float]:
        """Get current filtered angles for all tracked angles"""
        angles = {}
        for name, filter_obj in self.filters.items():
            state = filter_obj.get_state()
            if state:
                angles[name] = state['angle']
            else:
                angles[name] = None
        return angles
    
    def get_angular_velocities(self) -> Dict[str, float]:
        """Get current angular velocities for all tracked angles"""
        velocities = {}
        for name, filter_obj in self.filters.items():
            state = filter_obj.get_state()
            if state:
                velocities[name] = state['angular_velocity']
            else:
                velocities[name] = None
        return velocities
    
    def reset_all(self):
        """Reset all filters"""
        for filter_obj in self.filters.values():
            filter_obj.reset()
    
    def reset_filter(self, angle_name: str):
        """Reset specific filter"""
        if angle_name in self.filters:
            self.filters[angle_name].reset()
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all filters"""
        stats = {}
        for name, filter_obj in self.filters.items():
            stats[name] = filter_obj.get_statistics()
        return stats


def create_posture_angle_filter() -> MultiAngleKalmanFilter:
    """Create multi-angle filter optimized for posture analysis"""
    angle_names = [
        'trunk_sagittal',      # Forward/backward trunk bend
        'trunk_lateral',       # Left/right trunk bend  
        'neck_flexion',        # Neck forward/backward
        'left_shoulder_flexion',
        'right_shoulder_flexion',
        'left_shoulder_abduction',
        'right_shoulder_abduction',
        'left_elbow_flexion',
        'right_elbow_flexion'
    ]
    
    return MultiAngleKalmanFilter(
        angle_names=angle_names,
        angle_type='degrees',
        process_noise=0.005,      # Low process noise for stable posture
        measurement_noise=0.5,    # Moderate measurement noise
        initial_uncertainty=5.0,  # Reasonable initial uncertainty
        outlier_threshold=2.5     # Moderate outlier sensitivity
    )


if __name__ == "__main__":
    # Quick test
    filter_obj = KalmanAngleFilter(angle_type='degrees')
    
    print("Testing Kalman angle filter...")
    
    # Simulate noisy angle measurements
    true_angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # True angle progression
    noise_levels = [2, 1, 3, 1, 2, 1, 4, 1, 2, 1]       # Measurement noise
    
    filtered_angles = []
    
    for i, (true_angle, noise) in enumerate(zip(true_angles, noise_levels)):
        # Add noise to measurement
        noisy_measurement = true_angle + np.random.normal(0, noise)
        
        # Confidence based on noise level
        confidence = 1.0 / (1.0 + noise)
        
        # Update filter
        result = filter_obj.update(noisy_measurement, confidence=confidence)
        
        filtered_angles.append(result['filtered_angle'])
        
        print(f"Frame {i:2d}: True={true_angle:6.1f}, Noisy={noisy_measurement:6.1f}, "
              f"Filtered={result['filtered_angle']:6.1f}, "
              f"Velocity={result['angular_velocity']:6.2f}, "
              f"Outlier={result['outlier_detected']}")
    
    # Test multi-angle filter
    multi_filter = create_posture_angle_filter()
    
    test_measurements = {
        'trunk_sagittal': 15.0,
        'trunk_lateral': -5.0,
        'neck_flexion': 25.0
    }
    
    test_confidences = {
        'trunk_sagittal': 0.9,
        'trunk_lateral': 0.7,
        'neck_flexion': 0.8
    }
    
    results = multi_filter.update(test_measurements, test_confidences)
    
    print("\nMulti-angle filter results:")
    for angle_name, result in results.items():
        print(f"  {angle_name}: {result['filtered_angle']:.1f}° "
              f"(confidence={result['confidence']:.1f})")
    
    # Get statistics
    stats = filter_obj.get_statistics()
    print(f"\nFilter statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("✓ Kalman angle filter test completed")
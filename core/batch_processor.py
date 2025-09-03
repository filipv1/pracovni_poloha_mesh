#!/usr/bin/env python3
"""
Batch Processor - High-performance batch processing for 3D pose analysis

Priority: HIGH
Dependencies: torch, numpy, enhanced_pipeline_integration
Test Coverage Required: 100%

This module implements optimized batch processing for maximum performance
with GPU acceleration and memory management.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import gc
import time
from dataclasses import dataclass

# Add core module to path
sys.path.append(str(Path(__file__).parent))

from enhanced_pipeline_integration import EnhancedMediaPipeConverter
from coordinate_system_fix import CoordinateSystemTransformer
from proactive_joint_validator import ProactiveJointValidator
from kalman_angle_filter import MultiAngleKalmanFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 32
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    num_workers: int = 4
    prefetch_factor: int = 2
    memory_limit: float = 0.8  # 80% of available memory
    optimization_level: str = 'balanced'  # 'speed', 'memory', 'balanced'
    enable_amp: bool = True  # Automatic Mixed Precision
    enable_compilation: bool = False  # torch.compile (requires PyTorch 2.0+)


class OptimizedCoordinateTransformer:
    """GPU-optimized coordinate system transformer"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Pre-compute transformation matrix on GPU
        mp_to_smplx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ], device=device, dtype=torch.float32)
        
        self.transform_matrix = mp_to_smplx
        
        logger.info(f"OptimizedCoordinateTransformer initialized on {device}")
    
    def transform_batch(self, landmarks_batch: torch.Tensor) -> torch.Tensor:
        """Transform batch of landmarks efficiently
        
        Args:
            landmarks_batch: (batch_size, 33, 3) tensor of landmarks
            
        Returns:
            Transformed landmarks tensor
        """
        # Efficient batch matrix multiplication
        # landmarks_batch @ transform_matrix.T
        return torch.matmul(landmarks_batch, self.transform_matrix.T)


class BatchValidationOptimizer:
    """Optimized batch validation for joint positions"""
    
    def __init__(self, device: torch.device, config: BatchConfig):
        self.device = device
        self.config = config
        
        # Pre-compute bone length constraints as tensors
        self.bone_constraints = self._prepare_bone_constraints()
        
        logger.info("BatchValidationOptimizer initialized")
    
    def _prepare_bone_constraints(self) -> Dict[str, torch.Tensor]:
        """Prepare bone length constraints as GPU tensors"""
        # Key bone pairs and their expected length ranges (normalized)
        constraints = {
            'upper_arms': {
                'pairs': [(11, 13), (12, 14)],  # shoulder to elbow
                'range': torch.tensor([0.25, 0.40], device=self.device)
            },
            'forearms': {
                'pairs': [(13, 15), (14, 16)],  # elbow to wrist  
                'range': torch.tensor([0.20, 0.35], device=self.device)
            },
            'thighs': {
                'pairs': [(23, 25), (24, 26)],  # hip to knee
                'range': torch.tensor([0.35, 0.55], device=self.device)
            },
            'shins': {
                'pairs': [(25, 27), (26, 28)],  # knee to ankle
                'range': torch.tensor([0.30, 0.50], device=self.device)
            }
        }
        
        return constraints
    
    def validate_batch(self, joints_batch: torch.Tensor) -> torch.Tensor:
        """Validate batch of joint positions
        
        Args:
            joints_batch: (batch_size, 33, 3) tensor of joint positions
            
        Returns:
            Validation mask (batch_size, 33) - True for valid joints
        """
        batch_size = joints_batch.shape[0]
        device = joints_batch.device
        
        # Initialize validity mask
        valid_mask = torch.ones((batch_size, 33), device=device, dtype=torch.bool)
        
        # Estimate body scale for each sample in batch
        body_scales = self._estimate_batch_body_scales(joints_batch)
        
        # Validate bone lengths for each constraint group
        for constraint_name, constraint_data in self.bone_constraints.items():
            pairs = constraint_data['pairs']
            length_range = constraint_data['range']
            
            for idx1, idx2 in pairs:
                # Calculate bone lengths for entire batch
                bone_vectors = joints_batch[:, idx2] - joints_batch[:, idx1]
                bone_lengths = torch.norm(bone_vectors, dim=1)
                
                # Scale expected ranges by body scale
                expected_min = length_range[0] * body_scales
                expected_max = length_range[1] * body_scales
                
                # Check validity
                length_valid = (bone_lengths >= expected_min) & (bone_lengths <= expected_max)
                
                # Update validity mask
                valid_mask[:, idx1] &= length_valid
                valid_mask[:, idx2] &= length_valid
        
        return valid_mask
    
    def _estimate_batch_body_scales(self, joints_batch: torch.Tensor) -> torch.Tensor:
        """Estimate body scale for batch using shoulder width"""
        # Calculate shoulder width for each sample
        left_shoulder = joints_batch[:, 11]  # index 11
        right_shoulder = joints_batch[:, 12]  # index 12
        
        shoulder_widths = torch.norm(right_shoulder - left_shoulder, dim=1)
        
        # Normalize by average shoulder width (0.35m)
        body_scales = torch.clamp(shoulder_widths / 0.35, min=0.5, max=2.0)
        
        return body_scales


class HighPerformanceBatchProcessor:
    """High-performance batch processor for 3D pose analysis"""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        
        # Determine optimal device
        self.device = self._select_optimal_device()
        
        # Initialize optimized components
        self.coord_transformer = OptimizedCoordinateTransformer(self.device)
        self.batch_validator = BatchValidationOptimizer(self.device, self.config)
        
        # Memory management
        self.memory_monitor = self._setup_memory_monitoring()
        
        # Performance tracking
        self.performance_stats = {
            'batches_processed': 0,
            'total_frames': 0,
            'processing_times': [],
            'throughput_history': [],
            'memory_usage': []
        }
        
        # Setup compilation if enabled and supported
        if self.config.enable_compilation:
            self._setup_compilation()
        
        logger.info(f"HighPerformanceBatchProcessor initialized on {self.device}")
        logger.info(f"Configuration: {self.config}")
    
    def _select_optimal_device(self) -> torch.device:
        """Select optimal device for processing"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                # Select GPU with most memory
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    best_gpu = 0
                    max_memory = 0
                    
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory
                        if total_memory > max_memory:
                            max_memory = total_memory
                            best_gpu = i
                    
                    device = torch.device(f'cuda:{best_gpu}')
                    logger.info(f"Selected GPU {best_gpu} with {max_memory/1e9:.1f}GB memory")
                    return device
            
            # Fall back to CPU
            logger.info("Using CPU (CUDA not available)")
            return torch.device('cpu')
        
        return torch.device(self.config.device)
    
    def _setup_memory_monitoring(self) -> Dict:
        """Setup memory monitoring"""
        monitor = {'enabled': True}
        
        if self.device.type == 'cuda':
            monitor['gpu_memory'] = True
            torch.cuda.empty_cache()
        else:
            monitor['gpu_memory'] = False
        
        return monitor
    
    def _setup_compilation(self):
        """Setup torch.compile if available"""
        try:
            if hasattr(torch, 'compile'):
                # Compile key operations for better performance
                self.coord_transformer.transform_batch = torch.compile(
                    self.coord_transformer.transform_batch,
                    mode='default'
                )
                logger.info("Torch compilation enabled")
            else:
                logger.warning("Torch compilation not available")
        except Exception as e:
            logger.warning(f"Failed to enable compilation: {e}")
    
    def process_landmarks_batch(self, landmarks_list: List[np.ndarray],
                               confidences_list: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Process batch of MediaPipe landmarks with maximum performance
        
        Args:
            landmarks_list: List of (33, 3) landmark arrays
            confidences_list: Optional list of confidence arrays
            
        Returns:
            List of processing results for each frame
        """
        start_time = time.time()
        
        if not landmarks_list:
            return []
        
        batch_size = len(landmarks_list)
        
        # Convert to tensors and move to GPU
        landmarks_batch = self._prepare_landmarks_batch(landmarks_list)
        confidences_batch = self._prepare_confidences_batch(confidences_list, batch_size)
        
        # Memory check before processing
        if not self._check_memory_availability(landmarks_batch):
            logger.warning("Insufficient memory, falling back to smaller batch")
            return self._process_in_smaller_batches(landmarks_list, confidences_list)
        
        with torch.inference_mode():  # Optimize inference
            # Step 1: Coordinate transformation (vectorized)
            transformed_batch = self.coord_transformer.transform_batch(landmarks_batch)
            
            # Step 2: Batch validation
            validity_mask = self.batch_validator.validate_batch(transformed_batch)
            
            # Step 3: Convert to SMPL-X format (batch operation)
            smplx_batch, weights_batch = self._convert_batch_to_smplx(
                transformed_batch, validity_mask, confidences_batch
            )
            
            # Step 4: Calculate angles (vectorized where possible)
            angles_batch = self._calculate_angles_batch(smplx_batch, weights_batch)
        
        # Convert back to CPU and individual results
        results = self._package_batch_results(
            smplx_batch, weights_batch, validity_mask, angles_batch
        )
        
        # Update performance statistics
        processing_time = time.time() - start_time
        self._update_performance_stats(batch_size, processing_time)
        
        return results
    
    def _prepare_landmarks_batch(self, landmarks_list: List[np.ndarray]) -> torch.Tensor:
        """Convert landmarks list to batch tensor"""
        # Stack landmarks into batch
        landmarks_array = np.stack([lm if lm is not None else np.zeros((33, 3)) 
                                   for lm in landmarks_list])
        
        # Convert to tensor and move to device
        return torch.from_numpy(landmarks_array).float().to(self.device)
    
    def _prepare_confidences_batch(self, confidences_list: Optional[List[np.ndarray]], 
                                 batch_size: int) -> torch.Tensor:
        """Convert confidences list to batch tensor"""
        if confidences_list is None:
            # Default confidences
            confidences_array = np.ones((batch_size, 33), dtype=np.float32) * 0.8
        else:
            confidences_array = np.stack([
                conf if conf is not None else np.ones(33) * 0.8
                for conf in confidences_list
            ])
        
        return torch.from_numpy(confidences_array).float().to(self.device)
    
    def _check_memory_availability(self, landmarks_batch: torch.Tensor) -> bool:
        """Check if sufficient memory available for batch processing"""
        if self.device.type == 'cuda':
            # Estimate memory requirement
            batch_size = landmarks_batch.shape[0]
            estimated_memory = batch_size * 33 * 3 * 4 * 10  # Conservative estimate
            
            free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
            
            return estimated_memory < free_memory * self.config.memory_limit
        
        return True  # Assume sufficient CPU memory
    
    def _process_in_smaller_batches(self, landmarks_list: List[np.ndarray],
                                  confidences_list: Optional[List[np.ndarray]]) -> List[Dict]:
        """Fall back to smaller batch sizes"""
        smaller_batch_size = max(1, self.config.batch_size // 2)
        results = []
        
        for i in range(0, len(landmarks_list), smaller_batch_size):
            batch_landmarks = landmarks_list[i:i+smaller_batch_size]
            batch_confidences = (confidences_list[i:i+smaller_batch_size] 
                               if confidences_list else None)
            
            batch_results = self.process_landmarks_batch(batch_landmarks, batch_confidences)
            results.extend(batch_results)
        
        return results
    
    def _convert_batch_to_smplx(self, landmarks_batch: torch.Tensor,
                               validity_mask: torch.Tensor,
                               confidences_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert batch to SMPL-X format efficiently"""
        batch_size = landmarks_batch.shape[0]
        device = landmarks_batch.device
        
        # Initialize SMPL-X batch tensors
        smplx_batch = torch.zeros((batch_size, 22, 3), device=device)
        weights_batch = torch.zeros((batch_size, 22), device=device)
        
        # Direct mapping indices (MediaPipe -> SMPL-X)
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
        
        # Apply direct mappings (vectorized)
        for smplx_idx, (mp_idx, base_confidence) in direct_mappings.items():
            if mp_idx < 33:
                smplx_batch[:, smplx_idx] = landmarks_batch[:, mp_idx]
                weights_batch[:, smplx_idx] = (
                    validity_mask[:, mp_idx].float() * 
                    confidences_batch[:, mp_idx] * 
                    base_confidence
                )
        
        # Calculate anatomical joints (vectorized where possible)
        self._calculate_anatomical_joints_batch(landmarks_batch, smplx_batch, weights_batch)
        
        return smplx_batch, weights_batch
    
    def _calculate_anatomical_joints_batch(self, landmarks_batch: torch.Tensor,
                                         smplx_batch: torch.Tensor,
                                         weights_batch: torch.Tensor):
        """Calculate anatomical joints for batch"""
        batch_size = landmarks_batch.shape[0]
        
        # Pelvis as center of hips (vectorized)
        left_hip = landmarks_batch[:, 23]   # (batch_size, 3)
        right_hip = landmarks_batch[:, 24]  # (batch_size, 3)
        pelvis = (left_hip + right_hip) / 2
        smplx_batch[:, 0] = pelvis
        weights_batch[:, 0] = 0.95
        
        # Spine chain calculation (vectorized)
        shoulder_center = (landmarks_batch[:, 11] + landmarks_batch[:, 12]) / 2
        spine_vector = shoulder_center - pelvis
        spine_length = torch.norm(spine_vector, dim=1, keepdim=True)
        
        # Avoid division by zero
        spine_length = torch.clamp(spine_length, min=1e-6)
        spine_unit = spine_vector / spine_length
        
        # Natural spine curvature (vectorized)
        spine_ratios = torch.tensor([0.2, 0.5, 0.8, 0.95], device=landmarks_batch.device)
        spine_indices = [3, 6, 9, 12]  # spine1, spine2, spine3, neck
        
        for i, (ratio, spine_idx) in enumerate(zip(spine_ratios, spine_indices)):
            smplx_batch[:, spine_idx] = pelvis + spine_unit * (spine_length * ratio)
            weights_batch[:, spine_idx] = 0.7 if i < 3 else 0.8  # neck has higher confidence
        
        # Feet positions (vectorized)
        foot_offset = torch.tensor([0, 0, -0.08], device=landmarks_batch.device)
        
        # Left foot
        valid_left_ankle = weights_batch[:, 7] > 0
        if torch.any(valid_left_ankle):
            smplx_batch[valid_left_ankle, 10] = (
                smplx_batch[valid_left_ankle, 7] + foot_offset
            )
            weights_batch[valid_left_ankle, 10] = weights_batch[valid_left_ankle, 7] * 0.8
        
        # Right foot
        valid_right_ankle = weights_batch[:, 8] > 0
        if torch.any(valid_right_ankle):
            smplx_batch[valid_right_ankle, 11] = (
                smplx_batch[valid_right_ankle, 8] + foot_offset
            )
            weights_batch[valid_right_ankle, 11] = weights_batch[valid_right_ankle, 8] * 0.8
        
        # Collar bones (vectorized where possible)
        valid_neck = weights_batch[:, 12] > 0
        
        if torch.any(valid_neck):
            neck = smplx_batch[:, 12]
            
            # Left collar
            valid_left_shoulder = weights_batch[:, 16] > 0
            valid_left = valid_neck & valid_left_shoulder
            if torch.any(valid_left):
                collar_vector = smplx_batch[valid_left, 16] - neck[valid_left]
                smplx_batch[valid_left, 13] = neck[valid_left] + collar_vector * 0.4
                weights_batch[valid_left, 13] = 0.6
            
            # Right collar
            valid_right_shoulder = weights_batch[:, 17] > 0
            valid_right = valid_neck & valid_right_shoulder
            if torch.any(valid_right):
                collar_vector = smplx_batch[valid_right, 17] - neck[valid_right]
                smplx_batch[valid_right, 14] = neck[valid_right] + collar_vector * 0.4
                weights_batch[valid_right, 14] = 0.6
    
    def _calculate_angles_batch(self, joints_batch: torch.Tensor,
                              weights_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate key angles for entire batch efficiently"""
        batch_size = joints_batch.shape[0]
        device = joints_batch.device
        
        angles = {}
        
        # Trunk angles (vectorized)
        valid_trunk = (weights_batch[:, 0] > 0) & (weights_batch[:, 12] > 0)
        if torch.any(valid_trunk):
            pelvis = joints_batch[valid_trunk, 0]
            neck = joints_batch[valid_trunk, 12]
            trunk_vector = neck - pelvis
            
            # Sagittal angle (forward/backward bend)
            sagittal_proj = torch.stack([torch.zeros_like(trunk_vector[:, 0]),
                                       trunk_vector[:, 1], 
                                       trunk_vector[:, 2]], dim=1)
            reference_up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
            
            sagittal_norm = torch.norm(sagittal_proj, dim=1, keepdim=True)
            sagittal_norm = torch.clamp(sagittal_norm, min=1e-6)
            sagittal_unit = sagittal_proj / sagittal_norm
            
            dot_product = torch.sum(sagittal_unit * reference_up, dim=1)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            trunk_sagittal = torch.rad2deg(torch.acos(dot_product))
            
            # Apply sign based on forward/backward
            forward_mask = trunk_vector[:, 1] > 0
            trunk_sagittal[forward_mask] = -trunk_sagittal[forward_mask]
            
            angles['trunk_sagittal'] = torch.full((batch_size,), float('nan'), device=device)
            angles['trunk_sagittal'][valid_trunk] = trunk_sagittal
        
        return angles
    
    def _package_batch_results(self, smplx_batch: torch.Tensor,
                             weights_batch: torch.Tensor,
                             validity_mask: torch.Tensor,
                             angles_batch: Dict[str, torch.Tensor]) -> List[Dict]:
        """Package batch results into individual frame results"""
        batch_size = smplx_batch.shape[0]
        results = []
        
        # Move to CPU for final packaging
        smplx_cpu = smplx_batch.cpu().numpy()
        weights_cpu = weights_batch.cpu().numpy()
        validity_cpu = validity_mask.cpu().numpy()
        
        angles_cpu = {}
        for angle_name, angle_tensor in angles_batch.items():
            angles_cpu[angle_name] = angle_tensor.cpu().numpy()
        
        for i in range(batch_size):
            result = {
                'joints': smplx_cpu[i],
                'weights': weights_cpu[i],
                'validation_result': {
                    'valid': bool(np.all(validity_cpu[i])),
                    'repair_applied': not np.all(validity_cpu[i]),
                    'violations': []  # Simplified for batch processing
                },
                'angles': {
                    'raw_angles': {name: float(values[i]) for name, values in angles_cpu.items()},
                    'filtered_angles': {},  # Would need Kalman filter integration
                    'angular_velocities': {}
                },
                'coordinate_transform_applied': True,
                'batch_processed': True
            }
            results.append(result)
        
        return results
    
    def _update_performance_stats(self, batch_size: int, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['batches_processed'] += 1
        self.performance_stats['total_frames'] += batch_size
        self.performance_stats['processing_times'].append(processing_time)
        
        # Calculate throughput (frames per second)
        throughput = batch_size / processing_time
        self.performance_stats['throughput_history'].append(throughput)
        
        # Memory usage tracking
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            self.performance_stats['memory_usage'].append(memory_used)
    
    def get_performance_statistics(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.performance_stats['processing_times']:
            return {'status': 'no_data'}
        
        times = np.array(self.performance_stats['processing_times'])
        throughputs = np.array(self.performance_stats['throughput_history'])
        
        stats = {
            'batches_processed': self.performance_stats['batches_processed'],
            'total_frames': self.performance_stats['total_frames'],
            'average_batch_time': float(np.mean(times)),
            'min_batch_time': float(np.min(times)),
            'max_batch_time': float(np.max(times)),
            'average_throughput': float(np.mean(throughputs)),
            'peak_throughput': float(np.max(throughputs)),
            'device': str(self.device),
            'batch_size': self.config.batch_size
        }
        
        if self.performance_stats['memory_usage']:
            memory_usage = np.array(self.performance_stats['memory_usage'])
            stats.update({
                'average_memory_gb': float(np.mean(memory_usage)),
                'peak_memory_gb': float(np.max(memory_usage))
            })
        
        return stats
    
    def optimize_batch_size(self, test_frames: List[np.ndarray]) -> int:
        """Automatically determine optimal batch size"""
        logger.info("Optimizing batch size...")
        
        best_batch_size = self.config.batch_size
        best_throughput = 0.0
        
        # Test different batch sizes
        test_sizes = [8, 16, 32, 64, 128]
        
        for batch_size in test_sizes:
            if batch_size > len(test_frames):
                continue
            
            # Test with sample data
            test_batch = test_frames[:batch_size]
            
            # Warm up
            self.process_landmarks_batch(test_batch)
            
            # Measure performance
            start_time = time.time()
            self.process_landmarks_batch(test_batch)
            processing_time = time.time() - start_time
            
            throughput = batch_size / processing_time
            
            logger.info(f"Batch size {batch_size}: {throughput:.1f} FPS")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
        
        logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.1f} FPS)")
        self.config.batch_size = best_batch_size
        
        return best_batch_size
    
    def clear_cache(self):
        """Clear GPU cache and reset statistics"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        gc.collect()
        
        self.performance_stats = {
            'batches_processed': 0,
            'total_frames': 0,
            'processing_times': [],
            'throughput_history': [],
            'memory_usage': []
        }
        
        logger.info("Cache cleared and statistics reset")


def create_high_performance_processor(batch_size: int = 32,
                                    device: str = 'auto',
                                    optimization_level: str = 'balanced') -> HighPerformanceBatchProcessor:
    """Factory function to create optimized batch processor"""
    config = BatchConfig(
        batch_size=batch_size,
        device=device,
        optimization_level=optimization_level
    )
    
    return HighPerformanceBatchProcessor(config)


if __name__ == "__main__":
    # Quick performance test
    processor = create_high_performance_processor(batch_size=16)
    
    print("Testing high-performance batch processor...")
    
    # Create test data
    test_landmarks = []
    for i in range(50):
        # Realistic pose with slight variation
        landmarks = np.zeros((33, 3), dtype=np.float32)
        landmarks[0] = [0, 0.8, 0]           # nose
        landmarks[11] = [-0.2, 0.6, 0]       # left_shoulder
        landmarks[12] = [0.2, 0.6, 0]        # right_shoulder
        landmarks[23] = [-0.15, 0, 0]        # left_hip
        landmarks[24] = [0.15, 0, 0]         # right_hip
        
        # Add variation
        landmarks += np.random.normal(0, 0.01, (33, 3))
        test_landmarks.append(landmarks)
    
    # Process in batches
    start_time = time.time()
    
    batch_size = processor.config.batch_size
    results = []
    
    for i in range(0, len(test_landmarks), batch_size):
        batch = test_landmarks[i:i+batch_size]
        batch_results = processor.process_landmarks_batch(batch)
        results.extend(batch_results)
    
    total_time = time.time() - start_time
    
    # Get performance stats
    stats = processor.get_performance_statistics()
    
    print(f"Batch processing completed:")
    print(f"  Frames processed: {len(results)}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average throughput: {stats['average_throughput']:.1f} FPS")
    print(f"  Device: {stats['device']}")
    print(f"  Batch size: {stats['batch_size']}")
    
    if 'average_memory_gb' in stats:
        print(f"  Average GPU memory: {stats['average_memory_gb']:.2f} GB")
    
    print("[PASS] High-performance batch processor test completed")
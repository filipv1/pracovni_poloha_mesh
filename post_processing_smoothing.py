#!/usr/bin/env python3
"""
Post-Processing Smoothing for Parallel 3D Human Mesh Pipeline
Apply temporal smoothing to PKL files generated without temporal smoothing
Goal: Achieve results as close as possible to original temporal smoothing during optimization
"""

import os
import sys
import time
import numpy as np
import pickle
import torch
from pathlib import Path
import json
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import libraries with proper error handling
try:
    import smplx
    SMPLX_AVAILABLE = True
    print("SMPL-X: Available for mesh regeneration")
except ImportError:
    SMPLX_AVAILABLE = False
    print("SMPL-X: Not Available - mesh regeneration will be skipped")


class PKLParameterExtractor:
    """Extract SMPL-X parameters from PKL mesh sequence files"""
    
    def __init__(self):
        self.parameter_names = ['body_pose', 'global_orient', 'transl', 'betas']
        
    def load_pkl_data(self, pkl_path):
        """Load PKL data and validate structure"""
        pkl_path = Path(pkl_path)
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"📂 Loading PKL data: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate PKL structure
        if 'mesh_sequence' not in data:
            raise ValueError("PKL file missing 'mesh_sequence' key")
        
        mesh_sequence = data['mesh_sequence']
        if not mesh_sequence:
            raise ValueError("Empty mesh sequence")
        
        print(f"✅ Loaded {len(mesh_sequence)} frames from PKL")
        
        # Check if this is from parallel processing
        metadata = data.get('metadata', {})
        if metadata.get('processing_method') == 'parallel_no_temporal_smoothing':
            print("🎯 Detected parallel processing PKL - ready for smoothing")
            original_temporal_weights = metadata.get('original_temporal_weights', {})
            print(f"📊 Original temporal weights: {original_temporal_weights}")
        else:
            print("⚠️  PKL may already have temporal smoothing applied")
        
        return data, mesh_sequence, metadata
    
    def extract_parameters_from_sequence(self, mesh_sequence):
        """Extract SMPL-X parameters from mesh sequence"""
        
        print(f"🔍 Extracting parameters from {len(mesh_sequence)} frames...")
        
        # Validate first frame structure
        first_frame = mesh_sequence[0]
        if 'smplx_params' not in first_frame:
            raise ValueError("Mesh frame missing 'smplx_params' key")
        
        smplx_params = first_frame['smplx_params']
        missing_params = [p for p in self.parameter_names if p not in smplx_params]
        if missing_params:
            raise ValueError(f"Missing SMPL-X parameters: {missing_params}")
        
        # Extract parameters from all frames
        extracted_params = []
        
        for frame_idx, mesh_frame in enumerate(mesh_sequence):
            frame_params = {}
            
            for param_name in self.parameter_names:
                param_data = mesh_frame['smplx_params'][param_name]
                
                # Convert to numpy if it's a torch tensor
                if hasattr(param_data, 'cpu'):
                    param_data = param_data.cpu().numpy()
                elif hasattr(param_data, 'numpy'):
                    param_data = param_data.numpy()
                
                # Ensure proper shape
                if param_data.ndim > 1:
                    param_data = param_data.flatten()
                
                frame_params[param_name] = param_data
            
            extracted_params.append(frame_params)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"  Extracted parameters from frame {frame_idx + 1}")
        
        print(f"✅ Extracted parameters from all {len(extracted_params)} frames")
        
        # Validate parameter shapes
        self._validate_parameter_shapes(extracted_params)
        
        return extracted_params
    
    def _validate_parameter_shapes(self, extracted_params):
        """Validate that all parameters have consistent shapes"""
        
        print("🔍 Validating parameter shapes...")
        
        if not extracted_params:
            raise ValueError("No parameters extracted")
        
        # Check shapes from first frame
        first_frame = extracted_params[0]
        expected_shapes = {
            param_name: param_data.shape
            for param_name, param_data in first_frame.items()
        }
        
        print(f"📊 Expected parameter shapes:")
        for param_name, shape in expected_shapes.items():
            print(f"   {param_name}: {shape}")
        
        # Validate all frames have consistent shapes
        for frame_idx, frame_params in enumerate(extracted_params):
            for param_name, expected_shape in expected_shapes.items():
                actual_shape = frame_params[param_name].shape
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch in frame {frame_idx}, parameter {param_name}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
        
        # Validate expected SMPL-X shapes
        expected_smplx_shapes = {
            'body_pose': (63,),    # 21 joints × 3 axis-angle parameters
            'global_orient': (3,), # Global rotation (axis-angle)
            'transl': (3,),        # Global translation
            'betas': (10,)         # Shape parameters
        }
        
        for param_name, expected_shape in expected_smplx_shapes.items():
            if param_name in expected_shapes:
                actual_shape = expected_shapes[param_name]
                if actual_shape != expected_shape:
                    print(f"⚠️  WARNING: {param_name} shape {actual_shape} differs from expected SMPL-X shape {expected_shape}")
        
        print("✅ Parameter shape validation complete")
    
    def convert_to_numpy_arrays(self, extracted_params):
        """Convert extracted parameters to numpy arrays for processing"""
        
        print("🔄 Converting parameters to numpy arrays...")
        
        # Get parameter dimensions
        first_frame = extracted_params[0]
        param_dimensions = {
            param_name: param_data.shape[0] if param_data.ndim > 0 else 1
            for param_name, param_data in first_frame.items()
        }
        
        # Create numpy arrays
        n_frames = len(extracted_params)
        numpy_params = {}
        
        for param_name, param_dim in param_dimensions.items():
            # Create array: (n_frames, param_dim)
            param_array = np.zeros((n_frames, param_dim))
            
            for frame_idx, frame_params in enumerate(extracted_params):
                param_data = frame_params[param_name]
                if param_data.ndim == 0:
                    param_array[frame_idx, 0] = param_data
                else:
                    param_array[frame_idx] = param_data
            
            numpy_params[param_name] = param_array
            print(f"   {param_name}: {param_array.shape}")
        
        print("✅ Parameter conversion complete")
        return numpy_params


class BasicTemporalSmoother:
    """Basic temporal smoothing algorithms to replicate original temporal smoothing behavior"""
    
    def __init__(self, original_temporal_weights=None):
        # Use original temporal weights from metadata, or defaults
        if original_temporal_weights:
            self.temporal_weights = original_temporal_weights
        else:
            # Default weights from original implementation
            self.temporal_weights = {
                'body_pose': 0.3,      # Was: temporal_alpha * 1.0
                'betas': 0.03,         # Was: temporal_alpha * 0.1  
                'global_orient': 0.0,  # Was not in original temporal loss
                'transl': 0.0          # Was not in original temporal loss
            }
        
        print(f"🎯 Using temporal weights: {self.temporal_weights}")
    
    def apply_moving_average_smoothing(self, param_arrays, window_sizes=None):
        """Apply moving average smoothing with parameter-specific window sizes"""
        
        if window_sizes is None:
            # Default window sizes based on temporal weights
            window_sizes = {
                'body_pose': 5,        # Strong smoothing for joint rotations
                'betas': 15,           # Very strong smoothing for shape (should be stable)
                'global_orient': 3,    # Light smoothing for global orientation
                'transl': 3            # Light smoothing for translation
            }
        
        print(f"📊 Applying moving average smoothing with windows: {window_sizes}")
        
        smoothed_params = {}
        
        for param_name, param_data in param_arrays.items():
            window_size = window_sizes.get(param_name, 5)
            temporal_weight = self.temporal_weights.get(param_name, 0.1)
            
            print(f"   Smoothing {param_name}: window={window_size}, weight={temporal_weight}")
            
            if temporal_weight == 0.0:
                # No smoothing for parameters not in original temporal loss
                smoothed_params[param_name] = param_data.copy()
                print(f"     → No smoothing applied (weight=0.0)")
            else:
                # Apply weighted moving average
                smoothed_param = self._apply_weighted_moving_average(
                    param_data, window_size, temporal_weight
                )
                smoothed_params[param_name] = smoothed_param
                
                # Calculate smoothing effect
                original_std = np.std(param_data)
                smoothed_std = np.std(smoothed_param)
                reduction = (original_std - smoothed_std) / original_std * 100
                print(f"     → Variance reduction: {reduction:.1f}%")
        
        return smoothed_params
    
    def _apply_weighted_moving_average(self, param_data, window_size, temporal_weight):
        """Apply weighted moving average smoothing"""
        
        n_frames, param_dim = param_data.shape
        smoothed_data = param_data.copy()
        
        # Ensure odd window size for centered smoothing
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        
        for frame_idx in range(n_frames):
            # Define window bounds
            start_idx = max(0, frame_idx - half_window)
            end_idx = min(n_frames, frame_idx + half_window + 1)
            
            # Extract window data
            window_data = param_data[start_idx:end_idx]
            window_frames = list(range(start_idx, end_idx))
            
            # Calculate weights (higher weight for current frame, decreasing with distance)
            weights = []
            for w_frame in window_frames:
                distance = abs(w_frame - frame_idx)
                # Exponential decay weight (similar to temporal loss in original)
                weight = np.exp(-distance * temporal_weight)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Apply weighted average
            for param_idx in range(param_dim):
                param_values = window_data[:, param_idx]
                smoothed_value = np.average(param_values, weights=weights)
                smoothed_data[frame_idx, param_idx] = smoothed_value
        
        return smoothed_data
    
    def apply_savgol_smoothing(self, param_arrays, window_lengths=None, polyorder=2):
        """Apply Savitzky-Golay smoothing (good for preserving features)"""
        
        if window_lengths is None:
            window_lengths = {
                'body_pose': 7,        # Moderate smoothing
                'betas': 15,           # Strong smoothing  
                'global_orient': 5,    # Light smoothing
                'transl': 5            # Light smoothing
            }
        
        print(f"📊 Applying Savitzky-Golay smoothing with windows: {window_lengths}")
        
        smoothed_params = {}
        
        for param_name, param_data in param_arrays.items():
            window_length = window_lengths.get(param_name, 7)
            temporal_weight = self.temporal_weights.get(param_name, 0.1)
            
            # Ensure window length is odd and not larger than data
            window_length = min(window_length, param_data.shape[0])
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < 3:
                window_length = 3
            
            print(f"   Smoothing {param_name}: window={window_length}, poly={polyorder}")
            
            if temporal_weight == 0.0:
                # No smoothing
                smoothed_params[param_name] = param_data.copy()
            else:
                # Apply Savitzky-Golay filter
                smoothed_param = signal.savgol_filter(
                    param_data, window_length, polyorder, axis=0
                )
                
                # Blend with original based on temporal weight
                blend_factor = min(temporal_weight, 1.0)  # Cap at 1.0
                final_param = (
                    blend_factor * smoothed_param + 
                    (1.0 - blend_factor) * param_data
                )
                
                smoothed_params[param_name] = final_param
                
                # Calculate effect
                original_std = np.std(param_data)
                smoothed_std = np.std(final_param)
                reduction = (original_std - smoothed_std) / original_std * 100
                print(f"     → Variance reduction: {reduction:.1f}%")
        
        return smoothed_params


class AdvancedTemporalSmoother:
    """Advanced smoothing algorithms for optimal quality"""
    
    def __init__(self, original_temporal_weights=None):
        if original_temporal_weights:
            self.temporal_weights = original_temporal_weights
        else:
            self.temporal_weights = {
                'body_pose': 0.3,
                'betas': 0.03, 
                'global_orient': 0.0,
                'transl': 0.0
            }
    
    def apply_bilateral_smoothing(self, param_arrays, spatial_sigmas=None, temporal_sigmas=None):
        """Apply bilateral filtering - preserves sharp movements while smoothing noise"""
        
        if spatial_sigmas is None:
            spatial_sigmas = {
                'body_pose': 2.0,      # Higher = more spatial smoothing
                'betas': 0.5,          # Lower = preserve shape changes
                'global_orient': 1.5,  
                'transl': 1.5
            }
        
        if temporal_sigmas is None:
            temporal_sigmas = {
                'body_pose': 0.3,      # Matches original temporal_alpha
                'betas': 0.1,          # Matches original beta weight
                'global_orient': 0.2,
                'transl': 0.2
            }
        
        print(f"🔬 Applying bilateral smoothing")
        print(f"   Spatial sigmas: {spatial_sigmas}")
        print(f"   Temporal sigmas: {temporal_sigmas}")
        
        smoothed_params = {}
        
        for param_name, param_data in param_arrays.items():
            spatial_sigma = spatial_sigmas.get(param_name, 1.0)
            temporal_sigma = temporal_sigmas.get(param_name, 0.3)
            temporal_weight = self.temporal_weights.get(param_name, 0.1)
            
            print(f"   Processing {param_name}...")
            
            if temporal_weight == 0.0:
                smoothed_params[param_name] = param_data.copy()
                continue
            
            smoothed_param = self._apply_bilateral_filter(
                param_data, spatial_sigma, temporal_sigma
            )
            
            # Blend with original based on temporal weight
            blend_factor = min(temporal_weight, 1.0)
            final_param = (
                blend_factor * smoothed_param + 
                (1.0 - blend_factor) * param_data
            )
            
            smoothed_params[param_name] = final_param
            
            # Calculate effect
            original_std = np.std(param_data)
            smoothed_std = np.std(final_param)
            reduction = (original_std - smoothed_std) / original_std * 100
            print(f"     → Variance reduction: {reduction:.1f}%")
        
        return smoothed_params
    
    def _apply_bilateral_filter(self, param_data, spatial_sigma, temporal_sigma):
        """Apply bilateral filter to parameter sequence"""
        
        n_frames, param_dim = param_data.shape
        smoothed_data = np.zeros_like(param_data)
        
        # Define temporal window (5 frames = original max_history)
        temporal_window = 5
        half_window = temporal_window // 2
        
        for frame_idx in range(n_frames):
            # Define temporal window bounds
            start_idx = max(0, frame_idx - half_window)
            end_idx = min(n_frames, frame_idx + half_window + 1)
            
            for param_idx in range(param_dim):
                # Get current parameter value
                current_value = param_data[frame_idx, param_idx]
                
                # Collect weights and values from temporal window
                weights = []
                values = []
                
                for other_frame in range(start_idx, end_idx):
                    other_value = param_data[other_frame, param_idx]
                    
                    # Temporal weight (distance in time)
                    temporal_distance = abs(other_frame - frame_idx)
                    temporal_weight = np.exp(-(temporal_distance**2) / (2 * temporal_sigma**2))
                    
                    # Spatial weight (difference in parameter value)
                    spatial_distance = abs(current_value - other_value)
                    spatial_weight = np.exp(-(spatial_distance**2) / (2 * spatial_sigma**2))
                    
                    # Combined weight
                    total_weight = temporal_weight * spatial_weight
                    
                    weights.append(total_weight)
                    values.append(other_value)
                
                # Normalize weights and compute weighted average
                weights = np.array(weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    smoothed_value = np.average(values, weights=weights)
                    smoothed_data[frame_idx, param_idx] = smoothed_value
                else:
                    # Fallback to original value
                    smoothed_data[frame_idx, param_idx] = current_value
        
        return smoothed_data
    
    def stabilize_shape_parameters(self, param_arrays, method='median'):
        """Stabilize shape parameters (betas) across sequence"""
        
        print(f"🎯 Stabilizing shape parameters using {method} method")
        
        stabilized_params = param_arrays.copy()
        
        if 'betas' in param_arrays:
            betas_data = param_arrays['betas']
            n_frames, n_betas = betas_data.shape
            
            if method == 'median':
                # Use median shape across all frames
                stable_betas = np.median(betas_data, axis=0)
                stabilized_betas = np.tile(stable_betas, (n_frames, 1))
                
            elif method == 'mean':
                # Use mean shape across all frames
                stable_betas = np.mean(betas_data, axis=0)
                stabilized_betas = np.tile(stable_betas, (n_frames, 1))
                
            elif method == 'heavy_smooth':
                # Heavy smoothing (window = 50% of sequence length)
                window_length = max(15, min(51, n_frames // 2))
                if window_length % 2 == 0:
                    window_length -= 1
                stabilized_betas = signal.savgol_filter(
                    betas_data, window_length, 2, axis=0
                )
            
            stabilized_params['betas'] = stabilized_betas
            
            # Calculate stabilization effect
            original_var = np.var(betas_data, axis=0).mean()
            stabilized_var = np.var(stabilized_betas, axis=0).mean()
            reduction = (original_var - stabilized_var) / original_var * 100
            print(f"   Shape parameter variance reduction: {reduction:.1f}%")
        
        return stabilized_params


class OutlierDetector:
    """Detect and correct outliers in parameter sequences from parallel processing"""
    
    def __init__(self, outlier_threshold=3.0):
        self.outlier_threshold = outlier_threshold
        
    def detect_and_correct_outliers(self, param_arrays, method='interpolate'):
        """Detect and correct outliers in parameter sequences"""
        
        print(f"🔍 Detecting outliers (threshold: {self.outlier_threshold}σ)")
        
        corrected_params = {}
        total_outliers = 0
        
        for param_name, param_data in param_arrays.items():
            print(f"   Analyzing {param_name}...")
            
            # Detect outliers
            outlier_frames = self._detect_outliers_zscore(param_data)
            
            if len(outlier_frames) > 0:
                print(f"     Found {len(outlier_frames)} outlier frames: {outlier_frames[:10]}...")
                total_outliers += len(outlier_frames)
                
                # Correct outliers
                if method == 'interpolate':
                    corrected_data = self._correct_outliers_interpolation(param_data, outlier_frames)
                elif method == 'median_filter':
                    corrected_data = self._correct_outliers_median_filter(param_data)
                elif method == 'neighbor_average':
                    corrected_data = self._correct_outliers_neighbor_average(param_data, outlier_frames)
                else:
                    corrected_data = param_data.copy()
                    
                corrected_params[param_name] = corrected_data
            else:
                print(f"     No outliers detected")
                corrected_params[param_name] = param_data.copy()
        
        print(f"✅ Outlier detection complete: {total_outliers} outliers corrected")
        return corrected_params
    
    def _detect_outliers_zscore(self, param_data):
        """Detect outliers using Z-score method"""
        
        n_frames, param_dim = param_data.shape
        outlier_frames = set()
        
        # For each parameter dimension
        for dim_idx in range(param_dim):
            dim_data = param_data[:, dim_idx]
            
            # Calculate frame-to-frame differences (temporal derivative)
            if n_frames > 1:
                diff_data = np.diff(dim_data)
                
                # Calculate Z-scores of differences
                if len(diff_data) > 0:
                    mean_diff = np.mean(diff_data)
                    std_diff = np.std(diff_data)
                    
                    if std_diff > 0:
                        z_scores = np.abs(diff_data - mean_diff) / std_diff
                        
                        # Find outlier frames
                        outlier_indices = np.where(z_scores > self.outlier_threshold)[0]
                        
                        # Convert diff indices to frame indices
                        for diff_idx in outlier_indices:
                            # An outlier in diff[i] means frames i and i+1 are suspect
                            outlier_frames.add(diff_idx)
                            outlier_frames.add(diff_idx + 1)
            
            # Also check for extreme absolute values
            if n_frames > 0:
                mean_val = np.mean(dim_data)
                std_val = np.std(dim_data)
                
                if std_val > 0:
                    z_scores_abs = np.abs(dim_data - mean_val) / std_val
                    extreme_indices = np.where(z_scores_abs > self.outlier_threshold * 1.5)[0]  # Higher threshold for absolute values
                    
                    for frame_idx in extreme_indices:
                        outlier_frames.add(frame_idx)
        
        return sorted(list(outlier_frames))
    
    def _correct_outliers_interpolation(self, param_data, outlier_frames):
        """Correct outliers using linear interpolation"""
        
        corrected_data = param_data.copy()
        n_frames, param_dim = param_data.shape
        
        for frame_idx in outlier_frames:
            if 0 < frame_idx < n_frames - 1:
                # Find nearest non-outlier neighbors
                prev_frame = frame_idx - 1
                next_frame = frame_idx + 1
                
                # Extend search if neighbors are also outliers
                while prev_frame > 0 and prev_frame in outlier_frames:
                    prev_frame -= 1
                while next_frame < n_frames - 1 and next_frame in outlier_frames:
                    next_frame += 1
                
                # Interpolate between valid neighbors
                if prev_frame >= 0 and next_frame < n_frames:
                    alpha = (frame_idx - prev_frame) / (next_frame - prev_frame)
                    corrected_data[frame_idx] = (
                        (1 - alpha) * param_data[prev_frame] + 
                        alpha * param_data[next_frame]
                    )
            
            elif frame_idx == 0:
                # First frame - use second frame
                if n_frames > 1:
                    corrected_data[frame_idx] = param_data[1]
            
            elif frame_idx == n_frames - 1:
                # Last frame - use second-to-last frame
                if n_frames > 1:
                    corrected_data[frame_idx] = param_data[n_frames - 2]
        
        return corrected_data
    
    def _correct_outliers_median_filter(self, param_data, window_size=5):
        """Correct outliers using median filter"""
        
        from scipy import ndimage
        
        corrected_data = np.zeros_like(param_data)
        n_frames, param_dim = param_data.shape
        
        for dim_idx in range(param_dim):
            dim_data = param_data[:, dim_idx]
            
            # Apply median filter
            filtered_data = ndimage.median_filter(dim_data, size=window_size)
            corrected_data[:, dim_idx] = filtered_data
        
        return corrected_data
    
    def _correct_outliers_neighbor_average(self, param_data, outlier_frames, window_size=3):
        """Correct outliers using neighbor averaging"""
        
        corrected_data = param_data.copy()
        n_frames, param_dim = param_data.shape
        half_window = window_size // 2
        
        for frame_idx in outlier_frames:
            # Define neighborhood
            start_idx = max(0, frame_idx - half_window)
            end_idx = min(n_frames, frame_idx + half_window + 1)
            
            # Collect non-outlier neighbors
            neighbor_values = []
            for neighbor_idx in range(start_idx, end_idx):
                if neighbor_idx != frame_idx and neighbor_idx not in outlier_frames:
                    neighbor_values.append(param_data[neighbor_idx])
            
            # Average non-outlier neighbors
            if neighbor_values:
                corrected_data[frame_idx] = np.mean(neighbor_values, axis=0)
            
        return corrected_data
    
    def detect_optimization_failures(self, param_arrays, metadata=None):
        """Detect frames where optimization likely failed completely"""
        
        print("🔍 Detecting optimization failures...")
        
        failed_frames = set()
        
        for param_name, param_data in param_arrays.items():
            n_frames, param_dim = param_data.shape
            
            # Check for stuck/frozen parameters (same value across many frames)
            if n_frames > 10:  # Only check if enough frames
                for frame_idx in range(5, n_frames - 5):  # Skip boundary frames
                    # Check if parameter values are identical to neighbors
                    window_data = param_data[frame_idx-2:frame_idx+3]  # 5-frame window
                    
                    # If all values in window are very similar, it might be stuck
                    for dim_idx in range(param_dim):
                        window_values = window_data[:, dim_idx]
                        if np.std(window_values) < 1e-6:  # Very low variance
                            # Check if this is different from global pattern
                            global_std = np.std(param_data[:, dim_idx])
                            if global_std > 1e-3:  # Global variance exists
                                failed_frames.add(frame_idx)
            
            # Check for NaN or infinite values
            for frame_idx in range(n_frames):
                frame_data = param_data[frame_idx]
                if np.any(np.isnan(frame_data)) or np.any(np.isinf(frame_data)):
                    failed_frames.add(frame_idx)
            
            # Check for extreme parameter values (likely optimization divergence)
            body_pose_limits = 3.14  # Reasonable joint angle limits in radians
            beta_limits = 3.0       # Reasonable shape parameter limits
            
            if param_name == 'body_pose':
                extreme_frames = np.where(np.abs(param_data) > body_pose_limits)[0]
                failed_frames.update(extreme_frames)
            elif param_name == 'betas':
                extreme_frames = np.where(np.abs(param_data) > beta_limits)[0]
                failed_frames.update(extreme_frames)
        
        failed_frames = sorted(list(failed_frames))
        
        if failed_frames:
            print(f"   Found {len(failed_frames)} frames with optimization failures")
            print(f"   Failed frames: {failed_frames[:20]}{'...' if len(failed_frames) > 20 else ''}")
        else:
            print("   No optimization failures detected")
        
        return failed_frames
    
    def assess_parameter_quality(self, param_arrays):
        """Assess overall quality of parameters"""
        
        print("📊 Parameter quality assessment:")
        
        quality_metrics = {}
        
        for param_name, param_data in param_arrays.items():
            n_frames, param_dim = param_data.shape
            
            # Calculate metrics
            metrics = {}
            
            # Temporal smoothness (lower is smoother)
            if n_frames > 1:
                diff_data = np.diff(param_data, axis=0)
                temporal_smoothness = np.mean(np.abs(diff_data))
                metrics['temporal_smoothness'] = temporal_smoothness
            
            # Parameter stability (variance)
            parameter_variance = np.mean(np.var(param_data, axis=0))
            metrics['parameter_variance'] = parameter_variance
            
            # Outlier count
            outlier_frames = self._detect_outliers_zscore(param_data)
            metrics['outlier_count'] = len(outlier_frames)
            metrics['outlier_percentage'] = len(outlier_frames) / n_frames * 100
            
            # NaN/inf count
            nan_count = np.sum(np.isnan(param_data))
            inf_count = np.sum(np.isinf(param_data))
            metrics['nan_count'] = nan_count
            metrics['inf_count'] = inf_count
            
            quality_metrics[param_name] = metrics
            
            # Print summary
            print(f"   {param_name}:")
            print(f"     Temporal smoothness: {temporal_smoothness:.6f}")
            print(f"     Parameter variance: {parameter_variance:.6f}")
            print(f"     Outliers: {len(outlier_frames)} ({len(outlier_frames)/n_frames*100:.1f}%)")
            if nan_count > 0 or inf_count > 0:
                print(f"     Invalid values: {nan_count} NaN, {inf_count} Inf")
        
        return quality_metrics


class MeshRegenerator:
    """Regenerate meshes from smoothed SMPL-X parameters"""
    
    def __init__(self, smplx_path="models/smplx", device='cpu', gender='neutral'):
        self.device = torch.device(device)
        self.smplx_path = Path(smplx_path)
        self.gender = gender
        
        # Load SMPL-X model
        if SMPLX_AVAILABLE and self._verify_model_files():
            try:
                self.smplx_model = smplx.SMPLX(
                    model_path=str(self.smplx_path),
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
                print(f"✅ SMPL-X Model loaded for mesh regeneration ({gender})")
            except Exception as e:
                print(f"❌ SMPL-X Model load failed: {e}")
                self.model_ready = False
        else:
            self.model_ready = False
            print("❌ SMPL-X Model not available - mesh regeneration disabled")
    
    def _verify_model_files(self):
        """Verify all required SMPL-X model files exist"""
        required_files = [f"SMPLX_{self.gender.upper()}.npz"]
        
        for file in required_files:
            if not (self.smplx_path / file).exists():
                return False
        return True
    
    def regenerate_mesh_sequence(self, smoothed_param_arrays, original_metadata=None):
        """Regenerate complete mesh sequence from smoothed parameters"""
        
        if not self.model_ready:
            print("❌ Cannot regenerate meshes - SMPL-X model not available")
            return None
        
        n_frames = next(iter(smoothed_param_arrays.values())).shape[0]
        
        print(f"🔄 Regenerating {n_frames} meshes from smoothed parameters...")
        
        regenerated_sequence = []
        
        for frame_idx in range(n_frames):
            # Extract parameters for this frame
            frame_params = {}
            for param_name, param_array in smoothed_param_arrays.items():
                frame_params[param_name] = torch.tensor(
                    param_array[frame_idx], 
                    dtype=torch.float32, 
                    device=self.device
                ).unsqueeze(0)  # Add batch dimension
            
            # Generate mesh with SMPL-X
            mesh_data = self._generate_single_mesh(frame_params, frame_idx)
            
            if mesh_data:
                regenerated_sequence.append(mesh_data)
            
            if (frame_idx + 1) % 50 == 0:
                print(f"  Regenerated {frame_idx + 1}/{n_frames} meshes")
        
        print(f"✅ Regenerated {len(regenerated_sequence)} meshes")
        return regenerated_sequence
    
    def _generate_single_mesh(self, frame_params, frame_idx):
        """Generate single mesh from parameters"""
        
        try:
            with torch.no_grad():
                # Forward pass through SMPL-X
                smpl_output = self.smplx_model(
                    body_pose=frame_params['body_pose'],
                    global_orient=frame_params['global_orient'],
                    transl=frame_params['transl'],
                    betas=frame_params['betas']
                )
                
                # Extract mesh data
                vertices = smpl_output.vertices[0].cpu().numpy()
                faces = self.smplx_model.faces
                joints = smpl_output.joints[0].cpu().numpy()
                
                # Fix orientation (same as original)
                vertices[:, 1] *= -1
                joints[:, 1] *= -1
                
                # Create mesh result
                mesh_result = {
                    'vertices': vertices,
                    'faces': faces,
                    'joints': joints,
                    'smplx_params': {
                        k: v.cpu().numpy() for k, v in frame_params.items()
                    },
                    'fitting_error': 0.0,  # No optimization error for regenerated meshes
                    'vertex_count': len(vertices),
                    'face_count': len(faces),
                    'frame_id': frame_idx,
                    'regenerated_from_smoothed': True
                }
                
                return mesh_result
                
        except Exception as e:
            print(f"❌ Error generating mesh for frame {frame_idx}: {e}")
            return None


class PostProcessingSmoothingPipeline:
    """Complete post-processing smoothing pipeline"""
    
    def __init__(self, smplx_path="models/smplx", device='cpu', gender='neutral'):
        self.extractor = PKLParameterExtractor()
        self.basic_smoother = BasicTemporalSmoother()
        self.advanced_smoother = AdvancedTemporalSmoother()
        self.outlier_detector = OutlierDetector(outlier_threshold=3.0)
        self.regenerator = MeshRegenerator(smplx_path, device, gender)
        
    def apply_post_processing_smoothing(self, input_pkl_path, output_pkl_path, 
                                       smoothing_method='bilateral', 
                                       stabilize_shape=True,
                                       quality_assessment=True):
        """Complete post-processing smoothing pipeline"""
        
        print("🚀 STARTING POST-PROCESSING SMOOTHING PIPELINE")
        print("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Load and extract parameters
        print("\n📂 STEP 1: Loading PKL data and extracting parameters")
        print("-" * 50)
        
        try:
            data, mesh_sequence, metadata = self.extractor.load_pkl_data(input_pkl_path)
            extracted_params = self.extractor.extract_parameters_from_sequence(mesh_sequence)
            numpy_params = self.extractor.convert_to_numpy_arrays(extracted_params)
            
            original_temporal_weights = metadata.get('original_temporal_weights', None)
            
            # Update smoothers with original weights
            self.basic_smoother = BasicTemporalSmoother(original_temporal_weights)
            self.advanced_smoother = AdvancedTemporalSmoother(original_temporal_weights)
            
        except Exception as e:
            print(f"❌ Error in parameter extraction: {e}")
            return False
        
        # Step 2: Outlier detection and correction
        print(f"\n🔍 STEP 2: Outlier detection and correction")
        print("-" * 50)
        
        try:
            # Assess initial quality
            self.outlier_detector.assess_parameter_quality(numpy_params)
            
            # Detect and correct outliers
            corrected_params = self.outlier_detector.detect_and_correct_outliers(
                numpy_params, method='interpolate'
            )
            
            # Detect optimization failures
            failed_frames = self.outlier_detector.detect_optimization_failures(corrected_params)
            
        except Exception as e:
            print(f"❌ Error in outlier detection: {e}")
            corrected_params = numpy_params  # Use original if outlier detection fails
        
        # Step 3: Apply smoothing
        print(f"\n🎯 STEP 3: Applying {smoothing_method} smoothing")
        print("-" * 50)
        
        try:
            if smoothing_method == 'moving_average':
                smoothed_params = self.basic_smoother.apply_moving_average_smoothing(corrected_params)
            elif smoothing_method == 'savgol':
                smoothed_params = self.basic_smoother.apply_savgol_smoothing(corrected_params)
            elif smoothing_method == 'bilateral':
                smoothed_params = self.advanced_smoother.apply_bilateral_smoothing(corrected_params)
            else:
                print(f"❌ Unknown smoothing method: {smoothing_method}")
                return False
                
        except Exception as e:
            print(f"❌ Error in smoothing: {e}")
            return False
        
        # Step 3: Shape stabilization (optional)
        if stabilize_shape:
            print(f"\n🎯 STEP 3: Shape parameter stabilization")
            print("-" * 50)
            
            try:
                smoothed_params = self.advanced_smoother.stabilize_shape_parameters(
                    smoothed_params, method='heavy_smooth'
                )
            except Exception as e:
                print(f"❌ Error in shape stabilization: {e}")
                return False
        
        # Step 4: Quality assessment (optional)
        if quality_assessment:
            print(f"\n📊 STEP 4: Quality assessment")
            print("-" * 50)
            
            try:
                self._assess_smoothing_quality(corrected_params, smoothed_params)
            except Exception as e:
                print(f"⚠️ Warning in quality assessment: {e}")
        
        # Step 5: Mesh regeneration
        print(f"\n🔄 STEP 5: Regenerating meshes from smoothed parameters")
        print("-" * 50)
        
        try:
            regenerated_sequence = self.regenerator.regenerate_mesh_sequence(
                smoothed_params, metadata
            )
            
            if not regenerated_sequence:
                print("❌ Mesh regeneration failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in mesh regeneration: {e}")
            return False
        
        # Step 6: Save smoothed PKL
        print(f"\n💾 STEP 6: Saving smoothed PKL")
        print("-" * 50)
        
        try:
            # Update metadata
            smoothed_metadata = metadata.copy()
            smoothed_metadata.update({
                'post_processing_applied': True,
                'smoothing_method': smoothing_method,
                'shape_stabilized': stabilize_shape,
                'processing_date_smoothing': time.strftime('%Y-%m-%d %H:%M:%S'),
                'original_frames': len(mesh_sequence),
                'smoothed_frames': len(regenerated_sequence)
            })
            
            # Create new PKL data
            smoothed_data = {
                'mesh_sequence': regenerated_sequence,
                'metadata': smoothed_metadata,
                'original_metadata': metadata,  # Keep original for reference
                'smoothing_stats': {
                    'method': smoothing_method,
                    'processing_time_seconds': time.time() - start_time,
                    'shape_stabilized': stabilize_shape
                }
            }
            
            # Save smoothed PKL
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(smoothed_data, f)
            
            print(f"✅ Smoothed PKL saved: {output_pkl_path}")
            
        except Exception as e:
            print(f"❌ Error saving PKL: {e}")
            return False
        
        # Step 7: Summary
        processing_time = time.time() - start_time
        
        print(f"\n🎉 POST-PROCESSING SMOOTHING COMPLETE!")
        print("=" * 70)
        print(f"   Input frames: {len(mesh_sequence)}")
        print(f"   Output frames: {len(regenerated_sequence)}")  
        print(f"   Smoothing method: {smoothing_method}")
        print(f"   Shape stabilized: {stabilize_shape}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        print(f"   Output file: {output_pkl_path}")
        
        return True
    
    def _assess_smoothing_quality(self, original_params, smoothed_params):
        """Assess quality of smoothing"""
        
        print("📊 Smoothing quality assessment:")
        
        for param_name in original_params.keys():
            original = original_params[param_name]
            smoothed = smoothed_params[param_name]
            
            # Calculate variance reduction
            original_var = np.var(original, axis=0).mean()
            smoothed_var = np.var(smoothed, axis=0).mean()
            var_reduction = (original_var - smoothed_var) / original_var * 100
            
            # Calculate temporal smoothness (difference between consecutive frames)
            original_diff = np.diff(original, axis=0)
            smoothed_diff = np.diff(smoothed, axis=0)
            
            original_smoothness = np.mean(np.abs(original_diff))
            smoothed_smoothness = np.mean(np.abs(smoothed_diff))
            smoothness_improvement = (original_smoothness - smoothed_smoothness) / original_smoothness * 100
            
            print(f"   {param_name}:")
            print(f"     Variance reduction: {var_reduction:.1f}%")
            print(f"     Smoothness improvement: {smoothness_improvement:.1f}%")


def main():
    """Test parameter extraction and full pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python post_processing_smoothing.py <pkl_file> [output_pkl] [method]")
        print("Methods: moving_average, savgol, bilateral (default)")
        return
    
    pkl_file = sys.argv[1]
    
    # Parse arguments
    output_pkl = sys.argv[2] if len(sys.argv) > 2 else None
    smoothing_method = sys.argv[3] if len(sys.argv) > 3 else 'bilateral'
    
    if output_pkl:
        # Run full pipeline
        print(f"🚀 Running full post-processing smoothing pipeline")
        print(f"   Input: {pkl_file}")
        print(f"   Output: {output_pkl}")
        print(f"   Method: {smoothing_method}")
        
        try:
            # Initialize pipeline
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipeline = PostProcessingSmoothingPipeline(
                smplx_path="models/smplx",
                device=device,
                gender='neutral'
            )
            
            # Run pipeline
            success = pipeline.apply_post_processing_smoothing(
                input_pkl_path=pkl_file,
                output_pkl_path=output_pkl,
                smoothing_method=smoothing_method,
                stabilize_shape=True,
                quality_assessment=True
            )
            
            if success:
                print(f"\n🏆 PIPELINE COMPLETED SUCCESSFULLY!")
                print(f"   Smoothed PKL: {output_pkl}")
            else:
                print(f"\n💥 PIPELINE FAILED!")
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        # Test mode - parameter extraction only
        print(f"🧪 Testing parameter extraction and smoothing algorithms")
        
        try:
            # Test parameter extraction
            extractor = PKLParameterExtractor()
            
            # Load PKL data
            data, mesh_sequence, metadata = extractor.load_pkl_data(pkl_file)
            
            # Extract parameters
            extracted_params = extractor.extract_parameters_from_sequence(mesh_sequence)
            
            # Convert to numpy arrays
            numpy_params = extractor.convert_to_numpy_arrays(extracted_params)
            
            # Test basic smoothing
            original_weights = metadata.get('original_temporal_weights', None)
            smoother = BasicTemporalSmoother(original_weights)
            
            print("\n🧪 Testing basic smoothing...")
            smoothed_basic = smoother.apply_moving_average_smoothing(numpy_params)
            
            print("\n🧪 Testing Savitzky-Golay smoothing...")
            smoothed_savgol = smoother.apply_savgol_smoothing(numpy_params)
            
            # Test advanced smoothing
            advanced_smoother = AdvancedTemporalSmoother(original_weights)
            
            print("\n🧪 Testing bilateral smoothing...")
            smoothed_bilateral = advanced_smoother.apply_bilateral_smoothing(numpy_params)
            
            print("\n🧪 Testing shape stabilization...")
            stabilized = advanced_smoother.stabilize_shape_parameters(numpy_params)
            
            print("\n✅ All smoothing tests completed successfully!")
            print(f"   To run full pipeline: python {sys.argv[0]} {pkl_file} output_smoothed.pkl {smoothing_method}")
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Parameter Optimizer for Post-Processing Smoothing
Optimize smoothing parameters to achieve maximum similarity to original temporal smoothing
Uses grid search and Bayesian optimization to find optimal parameters
"""

import os
import sys
import time
import numpy as np
import pickle
from pathlib import Path
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from post_processing_smoothing import (
    PKLParameterExtractor,
    BasicTemporalSmoother,
    AdvancedTemporalSmoother,
    OutlierDetector
)
from comparison_tools import PKLComparator


class SmoothingParameterOptimizer:
    """Optimize smoothing parameters for maximum similarity to original temporal smoothing"""
    
    def __init__(self, reference_pkl_path):
        self.reference_pkl_path = reference_pkl_path
        self.extractor = PKLParameterExtractor()
        self.comparator = PKLComparator()
        
        # Load reference (serial) parameters
        print(f"📂 Loading reference PKL: {Path(reference_pkl_path).name}")
        self.reference_params, self.reference_metadata, _ = self.extractor.load_and_extract_params(reference_pkl_path)
        
        if self.reference_params is None:
            raise ValueError(f"Failed to load reference PKL: {reference_pkl_path}")
        
        print(f"✅ Loaded reference with {self.reference_params['body_pose'].shape[0]} frames")
        
        # Extract original temporal weights for guidance
        self.original_temporal_weights = self.reference_metadata.get('original_temporal_weights', {
            'body_pose': 0.3,
            'betas': 0.03,
            'global_orient': 0.0,
            'transl': 0.0
        })
        
        print(f"🎯 Reference temporal weights: {self.original_temporal_weights}")
    
    def create_synthetic_noisy_data(self, noise_level=0.1, outlier_rate=0.05):
        """Create synthetic noisy version of reference data (simulating parallel processing)"""
        
        print(f"🧪 Creating synthetic noisy data (noise={noise_level}, outliers={outlier_rate})")
        
        synthetic_params = {}
        
        for param_name, param_data in self.reference_params.items():
            noisy_data = param_data.copy()
            n_frames, param_dim = param_data.shape
            
            # Add random noise
            if param_name == 'body_pose':
                noise_std = noise_level * 0.2  # Joint angles in radians
            elif param_name == 'betas':
                noise_std = noise_level * 0.1  # Shape parameters
            else:
                noise_std = noise_level * 0.05  # Global orient/transl
            
            noise = np.random.normal(0, noise_std, param_data.shape)
            noisy_data += noise
            
            # Add outliers
            n_outliers = int(n_frames * outlier_rate)
            if n_outliers > 0:
                outlier_frames = np.random.choice(n_frames, n_outliers, replace=False)
                outlier_strength = noise_level * 5  # Strong outliers
                
                for frame_idx in outlier_frames:
                    outlier_noise = np.random.normal(0, outlier_strength, param_dim)
                    noisy_data[frame_idx] += outlier_noise
            
            synthetic_params[param_name] = noisy_data
        
        print(f"✅ Created synthetic noisy data with {n_outliers} outliers per parameter")
        return synthetic_params
    
    def evaluate_smoothing_config(self, config, synthetic_noisy_params):
        """Evaluate a specific smoothing configuration"""
        
        try:
            # Initialize components with config
            outlier_detector = OutlierDetector(outlier_threshold=config['outlier_threshold'])
            
            # Step 1: Outlier correction
            corrected_params = outlier_detector.detect_and_correct_outliers(
                synthetic_noisy_params, method=config['outlier_method']
            )
            
            # Step 2: Apply smoothing based on method
            if config['smoothing_method'] == 'moving_average':
                smoother = BasicTemporalSmoother(self.original_temporal_weights)
                smoothed_params = smoother.apply_moving_average_smoothing(
                    corrected_params, window_sizes=config['window_sizes']
                )
            elif config['smoothing_method'] == 'savgol':
                smoother = BasicTemporalSmoother(self.original_temporal_weights)
                smoothed_params = smoother.apply_savgol_smoothing(
                    corrected_params, 
                    window_lengths=config['window_sizes'],
                    polyorder=config.get('polyorder', 2)
                )
            elif config['smoothing_method'] == 'bilateral':
                smoother = AdvancedTemporalSmoother(self.original_temporal_weights)
                smoothed_params = smoother.apply_bilateral_smoothing(
                    corrected_params,
                    spatial_sigmas=config['spatial_sigmas'],
                    temporal_sigmas=config['temporal_sigmas']
                )
            else:
                return 0.0  # Invalid method
            
            # Step 3: Shape stabilization
            if config['stabilize_shape']:
                smoothed_params = smoother.stabilize_shape_parameters(
                    smoothed_params, method=config['shape_method']
                )
            
            # Step 4: Calculate similarity to reference
            similarity_score = self._calculate_similarity_score(self.reference_params, smoothed_params)
            
            return similarity_score
            
        except Exception as e:
            print(f"⚠️  Config evaluation failed: {e}")
            return 0.0  # Return poor score for failed configs
    
    def _calculate_similarity_score(self, reference_params, candidate_params):
        """Calculate similarity score between reference and candidate parameters"""
        
        parameter_weights = {
            'body_pose': 0.5,      # Most important
            'betas': 0.3,          # Shape consistency  
            'global_orient': 0.1,  # Less important
            'transl': 0.1          # Less important
        }
        
        weighted_scores = []
        total_weight = 0
        
        for param_name, weight in parameter_weights.items():
            if param_name not in reference_params or param_name not in candidate_params:
                continue
            
            ref_data = reference_params[param_name]
            cand_data = candidate_params[param_name]
            
            # Ensure same shape
            min_frames = min(ref_data.shape[0], cand_data.shape[0])
            ref_data = ref_data[:min_frames]
            cand_data = cand_data[:min_frames]
            
            # Calculate multiple similarity metrics
            mse = np.mean((ref_data - cand_data)**2)
            param_range = np.ptp(ref_data)
            
            if param_range > 0:
                normalized_mse = mse / (param_range**2)
                similarity = max(0.0, 1.0 - normalized_mse)
            else:
                similarity = 1.0 if mse < 1e-6 else 0.0
            
            # Temporal smoothness matching
            ref_diff = np.diff(ref_data, axis=0)
            cand_diff = np.diff(cand_data, axis=0)
            
            ref_smoothness = np.mean(np.abs(ref_diff))
            cand_smoothness = np.mean(np.abs(cand_diff))
            
            if ref_smoothness > 0:
                temporal_match = 1.0 - abs(ref_smoothness - cand_smoothness) / ref_smoothness
            else:
                temporal_match = 1.0
            
            # Combined score
            param_score = similarity * 0.7 + temporal_match * 0.3
            
            weighted_scores.append(param_score * weight)
            total_weight += weight
        
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.0
        
        return overall_score
    
    def grid_search_optimization(self, synthetic_noisy_params):
        """Perform grid search to find optimal parameters"""
        
        print(f"\n🔍 STARTING GRID SEARCH OPTIMIZATION")
        print("-" * 50)
        
        # Define parameter search spaces
        search_space = {
            'smoothing_method': ['bilateral', 'savgol', 'moving_average'],
            'outlier_threshold': [2.5, 3.0, 3.5],
            'outlier_method': ['interpolate', 'median_filter'],
            'stabilize_shape': [True, False],
            'shape_method': ['heavy_smooth', 'median']
        }
        
        # Method-specific parameters
        bilateral_params = {
            'spatial_sigmas': [
                {'body_pose': 1.5, 'betas': 0.3, 'global_orient': 1.0, 'transl': 1.0},
                {'body_pose': 2.0, 'betas': 0.5, 'global_orient': 1.5, 'transl': 1.5},
                {'body_pose': 2.5, 'betas': 0.8, 'global_orient': 2.0, 'transl': 2.0}
            ],
            'temporal_sigmas': [
                {'body_pose': 0.2, 'betas': 0.05, 'global_orient': 0.15, 'transl': 0.15},
                {'body_pose': 0.3, 'betas': 0.1, 'global_orient': 0.2, 'transl': 0.2},
                {'body_pose': 0.4, 'betas': 0.15, 'global_orient': 0.25, 'transl': 0.25}
            ]
        }
        
        window_sizes_options = [
            {'body_pose': 5, 'betas': 15, 'global_orient': 3, 'transl': 3},
            {'body_pose': 7, 'betas': 21, 'global_orient': 5, 'transl': 5},
            {'body_pose': 9, 'betas': 31, 'global_orient': 7, 'transl': 7}
        ]
        
        best_score = 0.0
        best_config = None
        all_results = []
        
        # Generate all combinations (reduced for efficiency)
        total_configs = 0
        
        for method in search_space['smoothing_method']:
            for outlier_thresh in search_space['outlier_threshold']:
                for outlier_method in search_space['outlier_method']:
                    for stabilize in search_space['stabilize_shape']:
                        for shape_method in search_space['shape_method']:
                            
                            if method == 'bilateral':
                                for spatial_sig in bilateral_params['spatial_sigmas']:
                                    for temporal_sig in bilateral_params['temporal_sigmas']:
                                        config = {
                                            'smoothing_method': method,
                                            'outlier_threshold': outlier_thresh,
                                            'outlier_method': outlier_method,
                                            'stabilize_shape': stabilize,
                                            'shape_method': shape_method,
                                            'spatial_sigmas': spatial_sig,
                                            'temporal_sigmas': temporal_sig
                                        }
                                        total_configs += 1
                                        
                                        score = self.evaluate_smoothing_config(config, synthetic_noisy_params)
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_config = config.copy()
                                        
                                        all_results.append((config.copy(), score))
                                        
                                        if total_configs % 10 == 0:
                                            print(f"   Evaluated {total_configs} configs, best: {best_score:.4f}")
                            
                            else:  # savgol or moving_average
                                for window_sizes in window_sizes_options:
                                    config = {
                                        'smoothing_method': method,
                                        'outlier_threshold': outlier_thresh,
                                        'outlier_method': outlier_method,
                                        'stabilize_shape': stabilize,
                                        'shape_method': shape_method,
                                        'window_sizes': window_sizes
                                    }
                                    
                                    if method == 'savgol':
                                        config['polyorder'] = 2
                                    
                                    total_configs += 1
                                    
                                    score = self.evaluate_smoothing_config(config, synthetic_noisy_params)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_config = config.copy()
                                    
                                    all_results.append((config.copy(), score))
                                    
                                    if total_configs % 10 == 0:
                                        print(f"   Evaluated {total_configs} configs, best: {best_score:.4f}")
        
        print(f"\n✅ Grid search complete!")
        print(f"   Total configurations: {total_configs}")
        print(f"   Best score: {best_score:.4f}")
        
        # Sort results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return best_config, best_score, all_results
    
    def fine_tune_best_config(self, best_config, synthetic_noisy_params):
        """Fine-tune the best configuration found by grid search"""
        
        print(f"\n🎯 FINE-TUNING BEST CONFIGURATION")
        print("-" * 50)
        print(f"Starting config score: {self.evaluate_smoothing_config(best_config, synthetic_noisy_params):.4f}")
        
        if best_config['smoothing_method'] == 'bilateral':
            # Fine-tune bilateral parameters
            return self._fine_tune_bilateral(best_config, synthetic_noisy_params)
        else:
            # Fine-tune window-based methods
            return self._fine_tune_window_based(best_config, synthetic_noisy_params)
    
    def _fine_tune_bilateral(self, config, synthetic_noisy_params):
        """Fine-tune bilateral smoothing parameters"""
        
        best_config = config.copy()
        best_score = self.evaluate_smoothing_config(best_config, synthetic_noisy_params)
        
        # Parameters to fine-tune
        params_to_tune = ['body_pose', 'betas']  # Focus on most important
        
        for param_name in params_to_tune:
            print(f"   Fine-tuning {param_name} parameters...")
            
            # Current values
            current_spatial = best_config['spatial_sigmas'][param_name]
            current_temporal = best_config['temporal_sigmas'][param_name]
            
            # Search around current values
            spatial_range = [current_spatial * 0.8, current_spatial * 0.9, current_spatial,
                           current_spatial * 1.1, current_spatial * 1.2]
            temporal_range = [current_temporal * 0.8, current_temporal * 0.9, current_temporal,
                            current_temporal * 1.1, current_temporal * 1.2]
            
            for spatial_val in spatial_range:
                for temporal_val in temporal_range:
                    test_config = best_config.copy()
                    test_config['spatial_sigmas'] = best_config['spatial_sigmas'].copy()
                    test_config['temporal_sigmas'] = best_config['temporal_sigmas'].copy()
                    test_config['spatial_sigmas'][param_name] = spatial_val
                    test_config['temporal_sigmas'][param_name] = temporal_val
                    
                    score = self.evaluate_smoothing_config(test_config, synthetic_noisy_params)
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        print(f"     Improved: {param_name} spatial={spatial_val:.3f}, temporal={temporal_val:.3f}, score={score:.4f}")
        
        print(f"✅ Fine-tuning complete, final score: {best_score:.4f}")
        return best_config, best_score
    
    def _fine_tune_window_based(self, config, synthetic_noisy_params):
        """Fine-tune window-based smoothing parameters"""
        
        best_config = config.copy()
        best_score = self.evaluate_smoothing_config(best_config, synthetic_noisy_params)
        
        # Parameters to fine-tune
        params_to_tune = ['body_pose', 'betas']
        
        for param_name in params_to_tune:
            print(f"   Fine-tuning {param_name} window size...")
            
            current_window = best_config['window_sizes'][param_name]
            
            # Search around current window size
            window_range = [max(3, current_window - 4), max(3, current_window - 2), current_window,
                          current_window + 2, current_window + 4]
            
            for window_size in window_range:
                test_config = best_config.copy()
                test_config['window_sizes'] = best_config['window_sizes'].copy()
                test_config['window_sizes'][param_name] = window_size
                
                score = self.evaluate_smoothing_config(test_config, synthetic_noisy_params)
                
                if score > best_score:
                    best_score = score
                    best_config = test_config.copy()
                    print(f"     Improved: {param_name} window={window_size}, score={score:.4f}")
        
        print(f"✅ Fine-tuning complete, final score: {best_score:.4f}")
        return best_config, best_score
    
    def save_optimization_results(self, best_config, best_score, all_results, output_path):
        """Save optimization results"""
        
        optimization_results = {
            'optimization_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reference_pkl': str(self.reference_pkl_path),
            'original_temporal_weights': self.original_temporal_weights,
            'best_configuration': best_config,
            'best_similarity_score': best_score,
            'total_configurations_tested': len(all_results),
            'top_10_configurations': all_results[:10] if len(all_results) >= 10 else all_results,
            'optimization_summary': {
                'method': best_config.get('smoothing_method', 'unknown'),
                'outlier_threshold': best_config.get('outlier_threshold', 0),
                'stabilize_shape': best_config.get('stabilize_shape', False),
                'expected_similarity': f"{best_score:.1%}"
            }
        }
        
        print(f"\n💾 Saving optimization results: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = str(output_path).replace('.json', '_summary.txt')
        self._save_readable_summary(optimization_results, summary_path)
        
        print(f"✅ Results saved: {output_path}")
        print(f"✅ Summary saved: {summary_path}")
    
    def _save_readable_summary(self, results, summary_path):
        """Save human-readable optimization summary"""
        
        with open(summary_path, 'w') as f:
            f.write("SMOOTHING PARAMETER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Optimization Date: {results['optimization_date']}\n")
            f.write(f"Reference PKL: {Path(results['reference_pkl']).name}\n")
            f.write(f"Configurations Tested: {results['total_configurations_tested']}\n\n")
            
            f.write(f"BEST CONFIGURATION (Score: {results['best_similarity_score']:.4f}):\n")
            f.write("-" * 30 + "\n")
            
            best_config = results['best_configuration']
            for key, value in best_config.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nTOP 5 CONFIGURATIONS:\n")
            f.write("-" * 20 + "\n")
            
            for i, (config, score) in enumerate(results['top_10_configurations'][:5]):
                f.write(f"\n{i+1}. Score: {score:.4f}\n")
                f.write(f"   Method: {config.get('smoothing_method', 'N/A')}\n")
                f.write(f"   Outlier Threshold: {config.get('outlier_threshold', 'N/A')}\n")
                f.write(f"   Shape Stabilization: {config.get('stabilize_shape', 'N/A')}\n")


def main():
    """Command line interface for parameter optimization"""
    
    if len(sys.argv) < 2:
        print("Usage: python parameter_optimizer.py <reference_pkl> [output_results.json]")
        print("Example: python parameter_optimizer.py original_serial.pkl optimization_results.json")
        return
    
    reference_pkl = sys.argv[1]
    output_results = sys.argv[2] if len(sys.argv) > 2 else "optimization_results.json"
    
    print("🎯 SMOOTHING PARAMETER OPTIMIZER")
    print("=" * 60)
    
    try:
        # Initialize optimizer
        optimizer = SmoothingParameterOptimizer(reference_pkl)
        
        # Create synthetic noisy data for optimization
        synthetic_noisy_params = optimizer.create_synthetic_noisy_data(
            noise_level=0.1, outlier_rate=0.05
        )
        
        # Run grid search optimization
        best_config, best_score, all_results = optimizer.grid_search_optimization(synthetic_noisy_params)
        
        if best_config is None:
            print("❌ Optimization failed")
            return
        
        # Fine-tune best configuration
        best_config, best_score = optimizer.fine_tune_best_config(best_config, synthetic_noisy_params)
        
        # Save results
        optimizer.save_optimization_results(best_config, best_score, all_results, output_results)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"   Best similarity score: {best_score:.4f} ({best_score:.1%})")
        print(f"   Best method: {best_config.get('smoothing_method', 'unknown')}")
        print(f"   Results saved: {output_results}")
        
        if best_score > 0.95:
            print("🎉 EXCELLENT: Near-perfect similarity achieved!")
        elif best_score > 0.90:
            print("✅ SUCCESS: High similarity achieved!")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Consider additional tuning")
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comparison Tools for Quality Assessment
Compare serial temporal smoothing vs parallel + post-processing smoothing
Goal: Quantify similarity to achieve maximum identity
"""

import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from post_processing_smoothing import PKLParameterExtractor


class PKLComparator:
    """Compare two PKL files (serial vs parallel+post-processed)"""
    
    def __init__(self):
        self.extractor = PKLParameterExtractor()
        self.comparison_metrics = {}
        
    def load_and_extract_params(self, pkl_path):
        """Load PKL and extract parameters for comparison"""
        
        print(f"📂 Loading PKL: {Path(pkl_path).name}")
        
        try:
            data, mesh_sequence, metadata = self.extractor.load_pkl_data(pkl_path)
            extracted_params = self.extractor.extract_parameters_from_sequence(mesh_sequence)
            numpy_params = self.extractor.convert_to_numpy_arrays(extracted_params)
            
            return numpy_params, metadata, mesh_sequence
            
        except Exception as e:
            print(f"❌ Error loading {pkl_path}: {e}")
            return None, None, None
    
    def compare_pkl_files(self, serial_pkl_path, parallel_pkl_path, output_report_path=None):
        """Compare serial vs parallel+post-processed PKL files"""
        
        print("🔍 COMPARING SERIAL vs PARALLEL+POST-PROCESSED PKL FILES")
        print("=" * 70)
        
        # Load both PKL files
        serial_params, serial_metadata, serial_sequence = self.load_and_extract_params(serial_pkl_path)
        parallel_params, parallel_metadata, parallel_sequence = self.load_and_extract_params(parallel_pkl_path)
        
        if serial_params is None or parallel_params is None:
            print("❌ Failed to load one or both PKL files")
            return None
        
        # Validate frame counts
        serial_frames = len(serial_sequence)
        parallel_frames = len(parallel_sequence)
        
        print(f"📊 Frame counts: Serial={serial_frames}, Parallel={parallel_frames}")
        
        if serial_frames != parallel_frames:
            print(f"⚠️  Frame count mismatch - truncating to {min(serial_frames, parallel_frames)} frames")
            min_frames = min(serial_frames, parallel_frames)
            
            # Truncate parameters
            for param_name in serial_params.keys():
                serial_params[param_name] = serial_params[param_name][:min_frames]
                parallel_params[param_name] = parallel_params[param_name][:min_frames]
        
        # Perform detailed comparison
        print(f"\n📈 DETAILED PARAMETER COMPARISON")
        print("-" * 50)
        
        comparison_results = {}
        
        for param_name in serial_params.keys():
            print(f"\n🔍 Analyzing {param_name}...")
            
            serial_data = serial_params[param_name]
            parallel_data = parallel_params[param_name]
            
            # Calculate comprehensive metrics
            metrics = self._calculate_parameter_metrics(serial_data, parallel_data, param_name)
            comparison_results[param_name] = metrics
            
            # Print key metrics
            print(f"   Similarity Score: {metrics['similarity_score']:.3f} (0.0-1.0, higher=better)")
            print(f"   MSE: {metrics['mse']:.6f}")
            print(f"   MAE: {metrics['mae']:.6f}")
            print(f"   Correlation: {metrics['correlation']:.4f}")
            print(f"   Temporal Smoothness Match: {metrics['temporal_smoothness_match']:.3f}")
        
        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(comparison_results)
        
        print(f"\n🎯 OVERALL SIMILARITY ASSESSMENT")
        print("-" * 50)
        print(f"   Overall Similarity Score: {overall_similarity:.3f}")
        
        if overall_similarity > 0.95:
            print(f"   ✅ EXCELLENT: Nearly identical to serial version")
        elif overall_similarity > 0.90:
            print(f"   ✅ VERY GOOD: High similarity to serial version")
        elif overall_similarity > 0.80:
            print(f"   ⚠️  GOOD: Acceptable similarity, minor differences")
        elif overall_similarity > 0.70:
            print(f"   ⚠️  FAIR: Noticeable differences, may need tuning")
        else:
            print(f"   ❌ POOR: Significant differences, requires improvement")
        
        # Create detailed comparison report
        full_report = {
            'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'serial_pkl': str(serial_pkl_path),
            'parallel_pkl': str(parallel_pkl_path),
            'serial_metadata': serial_metadata,
            'parallel_metadata': parallel_metadata,
            'frame_counts': {
                'serial': serial_frames,
                'parallel': parallel_frames,
                'compared': min(serial_frames, parallel_frames)
            },
            'parameter_comparisons': comparison_results,
            'overall_similarity': overall_similarity,
            'recommendations': self._generate_recommendations(comparison_results, overall_similarity)
        }
        
        # Save report if requested
        if output_report_path:
            self._save_comparison_report(full_report, output_report_path)
        
        self.comparison_metrics = comparison_results
        return full_report
    
    def _calculate_parameter_metrics(self, serial_data, parallel_data, param_name):
        """Calculate comprehensive metrics for parameter comparison"""
        
        n_frames, param_dim = serial_data.shape
        
        # Basic difference metrics
        mse = mean_squared_error(serial_data, parallel_data)
        mae = mean_absolute_error(serial_data, parallel_data)
        rmse = np.sqrt(mse)
        
        # Correlation analysis
        flattened_serial = serial_data.flatten()
        flattened_parallel = parallel_data.flatten()
        correlation, correlation_p_value = pearsonr(flattened_serial, flattened_parallel)
        
        # Temporal smoothness comparison
        serial_diff = np.diff(serial_data, axis=0)
        parallel_diff = np.diff(parallel_data, axis=0)
        
        serial_smoothness = np.mean(np.abs(serial_diff))
        parallel_smoothness = np.mean(np.abs(parallel_diff))
        
        if serial_smoothness > 0:
            temporal_smoothness_match = 1.0 - abs(serial_smoothness - parallel_smoothness) / serial_smoothness
        else:
            temporal_smoothness_match = 1.0
        
        # Parameter-specific similarity score
        # Normalize by parameter range for fair comparison
        param_range = np.ptp(serial_data)  # Peak-to-peak range
        if param_range > 0:
            normalized_mae = mae / param_range
            similarity_score = max(0.0, 1.0 - normalized_mae)
        else:
            similarity_score = 1.0 if mae < 1e-6 else 0.0
        
        # Frame-by-frame analysis
        frame_errors = []
        for frame_idx in range(n_frames):
            frame_error = np.linalg.norm(serial_data[frame_idx] - parallel_data[frame_idx])
            frame_errors.append(frame_error)
        
        frame_errors = np.array(frame_errors)
        worst_frame_idx = np.argmax(frame_errors)
        best_frame_idx = np.argmin(frame_errors)
        
        # Variance comparison
        serial_var = np.var(serial_data, axis=0).mean()
        parallel_var = np.var(parallel_data, axis=0).mean()
        variance_ratio = parallel_var / serial_var if serial_var > 0 else 1.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'correlation_p_value': correlation_p_value,
            'similarity_score': similarity_score,
            'temporal_smoothness_match': temporal_smoothness_match,
            'serial_smoothness': serial_smoothness,
            'parallel_smoothness': parallel_smoothness,
            'frame_errors': {
                'mean': np.mean(frame_errors),
                'std': np.std(frame_errors),
                'max': np.max(frame_errors),
                'min': np.min(frame_errors),
                'worst_frame': worst_frame_idx,
                'best_frame': best_frame_idx
            },
            'variance_comparison': {
                'serial_variance': serial_var,
                'parallel_variance': parallel_var,
                'variance_ratio': variance_ratio
            },
            'parameter_range': param_range,
            'n_frames': n_frames,
            'param_dim': param_dim
        }
    
    def _calculate_overall_similarity(self, comparison_results):
        """Calculate weighted overall similarity score"""
        
        # Weights for different parameter types (based on importance in original temporal loss)
        parameter_weights = {
            'body_pose': 0.5,      # Most important (was temporal_alpha * 1.0)
            'betas': 0.3,          # Important for consistent shape
            'global_orient': 0.1,  # Less important (wasn't in original temporal loss)
            'transl': 0.1          # Less important (wasn't in original temporal loss)
        }
        
        weighted_scores = []
        total_weight = 0
        
        for param_name, metrics in comparison_results.items():
            weight = parameter_weights.get(param_name, 0.1)  # Default weight
            similarity = metrics['similarity_score']
            correlation = max(0, metrics['correlation'])  # Ensure non-negative
            temporal_match = metrics['temporal_smoothness_match']
            
            # Combined score: similarity, correlation, and temporal matching
            combined_score = (similarity * 0.5 + correlation * 0.3 + temporal_match * 0.2)
            
            weighted_scores.append(combined_score * weight)
            total_weight += weight
        
        if total_weight > 0:
            overall_similarity = sum(weighted_scores) / total_weight
        else:
            overall_similarity = 0.0
        
        return overall_similarity
    
    def _generate_recommendations(self, comparison_results, overall_similarity):
        """Generate recommendations for improving similarity"""
        
        recommendations = []
        
        if overall_similarity < 0.90:
            recommendations.append("Overall similarity below 90% - consider parameter tuning")
        
        for param_name, metrics in comparison_results.items():
            similarity = metrics['similarity_score']
            correlation = metrics['correlation']
            temporal_match = metrics['temporal_smoothness_match']
            
            if similarity < 0.85:
                recommendations.append(f"{param_name}: Low similarity ({similarity:.3f}) - check smoothing strength")
            
            if correlation < 0.90:
                recommendations.append(f"{param_name}: Low correlation ({correlation:.3f}) - values may be offset or scaled")
            
            if temporal_match < 0.80:
                recommendations.append(f"{param_name}: Poor temporal matching ({temporal_match:.3f}) - adjust smoothing window size")
            
            # Parameter-specific recommendations
            if param_name == 'body_pose' and similarity < 0.90:
                recommendations.append("body_pose: Consider increasing bilateral smoothing spatial_sigma")
            
            if param_name == 'betas' and metrics['variance_comparison']['variance_ratio'] > 2.0:
                recommendations.append("betas: Shape parameters too variable - use stronger stabilization")
            
            # Check for problematic frames
            frame_errors = metrics['frame_errors']
            if frame_errors['max'] > frame_errors['mean'] * 5:
                recommendations.append(f"{param_name}: Frame {frame_errors['worst_frame']} has high error - check for outliers")
        
        if not recommendations:
            recommendations.append("Excellent similarity - no improvements needed")
        
        return recommendations
    
    def _save_comparison_report(self, report, output_path):
        """Save detailed comparison report"""
        
        print(f"\n💾 Saving comparison report: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also save human-readable summary
            summary_path = str(output_path).replace('.json', '_summary.txt')
            self._save_human_readable_summary(report, summary_path)
            
            print(f"✅ Report saved: {output_path}")
            print(f"✅ Summary saved: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def _save_human_readable_summary(self, report, summary_path):
        """Save human-readable summary report"""
        
        with open(summary_path, 'w') as f:
            f.write("PKL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Comparison Date: {report['comparison_date']}\n")
            f.write(f"Serial PKL: {Path(report['serial_pkl']).name}\n")
            f.write(f"Parallel PKL: {Path(report['parallel_pkl']).name}\n")
            f.write(f"Frames Compared: {report['frame_counts']['compared']}\n\n")
            
            f.write(f"OVERALL SIMILARITY: {report['overall_similarity']:.3f}\n")
            f.write("-" * 30 + "\n")
            
            # Parameter-by-parameter breakdown
            for param_name, metrics in report['parameter_comparisons'].items():
                f.write(f"\n{param_name.upper()}:\n")
                f.write(f"  Similarity Score: {metrics['similarity_score']:.3f}\n")
                f.write(f"  Correlation: {metrics['correlation']:.4f}\n")
                f.write(f"  MSE: {metrics['mse']:.6f}\n")
                f.write(f"  MAE: {metrics['mae']:.6f}\n")
                f.write(f"  Temporal Match: {metrics['temporal_smoothness_match']:.3f}\n")
                f.write(f"  Worst Frame: {metrics['frame_errors']['worst_frame']}\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")


class VisualizationTools:
    """Create visualizations for parameter comparisons"""
    
    def __init__(self):
        plt.style.use('dark_background')
        
    def create_comparison_plots(self, serial_params, parallel_params, output_dir, param_name='body_pose'):
        """Create visualization plots comparing serial vs parallel parameters"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        serial_data = serial_params[param_name]
        parallel_data = parallel_params[param_name]
        
        n_frames, param_dim = serial_data.shape
        
        print(f"📊 Creating comparison plots for {param_name}...")
        
        # Plot 1: Time series comparison (first few dimensions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{param_name} - Serial vs Parallel Comparison', fontsize=16, color='white')
        
        dims_to_plot = min(4, param_dim)
        
        for i in range(dims_to_plot):
            ax = axes[i//2, i%2]
            
            ax.plot(serial_data[:, i], label='Serial (Original)', color='cyan', linewidth=2)
            ax.plot(parallel_data[:, i], label='Parallel+Post-proc', color='orange', linewidth=2)
            
            ax.set_title(f'{param_name} Dimension {i}', color='white')
            ax.set_xlabel('Frame', color='white')
            ax.set_ylabel('Value', color='white')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{param_name}_time_series.png', dpi=300, facecolor='black')
        plt.close()
        
        # Plot 2: Error analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{param_name} - Error Analysis', fontsize=16, color='white')
        
        # Frame-by-frame error
        frame_errors = np.linalg.norm(serial_data - parallel_data, axis=1)
        ax1.plot(frame_errors, color='red', linewidth=2)
        ax1.set_title('Frame-by-Frame Error (L2 Norm)', color='white')
        ax1.set_xlabel('Frame', color='white')
        ax1.set_ylabel('Error', color='white')
        ax1.grid(True, alpha=0.3)
        
        # Error histogram
        ax2.hist(frame_errors, bins=30, color='orange', alpha=0.7, edgecolor='white')
        ax2.set_title('Error Distribution', color='white')
        ax2.set_xlabel('Error', color='white')
        ax2.set_ylabel('Frequency', color='white')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{param_name}_error_analysis.png', dpi=300, facecolor='black')
        plt.close()
        
        # Plot 3: Correlation scatter plot (for first dimension)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.scatter(serial_data[:, 0], parallel_data[:, 0], alpha=0.6, color='cyan')
        
        # Perfect correlation line
        min_val = min(serial_data[:, 0].min(), parallel_data[:, 0].min())
        max_val = max(serial_data[:, 0].max(), parallel_data[:, 0].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
        
        ax.set_title(f'{param_name} Correlation (Dimension 0)', color='white', fontsize=16)
        ax.set_xlabel('Serial Values', color='white')
        ax.set_ylabel('Parallel Values', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{param_name}_correlation.png', dpi=300, facecolor='black')
        plt.close()
        
        print(f"✅ Plots saved in {output_dir}")


def main():
    """Command line interface for comparison tools"""
    
    if len(sys.argv) < 3:
        print("Usage: python comparison_tools.py <serial_pkl> <parallel_pkl> [output_report]")
        print("Example: python comparison_tools.py original.pkl smoothed.pkl comparison_report.json")
        return
    
    serial_pkl = sys.argv[1]
    parallel_pkl = sys.argv[2]
    output_report = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("🔍 PKL COMPARISON TOOL")
    print("=" * 50)
    
    try:
        # Initialize comparator
        comparator = PKLComparator()
        
        # Perform comparison
        comparison_report = comparator.compare_pkl_files(serial_pkl, parallel_pkl, output_report)
        
        if comparison_report is None:
            print("❌ Comparison failed")
            return
        
        # Create visualizations if both files loaded successfully
        if '--plot' in sys.argv:
            print(f"\n📊 Creating visualization plots...")
            
            serial_params, _, _ = comparator.load_and_extract_params(serial_pkl)
            parallel_params, _, _ = comparator.load_and_extract_params(parallel_pkl)
            
            if serial_params and parallel_params:
                viz = VisualizationTools()
                
                plot_dir = "comparison_plots"
                viz.create_comparison_plots(serial_params, parallel_params, plot_dir, 'body_pose')
                viz.create_comparison_plots(serial_params, parallel_params, plot_dir, 'betas')
        
        overall_similarity = comparison_report['overall_similarity']
        print(f"\n🎯 FINAL RESULT: {overall_similarity:.3f} similarity")
        
        if overall_similarity > 0.95:
            print("🏆 EXCELLENT: Implementation successful!")
        elif overall_similarity > 0.90:
            print("✅ SUCCESS: High similarity achieved!")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Consider parameter tuning")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
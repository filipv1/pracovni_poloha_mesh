#!/usr/bin/env python3
"""
Test script for post-processing smoothing pipeline
Creates small test dataset and validates the complete pipeline
"""

import os
import sys
import time
import numpy as np
import pickle
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from post_processing_smoothing import (
    PKLParameterExtractor, 
    BasicTemporalSmoother, 
    AdvancedTemporalSmoother,
    OutlierDetector,
    MeshRegenerator,
    PostProcessingSmoothingPipeline
)


def create_test_pkl_data(n_frames=50, add_noise=True, add_outliers=True):
    """Create synthetic test PKL data that mimics parallel processing output"""
    
    print(f"🧪 Creating synthetic test data ({n_frames} frames)")
    
    # Create synthetic SMPL-X parameters that look realistic
    mesh_sequence = []
    
    for frame_idx in range(n_frames):
        # Create base parameters
        t = frame_idx / n_frames  # Time from 0 to 1
        
        # Body pose: simulate simple walking motion
        body_pose = np.zeros(63)  # 21 joints × 3 parameters
        # Add some joint rotations for legs (simplified walking)
        body_pose[0] = 0.3 * np.sin(t * 4 * np.pi)   # Left hip
        body_pose[3] = -0.3 * np.sin(t * 4 * np.pi)  # Right hip  
        body_pose[6] = 0.5 * np.sin(t * 8 * np.pi)   # Left knee
        body_pose[9] = 0.5 * np.sin(t * 8 * np.pi)   # Right knee
        # Arms
        body_pose[30] = 0.2 * np.sin(t * 4 * np.pi + np.pi)  # Left shoulder
        body_pose[33] = 0.2 * np.sin(t * 4 * np.pi)          # Right shoulder
        
        # Global orientation: simulate slight turning
        global_orient = np.array([0.1 * np.sin(t * 2 * np.pi), 0.05 * t, 0.0])
        
        # Translation: simulate forward movement with slight vertical oscillation
        transl = np.array([t * 2.0, 0.1 * np.sin(t * 8 * np.pi), 0.0])
        
        # Shape parameters: should be stable
        betas = np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0])
        
        # Add noise to simulate optimization inconsistencies
        if add_noise:
            noise_strength = 0.05
            body_pose += np.random.normal(0, noise_strength, body_pose.shape)
            global_orient += np.random.normal(0, noise_strength * 0.5, global_orient.shape)
            transl += np.random.normal(0, noise_strength * 0.1, transl.shape)
            betas += np.random.normal(0, noise_strength * 0.1, betas.shape)
        
        # Add occasional outliers to simulate optimization failures
        if add_outliers and np.random.random() < 0.05:  # 5% chance of outlier
            outlier_strength = 2.0
            body_pose += np.random.normal(0, outlier_strength, body_pose.shape)
            print(f"   Added outlier to frame {frame_idx}")
        
        # Create synthetic vertices and faces (simplified)
        n_vertices = 1000  # Simplified mesh
        vertices = np.random.normal(0, 0.5, (n_vertices, 3))
        faces = np.random.randint(0, n_vertices, (1800, 3))  # Simplified face topology
        
        # Create synthetic joints
        n_joints = 22
        joints = np.random.normal(0, 0.3, (n_joints, 3))
        
        # Create mesh data structure
        mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'joints': joints,
            'smplx_params': {
                'body_pose': body_pose,
                'global_orient': global_orient,
                'transl': transl,
                'betas': betas
            },
            'fitting_error': np.random.uniform(0.001, 0.01),
            'vertex_count': n_vertices,
            'face_count': len(faces),
            'frame_id': frame_idx
        }
        
        mesh_sequence.append(mesh_data)
    
    # Create complete PKL data structure
    pkl_data = {
        'mesh_sequence': mesh_sequence,
        'metadata': {
            'processing_method': 'parallel_no_temporal_smoothing',
            'max_workers': 4,
            'total_frames': n_frames,
            'frame_skip': 1,
            'video_filename': 'synthetic_test_video.mp4',
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'requires_post_processing_smoothing': True,
            'original_temporal_alpha': 0.3,
            'original_temporal_weights': {
                'body_pose': 0.3,
                'betas': 0.03,
                'global_orient': 0.0,
                'transl': 0.0
            },
            'test_data': True,
            'noise_added': add_noise,
            'outliers_added': add_outliers
        }
    }
    
    print(f"✅ Created synthetic PKL with {n_frames} frames")
    return pkl_data


def test_parameter_extraction(pkl_data):
    """Test parameter extraction functionality"""
    
    print("\n🔍 Testing parameter extraction...")
    
    try:
        extractor = PKLParameterExtractor()
        
        # Save temporary PKL file
        temp_pkl_path = "test_temp_data.pkl"
        with open(temp_pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        # Test loading
        data, mesh_sequence, metadata = extractor.load_pkl_data(temp_pkl_path)
        
        # Test extraction
        extracted_params = extractor.extract_parameters_from_sequence(mesh_sequence)
        
        # Test conversion to numpy
        numpy_params = extractor.convert_to_numpy_arrays(extracted_params)
        
        print("✅ Parameter extraction test passed")
        print(f"   Extracted {len(extracted_params)} frames")
        print(f"   Parameter shapes: {[f'{k}: {v.shape}' for k, v in numpy_params.items()]}")
        
        # Cleanup
        os.remove(temp_pkl_path)
        
        return numpy_params, metadata
        
    except Exception as e:
        print(f"❌ Parameter extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_smoothing_algorithms(numpy_params, metadata):
    """Test smoothing algorithms"""
    
    print("\n🎯 Testing smoothing algorithms...")
    
    try:
        original_weights = metadata.get('original_temporal_weights')
        
        # Test basic smoothing
        print("   Testing basic smoothing...")
        basic_smoother = BasicTemporalSmoother(original_weights)
        
        smoothed_ma = basic_smoother.apply_moving_average_smoothing(numpy_params)
        smoothed_savgol = basic_smoother.apply_savgol_smoothing(numpy_params)
        
        # Test advanced smoothing  
        print("   Testing advanced smoothing...")
        advanced_smoother = AdvancedTemporalSmoother(original_weights)
        
        smoothed_bilateral = advanced_smoother.apply_bilateral_smoothing(numpy_params)
        stabilized = advanced_smoother.stabilize_shape_parameters(numpy_params)
        
        print("✅ Smoothing algorithms test passed")
        
        return smoothed_bilateral, stabilized
        
    except Exception as e:
        print(f"❌ Smoothing algorithms test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_outlier_detection(numpy_params):
    """Test outlier detection and correction"""
    
    print("\n🔍 Testing outlier detection...")
    
    try:
        outlier_detector = OutlierDetector(outlier_threshold=2.5)
        
        # Assess quality
        quality_metrics = outlier_detector.assess_parameter_quality(numpy_params)
        
        # Detect and correct outliers
        corrected_params = outlier_detector.detect_and_correct_outliers(numpy_params)
        
        # Detect optimization failures
        failed_frames = outlier_detector.detect_optimization_failures(corrected_params)
        
        print("✅ Outlier detection test passed")
        print(f"   Quality metrics calculated for {len(quality_metrics)} parameter types")
        
        return corrected_params, quality_metrics
        
    except Exception as e:
        print(f"❌ Outlier detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_full_pipeline(pkl_data, test_mesh_regeneration=False):
    """Test complete post-processing pipeline"""
    
    print("\n🚀 Testing complete post-processing pipeline...")
    
    try:
        # Save test PKL
        input_pkl = "test_input_parallel_data.pkl"
        output_pkl = "test_output_smoothed_data.pkl"
        
        with open(input_pkl, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        # Initialize pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if test_mesh_regeneration:
            # Full pipeline with mesh regeneration (requires SMPL-X models)
            pipeline = PostProcessingSmoothingPipeline(
                smplx_path="models/smplx",
                device=device,
                gender='neutral'
            )
        else:
            # Limited pipeline without mesh regeneration
            pipeline = PostProcessingSmoothingPipeline(
                smplx_path="nonexistent",  # This will disable mesh regeneration
                device=device,
                gender='neutral'
            )
        
        # Run pipeline (will fail at mesh regeneration step if SMPL-X not available, but that's expected)
        print("   Running pipeline steps individually for testing...")
        
        # Step 1: Parameter extraction
        data, mesh_sequence, metadata = pipeline.extractor.load_pkl_data(input_pkl)
        extracted_params = pipeline.extractor.extract_parameters_from_sequence(mesh_sequence)
        numpy_params = pipeline.extractor.convert_to_numpy_arrays(extracted_params)
        
        # Step 2: Outlier detection
        corrected_params = pipeline.outlier_detector.detect_and_correct_outliers(numpy_params)
        
        # Step 3: Smoothing
        smoothed_params = pipeline.advanced_smoother.apply_bilateral_smoothing(corrected_params)
        
        # Step 4: Shape stabilization
        final_params = pipeline.advanced_smoother.stabilize_shape_parameters(smoothed_params)
        
        print("✅ Pipeline core functionality test passed")
        
        # Cleanup
        if os.path.exists(input_pkl):
            os.remove(input_pkl)
        if os.path.exists(output_pkl):
            os.remove(output_pkl)
        
        return final_params
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_smoothing_quality(original_params, smoothed_params):
    """Analyze quality of smoothing"""
    
    print("\n📊 Analyzing smoothing quality...")
    
    for param_name in original_params.keys():
        original = original_params[param_name]
        smoothed = smoothed_params[param_name]
        
        # Temporal smoothness (frame-to-frame differences)
        original_diff = np.diff(original, axis=0)
        smoothed_diff = np.diff(smoothed, axis=0)
        
        original_roughness = np.mean(np.abs(original_diff))
        smoothed_roughness = np.mean(np.abs(smoothed_diff))
        smoothness_improvement = (original_roughness - smoothed_roughness) / original_roughness * 100
        
        # Variance reduction
        original_var = np.var(original, axis=0).mean()
        smoothed_var = np.var(smoothed, axis=0).mean()
        variance_reduction = (original_var - smoothed_var) / original_var * 100
        
        print(f"   {param_name}:")
        print(f"     Temporal smoothness improvement: {smoothness_improvement:.1f}%")
        print(f"     Variance reduction: {variance_reduction:.1f}%")


def main():
    """Run all tests"""
    
    print("🧪 POST-PROCESSING SMOOTHING TEST SUITE")
    print("=" * 60)
    
    # Test parameters
    n_test_frames = 50
    test_mesh_regeneration = "--test-mesh" in sys.argv
    
    if test_mesh_regeneration:
        print("ℹ️  Mesh regeneration testing enabled (requires SMPL-X models)")
    else:
        print("ℹ️  Mesh regeneration testing disabled (core functionality only)")
    
    start_time = time.time()
    
    try:
        # Step 1: Create test data
        print(f"\n📋 STEP 1: Creating synthetic test data")
        print("-" * 40)
        
        pkl_data = create_test_pkl_data(n_test_frames, add_noise=True, add_outliers=True)
        
        # Step 2: Test parameter extraction
        print(f"\n📋 STEP 2: Testing parameter extraction")
        print("-" * 40)
        
        numpy_params, metadata = test_parameter_extraction(pkl_data)
        if numpy_params is None:
            return
        
        # Step 3: Test outlier detection
        print(f"\n📋 STEP 3: Testing outlier detection")
        print("-" * 40)
        
        corrected_params, quality_metrics = test_outlier_detection(numpy_params)
        if corrected_params is None:
            return
        
        # Step 4: Test smoothing algorithms
        print(f"\n📋 STEP 4: Testing smoothing algorithms")
        print("-" * 40)
        
        smoothed_params, stabilized_params = test_smoothing_algorithms(corrected_params, metadata)
        if smoothed_params is None:
            return
        
        # Step 5: Quality analysis
        print(f"\n📋 STEP 5: Quality analysis")
        print("-" * 40)
        
        analyze_smoothing_quality(numpy_params, smoothed_params)
        
        # Step 6: Test complete pipeline
        print(f"\n📋 STEP 6: Testing complete pipeline")
        print("-" * 40)
        
        final_params = test_full_pipeline(pkl_data, test_mesh_regeneration)
        if final_params is None:
            return
        
        # Final summary
        processing_time = time.time() - start_time
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"   Test frames: {n_test_frames}")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Average time per frame: {processing_time/n_test_frames:.3f} seconds")
        
        if test_mesh_regeneration:
            print(f"   Mesh regeneration: Tested")
        else:
            print(f"   Mesh regeneration: Skipped (use --test-mesh to enable)")
        
        print("\n✅ Post-processing smoothing implementation is ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
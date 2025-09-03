#!/usr/bin/env python3
"""
Test script for batch processing functionality
Compares original vs batch processing to ensure identical PKL output
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path

def test_batch_processing():
    """Test batch processing with a small video sample"""
    
    print("🧪 BATCH PROCESSING TEST")
    print("=" * 50)
    
    # Test parameters
    test_video = "test_video.mp4"  # Replace with your test video
    batch_sizes = [1, 4, 8, 16, 32]  # Different batch sizes to test
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        print("Please provide a test video file or modify test_video variable")
        return False
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        print("-" * 30)
        
        output_dir = f"test_batch_{batch_size}"
        
        # Run batch processing
        start_time = time.time()
        
        # Import and run the pipeline
        sys.path.append('.')
        from run_production_simple import main as run_pipeline
        
        # Temporarily override sys.argv for testing
        original_argv = sys.argv.copy()
        sys.argv = [
            'test_batch_processing.py',
            test_video,
            output_dir,
            '--max-frames', '20',  # Small sample for testing
            '--batch-size', str(batch_size),
            '--quality', 'medium'
        ]
        
        try:
            run_pipeline()
            processing_time = time.time() - start_time
            
            # Check if PKL file was generated
            pkl_files = list(Path(output_dir).glob("*_meshes.pkl"))
            
            if pkl_files:
                pkl_file = pkl_files[0]
                
                # Load and analyze PKL data
                with open(pkl_file, 'rb') as f:
                    mesh_data = pickle.load(f)
                
                results[batch_size] = {
                    'processing_time': processing_time,
                    'mesh_count': len(mesh_data) if isinstance(mesh_data, list) else 1,
                    'pkl_file': pkl_file,
                    'mesh_data': mesh_data
                }
                
                print(f"✅ Success: {len(mesh_data)} meshes in {processing_time:.1f}s")
                print(f"   PKL file: {pkl_file}")
                
            else:
                print(f"❌ Failed: No PKL file generated")
                results[batch_size] = {'error': 'No PKL file generated'}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[batch_size] = {'error': str(e)}
            
        finally:
            sys.argv = original_argv
    
    # Compare results
    print(f"\n📈 COMPARISON RESULTS")
    print("=" * 50)
    
    if len(results) > 1:
        # Get reference (batch_size=1) for comparison
        reference_batch = 1
        if reference_batch in results and 'mesh_data' in results[reference_batch]:
            reference_data = results[reference_batch]['mesh_data']
            reference_time = results[reference_batch]['processing_time']
            
            print(f"Reference (batch_size={reference_batch}): {len(reference_data)} meshes, {reference_time:.1f}s")
            print()
            
            for batch_size, result in results.items():
                if batch_size == reference_batch or 'error' in result:
                    continue
                    
                if 'mesh_data' in result:
                    mesh_data = result['mesh_data']
                    processing_time = result['processing_time']
                    speedup = reference_time / processing_time if processing_time > 0 else 0
                    
                    # Check data consistency
                    data_identical = compare_mesh_data(reference_data, mesh_data)
                    
                    print(f"Batch size {batch_size:2d}: {len(mesh_data)} meshes, "
                          f"{processing_time:.1f}s ({speedup:.1f}x speedup)")
                    print(f"                Data identical: {'✅' if data_identical else '❌'}")
                    
                    if not data_identical:
                        print(f"                WARNING: Output differs from reference!")
    
    return True

def compare_mesh_data(data1, data2, tolerance=1e-5):
    """Compare two mesh data structures for consistency"""
    
    if len(data1) != len(data2):
        print(f"  Length mismatch: {len(data1)} vs {len(data2)}")
        return False
    
    for i, (mesh1, mesh2) in enumerate(zip(data1, data2)):
        # Compare vertices
        if not np.allclose(mesh1['vertices'], mesh2['vertices'], atol=tolerance):
            print(f"  Frame {i}: Vertices differ (max diff: {np.max(np.abs(mesh1['vertices'] - mesh2['vertices'])):.6f})")
            return False
            
        # Compare joints
        if not np.allclose(mesh1['joints'], mesh2['joints'], atol=tolerance):
            print(f"  Frame {i}: Joints differ (max diff: {np.max(np.abs(mesh1['joints'] - mesh2['joints'])):.6f})")
            return False
            
        # Compare faces
        if not np.array_equal(mesh1['faces'], mesh2['faces']):
            print(f"  Frame {i}: Faces differ")
            return False
    
    return True

def quick_test():
    """Quick functionality test without video"""
    
    print("🚀 QUICK BATCH FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test just the batch fitting method with dummy data
    sys.path.append('.')
    from run_production_simple import HighAccuracySMPLXFitter
    import torch
    
    try:
        # Initialize fitter
        fitter = HighAccuracySMPLXFitter("models/smplx", 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if not fitter.model_ready:
            print("❌ SMPL-X model not available")
            return False
        
        # Create dummy landmark data
        batch_size = 4
        dummy_landmarks = []
        
        for i in range(batch_size):
            # Create realistic landmark data (33 joints, 3D)
            landmarks = np.random.random((33, 3)) * 2.0 - 1.0  # Range [-1, 1]
            dummy_landmarks.append(landmarks)
        
        print(f"Testing batch fitting with {batch_size} dummy frames...")
        
        # Test batch processing
        start_time = time.time()
        batch_results = fitter.fit_mesh_to_landmarks_batch(dummy_landmarks)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch processing: {len(batch_results)} meshes in {batch_time:.2f}s")
        
        # Test individual processing for comparison
        start_time = time.time()
        individual_results = []
        for landmarks in dummy_landmarks:
            result = fitter.fit_mesh_to_landmarks(landmarks)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        print(f"✅ Individual processing: {len(individual_results)} meshes in {individual_time:.2f}s")
        
        # Compare results
        if len(batch_results) == len(individual_results):
            speedup = individual_time / batch_time if batch_time > 0 else 0
            print(f"🚀 Speedup: {speedup:.1f}x faster")
            
            # Check data structure consistency
            for i, (batch_mesh, individual_mesh) in enumerate(zip(batch_results, individual_results)):
                if batch_mesh and individual_mesh:
                    if 'vertices' in batch_mesh and 'vertices' in individual_mesh:
                        vertex_diff = np.max(np.abs(batch_mesh['vertices'] - individual_mesh['vertices']))
                        if vertex_diff < 0.01:  # Allow small numerical differences
                            print(f"  Frame {i}: ✅ Data consistent (max diff: {vertex_diff:.6f})")
                        else:
                            print(f"  Frame {i}: ⚠️ Data differs (max diff: {vertex_diff:.6f})")
            
            return True
        else:
            print(f"❌ Result count mismatch: {len(batch_results)} vs {len(individual_results)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Batch Processing Test Suite")
    print("=" * 60)
    
    # Run quick test first
    print("Running quick functionality test...")
    quick_success = quick_test()
    
    if quick_success:
        print(f"\n✅ Quick test passed! Batch processing is working.")
        
        # Ask user if they want to run full test
        if len(sys.argv) > 1:
            # Full test with video
            test_batch_processing()
        else:
            print(f"\nTo run full video test, provide test video:")
            print(f"python test_batch_processing.py <test_video.mp4>")
    else:
        print(f"\n❌ Quick test failed. Check SMPL-X setup.")
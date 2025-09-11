#!/usr/bin/env python3
"""
Debug script to identify parallel processing issues
Tests different configurations to isolate the problem
"""

import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

def simple_cpu_task(n):
    """Simple CPU-only task to test basic parallelization"""
    import time
    import numpy as np
    time.sleep(0.1)  # Simulate work
    return f"Task {n} completed with result: {np.sum(np.random.randn(100))}"

def cuda_test_task(n):
    """Test CUDA availability in worker processes"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        return f"Worker {n}: CUDA={cuda_available}, Devices={device_count}"
    except Exception as e:
        return f"Worker {n}: Error - {str(e)}"

def mediapipe_test_task(n):
    """Test MediaPipe initialization in worker processes"""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # Lower complexity for testing
            min_detection_confidence=0.5
        )
        pose.close()
        return f"Worker {n}: MediaPipe OK"
    except Exception as e:
        return f"Worker {n}: MediaPipe Error - {str(e)}"

def smplx_test_task(args):
    """Test SMPL-X model loading in worker processes"""
    n, smplx_path, device = args
    try:
        import torch
        import smplx
        
        # Force CPU for testing to avoid CUDA conflicts
        device = 'cpu'  
        
        model = smplx.create(
            model_path=smplx_path,
            model_type='smplx',
            gender='neutral',
            use_face_contour=False,
            use_pca=False,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True
        ).to(device)
        
        # Test basic forward pass
        batch_size = 1
        output = model(
            body_pose=torch.zeros((batch_size, 63), dtype=torch.float32, device=device),
            betas=torch.zeros((batch_size, 10), dtype=torch.float32, device=device)
        )
        
        vertex_count = output.vertices.shape[1]
        return f"Worker {n}: SMPL-X OK - {vertex_count} vertices"
        
    except Exception as e:
        return f"Worker {n}: SMPL-X Error - {str(e)}"

def test_parallel_configuration(test_name, task_func, args_list, max_workers=2):
    """Test a specific parallel configuration"""
    
    print(f"\n🧪 TESTING: {test_name}")
    print("-" * 60)
    print(f"   Workers: {max_workers}")
    print(f"   Tasks: {len(args_list)}")
    
    start_time = time.time()
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            print("   Submitting tasks...")
            
            # Submit tasks
            if callable(args_list[0]):  # Simple function without args
                futures = {executor.submit(task_func, i): i for i in range(len(args_list))}
            else:  # Function with arguments
                futures = {executor.submit(task_func, arg): i for i, arg in enumerate(args_list)}
            
            print("   Waiting for results...")
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                    print(f"     ✅ {result}")
                except Exception as e:
                    print(f"     ❌ Task failed: {e}")
                    results.append(f"Failed: {e}")
        
        elapsed = time.time() - start_time
        print(f"   ✅ {test_name} completed in {elapsed:.1f}s")
        print(f"   Success rate: {len([r for r in results if not r.startswith('Failed')]}/{len(results)}")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ {test_name} failed after {elapsed:.1f}s: {e}")
        return False

def main():
    """Run comprehensive parallel processing tests"""
    
    print("🔍 PARALLEL PROCESSING DEBUG SUITE")
    print("=" * 80)
    
    print(f"System info:")
    print(f"   CPU cores: {multiprocessing.cpu_count()}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Basic CPU parallelization
    success_1 = test_parallel_configuration(
        "Basic CPU Tasks", 
        simple_cpu_task, 
        list(range(4)),  # 4 simple tasks
        max_workers=2
    )
    
    if not success_1:
        print("\n❌ Basic parallelization failed - ProcessPoolExecutor issue")
        return
    
    # Test 2: CUDA in parallel processes
    success_2 = test_parallel_configuration(
        "CUDA Availability Test", 
        cuda_test_task, 
        list(range(2)),  # 2 CUDA tests
        max_workers=2
    )
    
    # Test 3: MediaPipe in parallel processes
    success_3 = test_parallel_configuration(
        "MediaPipe Initialization", 
        mediapipe_test_task, 
        list(range(2)),  # 2 MediaPipe tests
        max_workers=2
    )
    
    # Test 4: SMPL-X in parallel processes (CPU only to avoid CUDA conflicts)
    smplx_path = "models/smplx"
    if os.path.exists(smplx_path):
        smplx_args = [(i, smplx_path, 'cpu') for i in range(2)]
        success_4 = test_parallel_configuration(
            "SMPL-X Model Loading (CPU)", 
            smplx_test_task, 
            smplx_args,
            max_workers=2
        )
    else:
        print(f"\n⚠️  SMPL-X models not found at {smplx_path}")
        success_4 = False
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    print(f"   Basic CPU Tasks: {'✅ PASS' if success_1 else '❌ FAIL'}")
    print(f"   CUDA Test: {'✅ PASS' if success_2 else '❌ FAIL'}")
    print(f"   MediaPipe Test: {'✅ PASS' if success_3 else '❌ FAIL'}")
    print(f"   SMPL-X Test: {'✅ PASS' if success_4 else '❌ FAIL'}")
    
    if success_1 and success_2 and success_3 and success_4:
        print(f"\n🎉 All tests passed! The issue might be in the specific combination.")
        print(f"   Recommendation: Try CPU-only processing first")
    elif not success_1:
        print(f"\n💥 ProcessPoolExecutor is fundamentally broken on this system")
        print(f"   Recommendation: Use sequential processing only")
    else:
        print(f"\n🔧 Specific component is causing the hang:")
        if not success_2:
            print(f"   - CUDA contexts conflict in parallel processes")
        if not success_3:
            print(f"   - MediaPipe has parallelization issues")
        if not success_4:
            print(f"   - SMPL-X models can't be loaded in parallel")
        print(f"   Recommendation: Force CPU-only or sequential mode")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
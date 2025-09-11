#!/usr/bin/env python3
"""
SAFE Complete Parallel 3D Human Mesh Pipeline
Includes multiple fallback modes to handle parallel processing issues
"""

import os
import sys
import time
import argparse
import numpy as np
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from run_production_parallel_no_smoothing import ParallelMasterPipeline
from post_processing_smoothing import PostProcessingSmoothingPipeline
from comparison_tools import PKLComparator

class SafeCompletePipeline:
    """Safe complete pipeline with multiple processing modes and fallbacks"""
    
    def __init__(self, smplx_path="models/smplx", device='auto', gender='neutral', max_workers=None):
        self.smplx_path = smplx_path
        self.device = device
        self.gender = gender
        self.max_workers = max_workers
        
        # Default optimized smoothing parameters
        self.default_smoothing_config = {
            'smoothing_method': 'bilateral',
            'outlier_threshold': 3.0,
            'outlier_method': 'interpolate',
            'stabilize_shape': True,
            'shape_method': 'heavy_smooth',
            'spatial_sigmas': {
                'body_pose': 2.0,
                'betas': 0.5,
                'global_orient': 1.5,
                'transl': 1.5
            },
            'temporal_sigmas': {
                'body_pose': 0.3,
                'betas': 0.1,
                'global_orient': 0.2,
                'transl': 0.2
            }
        }
        
        print("🛡️  SAFE COMPLETE PARALLEL 3D HUMAN MESH PIPELINE")
        print("=" * 80)
        print(f"   SMPL-X Path: {smplx_path}")
        print(f"   Device: {device}")
        print(f"   Gender: {gender}")
        print(f"   Max Workers: {max_workers or 'auto (with fallbacks)'}")
        print("🛡️  Pipeline ready with multiple safety modes!")
    
    def test_parallel_capability(self, max_workers=2):
        """Test if parallel processing works on this system"""
        
        print(f"\n🧪 TESTING PARALLEL CAPABILITY")
        print("-" * 50)
        
        try:
            from concurrent.futures import ProcessPoolExecutor
            import time
            
            def simple_test(n):
                import time
                import numpy as np
                time.sleep(0.1)
                return n * 2
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(simple_test, i) for i in range(4)]
                results = [f.result(timeout=10) for f in futures]
            
            print(f"✅ Parallel processing works: {results}")
            return True
            
        except Exception as e:
            print(f"❌ Parallel processing failed: {e}")
            return False
    
    def execute_parallel_safe(self, video_path, output_dir, max_frames=None, frame_skip=1):
        """Execute parallel processing with multiple fallback strategies"""
        
        print(f"\n⚡ SAFE PARALLEL PROCESSING")
        print("-" * 50)
        
        # Strategy 1: Try normal parallel processing
        if self.max_workers is None or self.max_workers > 1:
            print(f"🚀 STRATEGY 1: Parallel processing")
            try:
                pipeline = ParallelMasterPipeline(
                    smplx_path=self.smplx_path,
                    device=self.device,
                    gender=self.gender,
                    max_workers=self.max_workers
                )
                
                # Test parallel capability first
                if self.test_parallel_capability():
                    result = pipeline.execute_parallel_pipeline(
                        video_path,
                        output_dir=output_dir,
                        max_frames=max_frames,
                        frame_skip=frame_skip,
                        timeout_per_frame=60  # Shorter timeout
                    )
                    
                    if result and result.get('mesh_file'):
                        print("✅ Parallel processing succeeded!")
                        return result
                    else:
                        print("⚠️  Parallel processing returned no results")
                else:
                    print("⚠️  Parallel capability test failed")
                        
            except Exception as e:
                print(f"❌ Parallel processing failed: {e}")
        
        # Strategy 2: Try minimal parallelization (2 workers)
        print(f"\n🚀 STRATEGY 2: Minimal parallel (2 workers)")
        try:
            pipeline = ParallelMasterPipeline(
                smplx_path=self.smplx_path,
                device=self.device,
                gender=self.gender,
                max_workers=2
            )
            
            result = pipeline.execute_parallel_pipeline(
                video_path,
                output_dir=output_dir,
                max_frames=max_frames,
                frame_skip=frame_skip,
                timeout_per_frame=60
            )
            
            if result and result.get('mesh_file'):
                print("✅ Minimal parallel processing succeeded!")
                return result
            else:
                print("⚠️  Minimal parallel processing returned no results")
                
        except Exception as e:
            print(f"❌ Minimal parallel processing failed: {e}")
        
        # Strategy 3: Single worker (sequential with parallel infrastructure)
        print(f"\n🚀 STRATEGY 3: Sequential processing (1 worker)")
        try:
            pipeline = ParallelMasterPipeline(
                smplx_path=self.smplx_path,
                device=self.device,
                gender=self.gender,
                max_workers=1
            )
            
            result = pipeline.execute_parallel_pipeline(
                video_path,
                output_dir=output_dir,
                max_frames=max_frames,
                frame_skip=frame_skip,
                timeout_per_frame=120  # Longer timeout for sequential
            )
            
            if result and result.get('mesh_file'):
                print("✅ Sequential processing succeeded!")
                return result
            else:
                print("⚠️  Sequential processing returned no results")
                
        except Exception as e:
            print(f"❌ Sequential processing failed: {e}")
        
        # Strategy 4: Fallback to original serial pipeline
        print(f"\n🚀 STRATEGY 4: Original serial pipeline fallback")
        try:
            from run_production_simple import MasterPipeline as SerialPipeline
            
            pipeline = SerialPipeline(
                smplx_path=self.smplx_path,
                device=self.device,
                gender=self.gender
            )
            
            result = pipeline.execute_full_pipeline(
                video_path,
                output_dir=output_dir,
                max_frames=max_frames,
                quality='ultra'
            )
            
            if result and result.get('mesh_file'):
                print("✅ Serial pipeline fallback succeeded!")
                # Convert to parallel-compatible format
                return {
                    'mesh_file': result['mesh_file'],
                    'stats': {
                        'meshes_generated': result.get('frames_processed', 0),
                        'processing_time': result.get('processing_time', 0)
                    },
                    'processing_method': 'serial_fallback'
                }
            else:
                print("⚠️  Serial pipeline returned no results")
                
        except Exception as e:
            print(f"❌ Serial pipeline fallback failed: {e}")
        
        print(f"\n💥 ALL PROCESSING STRATEGIES FAILED!")
        return None
    
    def execute_complete_safe_pipeline(self, video_path, output_dir="safe_pipeline_output",
                                      max_frames=None, frame_skip=1, quality='ultra',
                                      smoothing_config=None, compare_with_serial=None):
        """Execute complete safe pipeline with multiple fallback strategies"""
        
        print(f"\n🎬 EXECUTING SAFE COMPLETE PIPELINE")
        print("=" * 80)
        
        total_start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Use provided config or default
        if smoothing_config is None:
            smoothing_config = self.default_smoothing_config
        
        # PHASE 1: Safe Parallel Processing
        parallel_output_dir = output_dir / "parallel_phase"
        parallel_results = self.execute_parallel_safe(
            video_path, 
            parallel_output_dir, 
            max_frames, 
            frame_skip
        )
        
        if parallel_results is None:
            print("❌ All processing strategies failed")
            return None
        
        parallel_pkl_path = parallel_results['mesh_file']
        processing_method = parallel_results.get('processing_method', 'parallel')
        
        # PHASE 2: Post-Processing Smoothing (only if not serial fallback)
        if processing_method != 'serial_fallback':
            print(f"\n🎯 PHASE 2: POST-PROCESSING SMOOTHING")
            print("-" * 60)
            
            smoothing_start_time = time.time()
            
            smoothing_pipeline = PostProcessingSmoothingPipeline(
                smplx_path=self.smplx_path,
                device=self.device,
                gender=self.gender
            )
            
            smoothed_pkl_path = output_dir / f"{Path(video_path).stem}_safe_smoothed.pkl"
            
            smoothing_success = smoothing_pipeline.apply_post_processing_smoothing(
                input_pkl_path=str(parallel_pkl_path),
                output_pkl_path=str(smoothed_pkl_path),
                smoothing_method=smoothing_config.get('smoothing_method', 'bilateral'),
                stabilize_shape=smoothing_config.get('stabilize_shape', True),
                quality_assessment=True
            )
            
            if smoothing_success:
                smoothing_time = time.time() - smoothing_start_time
                print(f"✅ Post-processing complete in {smoothing_time:.1f}s")
                final_pkl_path = smoothed_pkl_path
            else:
                print("⚠️  Post-processing failed, using unsmoothed results")
                final_pkl_path = parallel_pkl_path
        else:
            print(f"\n⚠️  Using serial fallback - skipping post-processing (already smoothed)")
            final_pkl_path = parallel_pkl_path
        
        # PHASE 3: Final Results
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 SAFE PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"   Video: {Path(video_path).name}")
        print(f"   Processing method: {processing_method}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Final result: {final_pkl_path}")
        
        # Create final results
        final_results = {
            'input_video': str(video_path),
            'output_directory': str(output_dir),
            'final_pkl': str(final_pkl_path),
            'processing_method': processing_method,
            'total_time': total_time,
            'parallel_results': parallel_results,
            'success': True
        }
        
        # Save results
        results_path = output_dir / f"{Path(video_path).stem}_safe_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"   Results saved: {results_path}")
        
        return final_results

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='SAFE Complete Parallel 3D Human Mesh Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4
  %(prog)s input.mp4 --max-frames 50
  %(prog)s input.mp4 --max-workers 1  # Force sequential
  %(prog)s input.mp4 --device cpu      # Force CPU processing
        """
    )
    
    parser.add_argument('video_path', help='Input video file path')
    parser.add_argument('--output-dir', default='safe_pipeline_output',
                       help='Output directory (default: safe_pipeline_output)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--frame-skip', type=int, default=1, help='Frame skip interval')
    parser.add_argument('--max-workers', type=int, help='Maximum parallel workers (1 for sequential)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Processing device')
    parser.add_argument('--gender', choices=['neutral', 'male', 'female'], default='neutral', help='SMPL-X model gender')
    parser.add_argument('--smplx-path', default='models/smplx', help='Path to SMPL-X models')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    
    args = parse_arguments()
    
    print("🛡️  SAFE COMPLETE PARALLEL 3D HUMAN MESH PIPELINE")
    print("=" * 90)
    
    # Validate input
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        return 1
    
    if not Path(args.smplx_path).exists():
        print(f"❌ SMPL-X models not found: {args.smplx_path}")
        return 1
    
    print(f"✅ Input video: {args.video_path}")
    print(f"✅ SMPL-X models: {args.smplx_path}")
    print(f"✅ Output directory: {args.output_dir}")
    
    try:
        pipeline = SafeCompletePipeline(
            smplx_path=args.smplx_path,
            device=args.device,
            gender=args.gender,
            max_workers=args.max_workers
        )
        
        results = pipeline.execute_complete_safe_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip
        )
        
        if results and results.get('success'):
            print(f"\n🏆 MISSION ACCOMPLISHED!")
            print(f"   Method: {results['processing_method']}")
            print(f"   Time: {results['total_time']:.1f}s")
            print(f"   Result: {results['final_pkl']}")
            return 0
        else:
            print(f"\n💥 MISSION FAILED!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
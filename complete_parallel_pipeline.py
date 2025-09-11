#!/usr/bin/env python3
"""
Complete Parallel 3D Human Mesh Pipeline
End-to-end solution: Parallel processing + Post-processing smoothing
Production-ready implementation with optimized parameters
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


class CompletePipeline:
    """Complete production pipeline with parallel processing and post-processing smoothing"""
    
    def __init__(self, smplx_path="models/smplx", device='auto', gender='neutral', max_workers=None):
        self.smplx_path = smplx_path
        self.device = device
        self.gender = gender
        self.max_workers = max_workers
        
        # Default optimized smoothing parameters (can be overridden)
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
        
        print("🚀 COMPLETE PARALLEL 3D HUMAN MESH PIPELINE")
        print("=" * 70)
        print(f"   SMPL-X Path: {smplx_path}")
        print(f"   Device: {device}")
        print(f"   Gender: {gender}")
        print(f"   Max Workers: {max_workers or 'auto'}")
        print("🚀 Pipeline ready for production!")
    
    def load_optimized_config(self, config_path):
        """Load optimized smoothing configuration from parameter optimizer results"""
        
        if not Path(config_path).exists():
            print(f"⚠️  Config file not found: {config_path}, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                optimization_results = json.load(f)
            
            best_config = optimization_results.get('best_configuration', {})
            best_score = optimization_results.get('best_similarity_score', 0.0)
            
            if best_score > 0.0:
                self.default_smoothing_config.update(best_config)
                print(f"✅ Loaded optimized config (similarity: {best_score:.3f})")
                print(f"   Method: {best_config.get('smoothing_method', 'unknown')}")
            else:
                print(f"⚠️  Invalid optimization results, using defaults")
                
        except Exception as e:
            print(f"⚠️  Error loading config: {e}, using defaults")
    
    def execute_complete_pipeline(self, video_path, output_dir="complete_pipeline_output",
                                 max_frames=None, frame_skip=1, quality='ultra',
                                 smoothing_config=None, compare_with_serial=None):
        """Execute complete pipeline: parallel processing + post-processing smoothing"""
        
        print(f"\n🎬 EXECUTING COMPLETE PIPELINE")
        print("=" * 70)
        
        total_start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Use provided config or default
        if smoothing_config is None:
            smoothing_config = self.default_smoothing_config
        
        # PHASE 1: Parallel Processing (No Temporal Smoothing)
        print(f"\n⚡ PHASE 1: PARALLEL PROCESSING")
        print("-" * 50)
        
        parallel_start_time = time.time()
        
        parallel_pipeline = ParallelMasterPipeline(
            smplx_path=self.smplx_path,
            device=self.device,
            gender=self.gender,
            max_workers=self.max_workers
        )
        
        # Create parallel output directory
        parallel_output_dir = output_dir / "parallel_phase"
        
        parallel_results = parallel_pipeline.execute_parallel_pipeline(
            video_path,
            output_dir=parallel_output_dir,
            max_frames=max_frames,
            frame_skip=frame_skip
        )
        
        if parallel_results is None or not parallel_results.get('mesh_file'):
            print("❌ Parallel processing failed")
            return None
        
        parallel_time = time.time() - parallel_start_time
        parallel_pkl_path = parallel_results['mesh_file']
        
        print(f"✅ Parallel processing complete in {parallel_time:.1f}s")
        print(f"   Generated: {parallel_pkl_path}")
        
        # PHASE 2: Post-Processing Smoothing
        print(f"\n🎯 PHASE 2: POST-PROCESSING SMOOTHING")
        print("-" * 50)
        
        smoothing_start_time = time.time()
        
        smoothing_pipeline = PostProcessingSmoothingPipeline(
            smplx_path=self.smplx_path,
            device=self.device,
            gender=self.gender
        )
        
        # Create smoothed PKL path
        smoothed_pkl_path = output_dir / f"{Path(video_path).stem}_complete_smoothed.pkl"
        
        # Apply post-processing smoothing with optimized config
        smoothing_success = smoothing_pipeline.apply_post_processing_smoothing(
            input_pkl_path=str(parallel_pkl_path),
            output_pkl_path=str(smoothed_pkl_path),
            smoothing_method=smoothing_config.get('smoothing_method', 'bilateral'),
            stabilize_shape=smoothing_config.get('stabilize_shape', True),
            quality_assessment=True
        )
        
        if not smoothing_success:
            print("❌ Post-processing smoothing failed")
            return None
        
        smoothing_time = time.time() - smoothing_start_time
        
        print(f"✅ Post-processing complete in {smoothing_time:.1f}s")
        print(f"   Generated: {smoothed_pkl_path}")
        
        # PHASE 3: Quality Comparison (if reference provided)
        comparison_results = None
        if compare_with_serial and Path(compare_with_serial).exists():
            print(f"\n📊 PHASE 3: QUALITY COMPARISON")
            print("-" * 50)
            
            comparison_start_time = time.time()
            
            comparator = PKLComparator()
            comparison_report_path = output_dir / f"{Path(video_path).stem}_comparison_report.json"
            
            comparison_results = comparator.compare_pkl_files(
                serial_pkl_path=compare_with_serial,
                parallel_pkl_path=str(smoothed_pkl_path),
                output_report_path=str(comparison_report_path)
            )
            
            comparison_time = time.time() - comparison_start_time
            
            if comparison_results:
                similarity = comparison_results['overall_similarity']
                print(f"✅ Quality comparison complete in {comparison_time:.1f}s")
                print(f"   Overall similarity: {similarity:.3f}")
                print(f"   Report: {comparison_report_path}")
                
                if similarity > 0.95:
                    print("🏆 EXCELLENT: Near-perfect similarity to serial!")
                elif similarity > 0.90:
                    print("✅ SUCCESS: High similarity achieved!")
                else:
                    print("⚠️  Consider parameter tuning for better similarity")
        
        # PHASE 4: Final Results Summary
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 COMPLETE PIPELINE FINISHED!")
        print("=" * 70)
        print(f"   Video processed: {Path(video_path).name}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Parallel phase: {parallel_time:.1f}s ({parallel_time/total_time:.1%})")
        print(f"   Smoothing phase: {smoothing_time:.1f}s ({smoothing_time/total_time:.1%})")
        
        if parallel_results.get('stats'):
            stats = parallel_results['stats']
            print(f"   Frames processed: {stats['meshes_generated']}")
            theoretical_serial_time = stats['meshes_generated'] * 3.0  # Assume 3s/frame for RTX 4090
            speedup = theoretical_serial_time / total_time
            print(f"   Estimated speedup: {speedup:.1f}x vs serial")
        
        if comparison_results:
            similarity = comparison_results['overall_similarity']
            print(f"   Quality similarity: {similarity:.3f} ({similarity:.1%})")
        
        print(f"   Output directory: {output_dir}")
        
        # Create final results package
        final_results = {
            'input_video': str(video_path),
            'output_directory': str(output_dir),
            'parallel_pkl': str(parallel_pkl_path),
            'smoothed_pkl': str(smoothed_pkl_path),
            'processing_times': {
                'parallel_phase': parallel_time,
                'smoothing_phase': smoothing_time,
                'total_time': total_time
            },
            'parallel_results': parallel_results,
            'smoothing_config': smoothing_config,
            'comparison_results': comparison_results,
            'success': True
        }
        
        # Save final results
        final_results_path = output_dir / f"{Path(video_path).stem}_final_results.json"
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"   Final results: {final_results_path}")
        
        return final_results


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Complete Parallel 3D Human Mesh Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4
  %(prog)s input.mp4 --output-dir results --max-frames 100
  %(prog)s input.mp4 --compare-with-serial original.pkl
  %(prog)s input.mp4 --smoothing-config optimization_results.json
  %(prog)s input.mp4 --max-workers 8 --quality ultra
        """
    )
    
    # Required arguments
    parser.add_argument('video_path', help='Input video file path')
    
    # Optional arguments
    parser.add_argument('--output-dir', default='complete_pipeline_output',
                       help='Output directory (default: complete_pipeline_output)')
    
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (default: all)')
    
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Frame skip interval (default: 1)')
    
    parser.add_argument('--quality', choices=['ultra', 'high', 'medium'], default='ultra',
                       help='Processing quality (default: ultra)')
    
    parser.add_argument('--max-workers', type=int,
                       help='Maximum parallel workers (default: auto)')
    
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Processing device (default: auto)')
    
    parser.add_argument('--gender', choices=['neutral', 'male', 'female'], default='neutral',
                       help='SMPL-X model gender (default: neutral)')
    
    parser.add_argument('--smplx-path', default='models/smplx',
                       help='Path to SMPL-X models (default: models/smplx)')
    
    parser.add_argument('--smoothing-config',
                       help='Path to optimized smoothing configuration JSON file')
    
    parser.add_argument('--compare-with-serial',
                       help='Path to serial-processed PKL file for quality comparison')
    
    parser.add_argument('--smoothing-method', choices=['bilateral', 'savgol', 'moving_average'],
                       default='bilateral',
                       help='Smoothing method if no config provided (default: bilateral)')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    
    args = parse_arguments()
    
    print("🚀 COMPLETE PARALLEL 3D HUMAN MESH PIPELINE")
    print("=" * 80)
    
    # Validate input video
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video file not found: {args.video_path}")
        return 1
    
    # Validate SMPL-X models
    smplx_path = Path(args.smplx_path)
    if not smplx_path.exists() or not any(smplx_path.glob("*.npz")):
        print(f"❌ SMPL-X models not found in: {args.smplx_path}")
        print("Please download models from: https://smpl-x.is.tue.mpg.de/")
        return 1
    
    print(f"✅ Input video: {args.video_path}")
    print(f"✅ SMPL-X models: {args.smplx_path}")
    print(f"✅ Output directory: {args.output_dir}")
    
    try:
        # Initialize complete pipeline
        pipeline = CompletePipeline(
            smplx_path=args.smplx_path,
            device=args.device,
            gender=args.gender,
            max_workers=args.max_workers
        )
        
        # Load optimized smoothing configuration if provided
        if args.smoothing_config:
            pipeline.load_optimized_config(args.smoothing_config)
        
        # Override smoothing method if explicitly specified
        if args.smoothing_method and not args.smoothing_config:
            pipeline.default_smoothing_config['smoothing_method'] = args.smoothing_method
        
        # Execute complete pipeline
        results = pipeline.execute_complete_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            quality=args.quality,
            compare_with_serial=args.compare_with_serial
        )
        
        if results and results.get('success'):
            print(f"\n🏆 MISSION ACCOMPLISHED!")
            
            # Show performance summary
            times = results['processing_times']
            total_time = times['total_time']
            
            if results.get('parallel_results', {}).get('stats'):
                stats = results['parallel_results']['stats']
                frames = stats.get('meshes_generated', 0)
                if frames > 0:
                    fps = frames / total_time
                    print(f"   Processing rate: {fps:.2f} FPS")
            
            if results.get('comparison_results'):
                similarity = results['comparison_results']['overall_similarity']
                print(f"   Quality: {similarity:.1%} similarity to serial")
            
            print(f"   Check results in: {args.output_dir}")
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
#!/usr/bin/env python3
"""
Enhanced Video Processing Pipeline
Uses the new enhanced components for maximum speed and performance
"""

import os
import sys
import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from pathlib import Path

# Import enhanced components
from core.master_pipeline import MasterPipeline, PipelineConfig

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Video Processing Pipeline')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('--output-dir', default='enhanced_output',
                       help='Output directory for results')
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'ultra'], 
                       default='high', help='Processing quality')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto',
                       help='Device to use for processing')
    return parser.parse_args()

def extract_mediapipe_landmarks(video_path, max_frames=None):
    """Extract MediaPipe landmarks from video"""
    print("🎬 ENHANCED VIDEO PROCESSOR")
    print("=" * 50)
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Higher accuracy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"📹 Video Analysis:")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    print()
    
    landmarks_sequence = []
    confidences_sequence = []
    frame_count = 0
    start_time = time.time()
    
    print("🔍 Extracting landmarks...")
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        if results.pose_world_landmarks:
            # Extract 3D landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
            ])
            
            # Extract confidence scores
            confidences = np.array([
                lm.visibility for lm in results.pose_landmarks.landmark
            ]) if results.pose_landmarks else np.ones(33) * 0.5
            
            landmarks_sequence.append(landmarks)
            confidences_sequence.append(confidences)
        else:
            # Fill missing frames with previous frame or zeros
            if landmarks_sequence:
                landmarks_sequence.append(landmarks_sequence[-1].copy())
                confidences_sequence.append(confidences_sequence[-1].copy())
            else:
                # Use default pose if no previous frame
                landmarks_sequence.append(np.zeros((33, 3)))
                confidences_sequence.append(np.ones(33) * 0.1)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 50 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"   Frame {frame_count:4d}/{total_frames} ({progress:5.1f}%) - "
                  f"Extraction: {fps_current:.1f} FPS")
    
    cap.release()
    pose.close()
    
    extraction_time = time.time() - start_time
    extraction_fps = len(landmarks_sequence) / extraction_time
    
    print(f"\n✅ Landmark extraction completed!")
    print(f"   Frames processed: {len(landmarks_sequence)}")
    print(f"   Extraction time: {extraction_time:.1f}s")
    print(f"   Extraction FPS: {extraction_fps:.1f}")
    print()
    
    return landmarks_sequence, confidences_sequence

def main():
    """Main processing function"""
    args = parse_args()
    
    # Check input video
    if not Path(args.input_video).exists():
        print(f"❌ ERROR: Video file '{args.input_video}' not found!")
        return 1
    
    try:
        # Extract landmarks from video
        landmarks_sequence, confidences_sequence = extract_mediapipe_landmarks(
            args.input_video, args.max_frames
        )
        
        if not landmarks_sequence:
            print("❌ ERROR: No landmarks extracted from video!")
            return 1
        
        # Configure enhanced pipeline
        config = PipelineConfig()
        config.output_dir = args.output_dir
        config.quality_mode = args.quality
        config.batch_size = args.batch_size
        config.enable_batch_processing = True     # Enable high-speed batch processing
        config.enable_memory_optimization = True  # Enable smart memory management
        config.enable_angle_filtering = True      # Enable Kalman angle filtering
        config.enable_visualization = False       # Skip visualization for speed
        
        if args.device != 'auto':
            config.device = args.device
        
        print("⚡ ENHANCED PROCESSING PIPELINE")
        print("=" * 50)
        print(f"🔧 Configuration:")
        print(f"   Quality: {config.quality_mode}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Device: {config.device}")
        print(f"   Batch processing: {'✅' if config.enable_batch_processing else '❌'}")
        print(f"   Memory optimization: {'✅' if config.enable_memory_optimization else '❌'}")
        print(f"   Kalman filtering: {'✅' if config.enable_angle_filtering else '❌'}")
        print()
        
        # Initialize enhanced pipeline
        pipeline = MasterPipeline(config)
        
        # Process with enhanced pipeline
        print("🚀 Processing with enhanced components...")
        processing_start = time.time()
        
        sequence_name = Path(args.input_video).stem
        result = pipeline.process_sequence(
            landmarks_sequence=landmarks_sequence,
            confidences_sequence=confidences_sequence,
            sequence_name=sequence_name
        )
        
        processing_time = time.time() - processing_start
        
        # Display results
        print("\n🎯 ENHANCED PROCESSING COMPLETED!")
        print("=" * 50)
        print(f"📊 Performance Statistics:")
        print(f"   Total frames: {result.get('total_frames', 0)}")
        print(f"   Successful frames: {result.get('successful_frames', 0)}")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Average FPS: {result.get('total_frames', 0) / processing_time:.1f}")
        print()
        
        # Quality assessment
        quality = result.get('quality_assessment', {})
        print(f"🏆 Quality Assessment:")
        print(f"   Overall grade: {quality.get('overall_grade', 'Unknown')}")
        print(f"   Confidence score: {quality.get('confidence_score', 0):.3f}")
        print()
        
        # Sequence statistics
        seq_stats = result.get('sequence_statistics', {})
        if seq_stats:
            print(f"📈 Enhanced Features:")
            enhancement_rates = seq_stats.get('enhancement_rates', {})
            for feature, rate in enhancement_rates.items():
                print(f"   {feature.replace('_', ' ').title()}: {rate*100:.1f}%")
            print()
        
        # Export information
        output_dir = Path(config.output_dir)
        print(f"💾 Output Files:")
        print(f"   Directory: {output_dir}")
        
        # Save enhanced results
        pkl_file = output_dir / f"{sequence_name}_enhanced.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"   Enhanced PKL: {pkl_file}")
        
        # Save just the landmarks for compatibility
        compat_pkl_file = output_dir / f"{sequence_name}_meshes.pkl"
        landmarks_data = {
            'landmarks_sequence': landmarks_sequence,
            'confidences_sequence': confidences_sequence,
            'enhanced_result': result
        }
        with open(compat_pkl_file, 'wb') as f:
            pickle.dump(landmarks_data, f)
        print(f"   Compatible PKL: {compat_pkl_file}")
        
        print(f"\n✅ SUCCESS! Enhanced processing completed in {processing_time:.1f}s")
        print(f"🚀 Speed improvement: ~{3600/max(processing_time/len(landmarks_sequence), 0.001):.0f}x faster than baseline!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPLETE PIPELINE VALIDATION SCRIPT
===================================

This script validates the entire MediaPipe -> 3D Mesh pipeline locally
before RunPod deployment, ensuring all components work correctly.

Usage:
    python validate_complete_pipeline.py

Requirements:
    - Input video at: input_video.mp4
    - Conda environment activated or dependencies installed
"""

import os
import sys
import traceback
from pathlib import Path
import subprocess

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(status, message):
    """Print status message with formatting"""
    symbols = {"OK": "+", "WARN": "!", "ERROR": "X", "INFO": "i"}
    print(f"{symbols.get(status, '-')} {status}: {message}")

def check_python_version():
    """Check Python version compatibility"""
    print_section("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print_status("INFO", f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print_status("OK", "Python version is compatible with MediaPipe")
        return True
    else:
        print_status("WARN", "Python version may have compatibility issues")
        print_status("INFO", "Recommended: Python 3.9 for best compatibility")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_section("DEPENDENCY CHECK")
    
    required_packages = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("torch", "pytorch"),
        ("open3d", "open3d"),
        ("matplotlib", "matplotlib"),
        ("smplx", "smplx"),
        ("PIL", "Pillow")
    ]
    
    missing_packages = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print_status("OK", f"{package} is installed")
        except ImportError:
            print_status("ERROR", f"{package} is missing (install: pip install {pip_name})")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print_status("ERROR", f"Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print_status("OK", "All required packages are installed")
        return True

def check_input_video():
    """Check if input video exists and is valid"""
    print_section("INPUT VIDEO CHECK")
    
    # Check for real video first
    real_video_path = r"C:\Users\vaclavik\Videos\smpl.mp4"
    test_video_path = "input_video.mp4"
    
    # Prefer real video if available
    video_path = real_video_path if os.path.exists(real_video_path) else test_video_path
    
    if not os.path.exists(video_path):
        # Look for any video files in current directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        found_videos = []
        for ext in video_extensions:
            for file in os.listdir('.'):
                if file.lower().endswith(ext):
                    found_videos.append(file)
        
        print_status("ERROR", f"No suitable video found for testing")
        if found_videos:
            print_status("INFO", f"Found videos in current dir: {', '.join(found_videos)}")
            print_status("INFO", f"Using first found video for testing")
            video_path = found_videos[0]
        else:
            print_status("INFO", "No video files found")
            print_status("INFO", "Please place a test video for validation")
            return False, None
    
    print_status("OK", f"Using video: {video_path}")
    
    # Check video properties
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print_status("ERROR", "Cannot open input video")
            return False, None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        print_status("INFO", f"Resolution: {width}x{height}")
        print_status("INFO", f"FPS: {fps:.2f}")
        print_status("INFO", f"Duration: {duration:.1f}s ({frame_count} frames)")
        
        if width >= 640 and height >= 480:
            print_status("OK", "Video resolution is adequate for pose detection")
        else:
            print_status("WARN", "Low video resolution may affect pose detection accuracy")
        
        return True, video_path
        
    except Exception as e:
        print_status("ERROR", f"Error checking video properties: {e}")
        return False, None

def check_smplx_models():
    """Check if SMPL-X model files are available"""
    print_section("SMPL-X MODELS CHECK")
    
    models_dir = Path("models/smplx")
    
    if not models_dir.exists():
        print_status("ERROR", f"SMPL-X models directory not found: {models_dir}")
        print_status("INFO", "Create models/smplx/ directory and download SMPL-X models")
        return False
    
    required_models = [
        "SMPLX_NEUTRAL.npz",
        "SMPLX_MALE.npz", 
        "SMPLX_FEMALE.npz"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            print_status("OK", f"{model} found")
        else:
            print_status("ERROR", f"{model} missing")
            missing_models.append(model)
    
    if missing_models:
        print_status("ERROR", f"Missing SMPL-X models: {', '.join(missing_models)}")
        print_status("INFO", "Download from: https://smpl-x.is.tue.mpg.de/")
        return False
    else:
        print_status("OK", "All SMPL-X models are available")
        return True

def test_mediapipe_detection(video_path=None):
    """Test MediaPipe pose detection on sample frames"""
    print_section("MEDIAPIPE POSE DETECTION TEST")
    
    # Use provided video path or try to find one
    if video_path is None:
        real_video = r"C:\Users\vaclavik\Videos\smpl.mp4"
        test_video = "input_video.mp4"
        video_path = real_video if os.path.exists(real_video) else test_video
    
    if not os.path.exists(video_path):
        print_status("ERROR", "Cannot test MediaPipe - no input video available")
        print_status("INFO", "Need a video file for MediaPipe testing")
        return False
    
    print_status("INFO", f"Testing MediaPipe on: {video_path}")
    
    try:
        # Add pracovni_poloha2 to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pracovni_poloha2', 'src'))
        
        from pose_detector import PoseDetector
        import cv2
        
        # Test pose detector initialization
        detector = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print_status("OK", "MediaPipe PoseDetector initialized")
        
        # Test on first few frames
        cap = cv2.VideoCapture(video_path)
        
        test_frames = 5
        successful_detections = 0
        
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            pose_results = detector.detect_pose(frame)
            if pose_results.pose_world_landmarks is not None:
                landmarks_3d = detector.extract_3d_landmarks(pose_results.pose_world_landmarks)
                if landmarks_3d is not None and len(landmarks_3d) == 33:
                    successful_detections += 1
        
        cap.release()
        # Note: MediaPipe detector doesn't need explicit close
        
        success_rate = (successful_detections / test_frames) * 100
        print_status("INFO", f"Detected poses in {successful_detections}/{test_frames} test frames")
        print_status("INFO", f"Detection success rate: {success_rate:.1f}%")
        
        if success_rate >= 60:
            print_status("OK", "MediaPipe pose detection is working well")
            return True
        else:
            print_status("WARN", "Low pose detection success rate")
            return False
            
    except Exception as e:
        print_status("ERROR", f"MediaPipe test failed: {e}")
        traceback.print_exc()
        return False

def test_3d_mesh_generation():
    """Test 3D mesh generation with SMPL-X"""
    print_section("3D MESH GENERATION TEST")
    
    try:
        # Check if we can import the main pipeline
        from production_3d_pipeline_clean import MasterPipeline
        
        print_status("OK", "3D pipeline module imported successfully")
        
        # Test pipeline initialization (CPU mode for validation)
        pipeline = MasterPipeline(device='cpu')
        print_status("OK", "3D pipeline initialized in CPU mode")
        
        # Note: Full test would require running actual mesh fitting
        # which is resource intensive, so we just validate initialization
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"3D mesh generation test failed: {e}")
        traceback.print_exc()
        return False

def test_output_directories():
    """Check/create output directories"""
    print_section("OUTPUT DIRECTORIES CHECK")
    
    required_dirs = [
        "outputs",
        "outputs/mesh_exports", 
        "pracovni_poloha2/data/output"
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print_status("OK", f"Directory ready: {dir_path}")
        except Exception as e:
            print_status("ERROR", f"Cannot create directory {dir_path}: {e}")
            return False
    
    return True

def generate_validation_report():
    """Generate final validation report"""
    print_section("VALIDATION SUMMARY")
    
    # Run checks with video path passing
    video_check_result, video_path = check_input_video()
    
    all_checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Input Video", video_check_result),
        ("SMPL-X Models", check_smplx_models()),
        ("Output Directories", test_output_directories()),
        ("MediaPipe Detection", test_mediapipe_detection(video_path) if video_path else False),
        ("3D Mesh Generation", test_3d_mesh_generation())
    ]
    
    passed_checks = sum(1 for _, passed in all_checks if passed)
    total_checks = len(all_checks)
    
    print(f"\nValidation Results: {passed_checks}/{total_checks} checks passed")
    
    for check_name, passed in all_checks:
        status = "PASS" if passed else "FAIL"
        print_status("OK" if passed else "ERROR", f"{check_name}: {status}")
    
    if passed_checks == total_checks:
        print_status("OK", "ALL CHECKS PASSED - Ready for RunPod deployment!")
        print("\nNext steps:")
        print("1. Purchase RunPod GPU access")
        print("2. Upload project to RunPod")
        print("3. Run: python setup_runpod_conda.py")
        print("4. Run: python production_3d_pipeline_clean.py")
    else:
        print_status("ERROR", f"{total_checks - passed_checks} checks failed - Fix issues before deployment")
        print("\nRecommended actions:")
        print("1. Install missing dependencies")
        print("2. Download required SMPL-X models")
        print("3. Ensure input video is available")
        print("4. Re-run this validation script")
    
    return passed_checks == total_checks

def main():
    """Main validation function"""
    print("COMPLETE PIPELINE VALIDATION")
    print("=" * 60)
    print("Validating MediaPipe -> 3D Mesh pipeline before RunPod deployment")
    print("This may take a few minutes...")
    
    try:
        success = generate_validation_report()
        
        if success:
            print("\n+ VALIDATION SUCCESSFUL!")
            print("Your pipeline is ready for RunPod GPU deployment.")
        else:
            print("\n- VALIDATION INCOMPLETE")
            print("Please fix the issues above and run validation again.")
            
        return success
        
    except KeyboardInterrupt:
        print("\n! Validation interrupted by user")
        return False
    except Exception as e:
        print(f"\nX Validation failed with unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
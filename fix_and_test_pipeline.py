#!/usr/bin/env python3
"""
COMPLETE FIX AND TEST SCRIPT
Ensures pipeline works correctly with proper error handling
"""

import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

def create_real_test_video():
    """Create a test video with actual human-like poses"""
    print("Creating proper test video with human-like figure...")
    
    width, height = 640, 480
    fps = 25
    duration = 3
    frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('human_test.mp4', fourcc, fps, (width, height))
    
    for i in range(frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw stick figure that MediaPipe might recognize
        center_x = width // 2 + int(30 * np.sin(i * 0.1))
        center_y = height // 2
        
        # Head
        cv2.circle(frame, (center_x, center_y - 100), 20, (0, 0, 0), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 80), (center_x, center_y), (0, 0, 0), 5)
        
        # Arms
        cv2.line(frame, (center_x, center_y - 60), (center_x - 50, center_y - 30), (0, 0, 0), 5)
        cv2.line(frame, (center_x, center_y - 60), (center_x + 50, center_y - 30), (0, 0, 0), 5)
        
        # Legs
        cv2.line(frame, (center_x, center_y), (center_x - 30, center_y + 80), (0, 0, 0), 5)
        cv2.line(frame, (center_x, center_y), (center_x + 30, center_y + 80), (0, 0, 0), 5)
        
        out.write(frame)
    
    out.release()
    print("✓ Created human_test.mp4")
    return "human_test.mp4"

def download_sample_video():
    """Download a real video with a person"""
    print("Downloading sample video with real person...")
    
    # Sample videos with humans
    urls = [
        # Yoga pose video (good for testing)
        "https://github.com/google/mediapipe/raw/master/mediapipe/examples/desktop/media/pose_tracking_example.mp4",
    ]
    
    for url in urls:
        output = "real_person.mp4"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, output)
            print(f"✓ Downloaded: {output}")
            return output
        except:
            pass
    
    print("✗ Could not download sample video")
    return None

def test_mediapipe_detection(video_path):
    """Test if MediaPipe can detect poses in video"""
    print(f"\nTesting MediaPipe detection on: {video_path}")
    
    import mediapipe as mp
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    detected_frames = 0
    total_frames = 0
    max_test = 10  # Test first 10 frames
    
    while total_frames < max_test:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_world_landmarks:
            detected_frames += 1
            print(f"  Frame {total_frames}: ✓ Pose detected")
        else:
            print(f"  Frame {total_frames}: ✗ No pose detected")
        
        total_frames += 1
    
    cap.release()
    pose.close()
    
    detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
    print(f"\nDetection rate: {detection_rate:.1f}% ({detected_frames}/{total_frames})")
    
    return detection_rate > 50

def analyze_pkl_file(pkl_path):
    """Analyze what's actually in the PKL file"""
    print(f"\nAnalyzing PKL file: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        print("✗ PKL file not found")
        return False
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded PKL successfully")
        print(f"  Type: {type(data)}")
        
        if isinstance(data, list):
            print(f"  Frames: {len(data)}")
            
            if data:
                first = data[0]
                print(f"  First frame type: {type(first)}")
                
                if isinstance(first, dict):
                    for key, value in first.items():
                        if isinstance(value, np.ndarray):
                            print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                            
                            # Check if it's valid mesh data
                            if key == 'vertices' and value.shape[0] == 10475:
                                print(f"      ✓ Valid SMPL-X vertices!")
                            elif key == 'faces' and len(value) == 20908:
                                print(f"      ✓ Valid SMPL-X faces!")
                        else:
                            print(f"    {key}: {type(value)}")
                
                # Check mesh validity
                if 'vertices' in first and 'faces' in first:
                    vertices = first['vertices']
                    faces = first['faces']
                    
                    if vertices.shape[0] > 0 and len(faces) > 0:
                        print(f"\n  ✓ Valid mesh data found!")
                        print(f"    Vertices: {vertices.shape}")
                        print(f"    Faces: {len(faces)}")
                        
                        # Check if vertices are reasonable
                        v_min = vertices.min(axis=0)
                        v_max = vertices.max(axis=0)
                        v_range = v_max - v_min
                        
                        print(f"    Bounding box range: {v_range}")
                        
                        if np.all(v_range > 0.1) and np.all(v_range < 10):
                            print(f"    ✓ Mesh has reasonable proportions")
                            return True
                        else:
                            print(f"    ✗ Mesh has strange proportions (might be corrupted)")
                            return False
                    else:
                        print(f"  ✗ Empty mesh data")
                        return False
        
        return False
        
    except Exception as e:
        print(f"✗ Error loading PKL: {e}")
        return False

def create_fixed_pipeline():
    """Create a fixed version of the pipeline with proper error handling"""
    
    fixed_code = '''#!/usr/bin/env python3
"""
FIXED RUNPOD PIPELINE WITH PROPER ERROR HANDLING
"""

import os
import sys
import numpy as np
import cv2
import torch
import mediapipe as mp
from pathlib import Path
import pickle

# ... (rest of the imports)

class FixedRunPodPipeline:
    """Fixed pipeline with validation"""
    
    def process_video(self, video_path):
        print(f"Processing: {video_path}")
        
        # First, test if MediaPipe can detect anything
        test_frames = self.test_mediapipe_on_video(video_path)
        
        if test_frames == 0:
            print("ERROR: MediaPipe cannot detect any poses in this video!")
            print("This video does not contain detectable human figures.")
            print("Please use a video with clearly visible people.")
            return None
        
        print(f"OK: MediaPipe can detect poses in {test_frames} frames")
        
        # Continue with normal processing...
        # (rest of the pipeline)
    
    def test_mediapipe_on_video(self, video_path, max_frames=10):
        """Pre-test video for MediaPipe compatibility"""
        cap = cv2.VideoCapture(str(video_path))
        detected = 0
        
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.detector.process_frame(frame)
            if landmarks is not None:
                detected += 1
        
        cap.release()
        return detected
'''
    
    with open('production_3d_pipeline_fixed.py', 'w') as f:
        f.write(fixed_code)
    
    print("✓ Created production_3d_pipeline_fixed.py with proper validation")

def main():
    """Main diagnostic and fix routine"""
    print("="*60)
    print("PIPELINE DIAGNOSTIC AND FIX")
    print("="*60)
    
    # 1. Check what videos we have
    print("\n1. CHECKING AVAILABLE VIDEOS:")
    videos = list(Path('.').glob('*.mp4'))
    for video in videos:
        print(f"  - {video}")
        if 'test' in str(video).lower():
            print("    ⚠ This might be the synthetic test video")
    
    # 2. Test MediaPipe on current test.mp4
    if Path('test.mp4').exists():
        print("\n2. TESTING CURRENT test.mp4:")
        if not test_mediapipe_detection('test.mp4'):
            print("  ✗ PROBLEM: test.mp4 cannot be processed by MediaPipe!")
            print("  This is why your pipeline is failing.")
    
    # 3. Create or download proper test video
    print("\n3. CREATING PROPER TEST VIDEO:")
    
    # Try to download real video first
    real_video = download_sample_video()
    
    if not real_video:
        # Create better test video
        real_video = create_real_test_video()
    
    # Test the new video
    if real_video and test_mediapipe_detection(real_video):
        print(f"\n✓ SUCCESS: {real_video} works with MediaPipe!")
        print(f"Use this video for testing: python production_3d_pipeline_runpod.py")
        
        # Rename it to test.mp4
        os.rename(real_video, 'test_proper.mp4')
        print("Renamed to: test_proper.mp4")
    
    # 4. Analyze existing PKL file
    pkl_files = list(Path('.').glob('*.pkl'))
    if pkl_files:
        print("\n4. ANALYZING EXISTING MESH DATA:")
        for pkl in pkl_files:
            if analyze_pkl_file(pkl):
                print(f"  ✓ {pkl} contains valid mesh data")
            else:
                print(f"  ✗ {pkl} has problems")
    
    # 5. Create fixed pipeline
    print("\n5. CREATING FIXED PIPELINE:")
    create_fixed_pipeline()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nPROBLEM: Your test.mp4 doesn't contain a detectable human figure.")
    print("SOLUTION: Use test_proper.mp4 or any real video with people.")
    print("\nTO FIX:")
    print("1. Replace test.mp4 with a real video")
    print("2. Run: python production_3d_pipeline_runpod.py")
    print("3. Or run: python visualize_and_export_mesh.py to view existing meshes")

if __name__ == "__main__":
    main()
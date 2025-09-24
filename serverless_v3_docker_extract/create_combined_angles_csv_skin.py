#!/usr/bin/env python3
"""
Combined Angles CSV Creator with SKIN-BASED trunk angle
Uses vertex 2151 (cervical) and 5614 (lumbar) for trunk angle calculation
Keeps neck and arm angles as before
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import cv2
import json
import warnings
warnings.filterwarnings('ignore')

# Import existing calculators  
from neck_angle_calculator_skin import calculate_neck_angle_skin  # NEW skin-based neck
from neck_angle_calculator_like_arm import calculate_neck_angle_to_trunk_like_arm  # Original joint-based
from arm_angle_calculator import calculate_bilateral_arm_angles, SMPL_X_JOINT_INDICES

# SKIN VERTEX IDs for trunk
CERVICAL_SKIN_VERTEX = 2151  # Neck area on skin
LUMBAR_SKIN_VERTEX_1 = 5614  # Lower back option 1
LUMBAR_SKIN_VERTEX_2 = 4298  # Lower back option 2

def detect_video_fps(video_path):
    """
    Detect FPS of the original video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        fps: Video framerate, or 30.0 as fallback
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                print(f"  Detected video FPS: {fps:.2f}")
                return fps
    except Exception as e:
        print(f"  Warning: Could not detect FPS from video: {e}")
    
    print(f"  Using default FPS: 30.0")
    return 30.0

def find_original_video(pkl_file):
    """
    Try to find the original video file based on PKL filename
    
    Args:
        pkl_file: PKL file path
        
    Returns:
        video_path: Path to video file or None
    """
    pkl_path = Path(pkl_file)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Try same name with video extensions
    for ext in video_extensions:
        video_path = pkl_path.parent / (pkl_path.stem.replace('_meshes', '').replace('_filtered', '') + ext)
        if video_path.exists():
            print(f"  Found original video: {video_path}")
            return video_path
    
    # Look for any video files in same directory
    for ext in video_extensions:
        video_files = list(pkl_path.parent.glob(f'*{ext}'))
        if video_files:
            print(f"  Found video file: {video_files[0]}")
            return video_files[0]
    
    print(f"  No original video found for FPS detection")
    return None

def create_combined_angles_csv_skin(pkl_file, output_csv="combined_angles_skin.csv", lumbar_vertex=5614, video_path=None):
    """
    Create CSV with all angles, using SKIN vertices for trunk
    
    Args:
        pkl_file: Input PKL file with mesh data
        output_csv: Output CSV filename
        lumbar_vertex: Which lumbar vertex to use (5614 or 4298)
        video_path: Optional path to original video for FPS detection
    
    Output columns:
    - frame: Frame number (0-based)
    - time_seconds: Time in seconds (frame / fps)
    - fps: Video framerate
    - trunk_angle_skin: Trunk angle from SKIN vertices in degrees
    - trunk_angle_joints: Original trunk angle from joints (for comparison)
    - neck_angle_skin: Neck angle from SKIN vertices in degrees
    - neck_angle_joints: Original neck angle from joints (for comparison)
    - left_arm_angle: Left arm angle in degrees
    - right_arm_angle: Right arm angle in degrees
    """
    
    print("CREATING COMBINED ANGLES CSV WITH SKIN-BASED TRUNK & NECK")
    print("=" * 50)
    print(f"Using skin vertices:")
    print(f"  TRUNK: Lumbar {lumbar_vertex} -> Cervical {CERVICAL_SKIN_VERTEX}")
    print(f"  NECK: Cervical {CERVICAL_SKIN_VERTEX} -> Head 9002")
    
    # Load PKL data first to get FPS from metadata
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        return None
    
    print(f"\nLoading PKL file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # Handle both old and new PKL format
    if isinstance(pkl_data, dict) and 'mesh_sequence' in pkl_data:
        # New format with metadata
        meshes = pkl_data['mesh_sequence']
        metadata = pkl_data.get('metadata', {})
        fps = metadata.get('fps', 30.0)
        print(f"  FPS from PKL metadata: {fps:.2f}")
        if 'video_filename' in metadata:
            print(f"  Original video: {metadata['video_filename']}")
        if 'frame_skip' in metadata:
            print(f"  Frame skip: {metadata['frame_skip']}")
    else:
        # Old format - just mesh sequence
        meshes = pkl_data
        fps = 30.0  # Default fallback
        print(f"  Old PKL format detected")
        
        # Try to detect FPS from video as fallback
        if video_path and Path(video_path).exists():
            fps = detect_video_fps(video_path)
        else:
            # Try to find original video
            found_video = find_original_video(pkl_file)
            if found_video:
                fps = detect_video_fps(found_video)
        
        print(f"  Using FPS: {fps:.2f}")
    
    print(f"Loaded {len(meshes)} frames")
    
    # Verify vertices exist
    if meshes:
        first_frame = meshes[0]
        num_vertices = len(first_frame['vertices'])
        print(f"Mesh has {num_vertices} vertices")
        
        if lumbar_vertex >= num_vertices:
            print(f"ERROR: Lumbar vertex {lumbar_vertex} out of range!")
            return None
        if CERVICAL_SKIN_VERTEX >= num_vertices:
            print(f"ERROR: Cervical vertex {CERVICAL_SKIN_VERTEX} out of range!")
            return None
    
    # Process all frames
    results = []
    print(f"\nProcessing {len(meshes)} frames...")
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']
        vertices = mesh_data['vertices']
        
        result = {
            'frame': frame_idx,
            'time_seconds': frame_idx / fps,
            'fps': fps,
            'trunk_angle_skin': 0.0,
            'trunk_angle_joints': 0.0,
            'neck_angle_skin': 0.0,
            'neck_angle_joints': 0.0,
            'left_arm_angle': 0.0,
            'right_arm_angle': 0.0
        }
        
        # 1. Calculate SKIN-BASED trunk angle
        try:
            lumbar_skin = vertices[lumbar_vertex]
            cervical_skin = vertices[CERVICAL_SKIN_VERTEX]
            trunk_vector_skin = cervical_skin - lumbar_skin
            trunk_length = np.linalg.norm(trunk_vector_skin)
            
            if trunk_length > 0:
                spine_unit = trunk_vector_skin / trunk_length
                vertical = np.array([0, 1, 0])  # Y-up coordinate system
                cos_angle = np.dot(spine_unit, vertical)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                trunk_angle_skin = np.degrees(np.arccos(cos_angle))
                result['trunk_angle_skin'] = trunk_angle_skin
        except Exception as e:
            print(f"  Warning: Skin trunk angle failed for frame {frame_idx}: {e}")
        
        # 2. Calculate ORIGINAL trunk angle (from joints) for comparison
        try:
            lumbar_joint = joints[SMPL_X_JOINT_INDICES['spine1']]
            cervical_joint = joints[SMPL_X_JOINT_INDICES['neck']]
            trunk_vector_joints = cervical_joint - lumbar_joint
            trunk_length_joints = np.linalg.norm(trunk_vector_joints)
            
            if trunk_length_joints > 0:
                spine_unit_joints = trunk_vector_joints / trunk_length_joints
                vertical = np.array([0, 1, 0])
                cos_angle = np.dot(spine_unit_joints, vertical)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                trunk_angle_joints = np.degrees(np.arccos(cos_angle))
                result['trunk_angle_joints'] = trunk_angle_joints
        except Exception as e:
            print(f"  Warning: Joint trunk angle failed for frame {frame_idx}: {e}")
        
        # 3. Calculate SKIN-BASED neck angle
        try:
            neck_result_skin = calculate_neck_angle_skin(vertices, joints)
            if neck_result_skin:
                result['neck_angle_skin'] = neck_result_skin['sagittal_angle']
        except Exception as e:
            print(f"  Warning: Skin neck angle failed for frame {frame_idx}: {e}")
        
        # 4. Calculate ORIGINAL neck angle (from joints) for comparison
        try:
            neck_result_joints = calculate_neck_angle_to_trunk_like_arm(joints, vertices)
            if neck_result_joints:
                result['neck_angle_joints'] = neck_result_joints['sagittal_angle']
        except Exception as e:
            print(f"  Warning: Joint neck angle failed for frame {frame_idx}: {e}")
        
        # 5. Calculate arm angles (unchanged)
        try:
            arm_result = calculate_bilateral_arm_angles(joints)
            if arm_result:
                result['left_arm_angle'] = arm_result['left_arm']['sagittal_angle']
                result['right_arm_angle'] = arm_result['right_arm']['sagittal_angle']
        except Exception as e:
            print(f"  Warning: Arm angles failed for frame {frame_idx}: {e}")
        
        results.append(result)
        
        # Progress output
        if frame_idx % 50 == 0 or frame_idx < 5:
            print(f"Frame {frame_idx:3d}: T_skin={result['trunk_angle_skin']:6.1f}° T_joint={result['trunk_angle_joints']:6.1f}° "
                  f"N_skin={result['neck_angle_skin']:6.1f}° N_joint={result['neck_angle_joints']:6.1f}°")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"\nCOMBINED ANGLES CSV (SKIN-BASED) CREATED!")
    print(f"File: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Show statistics
    print(f"\nVIDEO INFORMATION:")
    print(f"Total frames: {len(df)}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Total duration: {len(df)/fps:.1f} seconds")
    
    print(f"\nANGLE STATISTICS:")
    print(f"Trunk (SKIN):    {df['trunk_angle_skin'].mean():6.1f}° ± {df['trunk_angle_skin'].std():5.1f}° "
          f"(range: {df['trunk_angle_skin'].min():6.1f}° to {df['trunk_angle_skin'].max():6.1f}°)")
    print(f"Trunk (joints):  {df['trunk_angle_joints'].mean():6.1f}° ± {df['trunk_angle_joints'].std():5.1f}° "
          f"(range: {df['trunk_angle_joints'].min():6.1f}° to {df['trunk_angle_joints'].max():6.1f}°)")
    
    print(f"\nNeck (SKIN):     {df['neck_angle_skin'].mean():6.1f}° ± {df['neck_angle_skin'].std():5.1f}° "
          f"(range: {df['neck_angle_skin'].min():6.1f}° to {df['neck_angle_skin'].max():6.1f}°)")
    print(f"Neck (joints):   {df['neck_angle_joints'].mean():6.1f}° ± {df['neck_angle_joints'].std():5.1f}° "
          f"(range: {df['neck_angle_joints'].min():6.1f}° to {df['neck_angle_joints'].max():6.1f}°)")
    
    # Compare skin vs joint differences
    trunk_diff = (df['trunk_angle_skin'] - df['trunk_angle_joints']).abs()
    neck_diff = (df['neck_angle_skin'] - df['neck_angle_joints']).abs()
    
    print(f"\nSKIN vs JOINT DIFFERENCES:")
    print(f"  Trunk - Mean diff: {trunk_diff.mean():.1f}°, Max diff: {trunk_diff.max():.1f}°")
    print(f"  Neck - Mean diff: {neck_diff.mean():.1f}°, Max diff: {neck_diff.max():.1f}°")
    
    print(f"\nLeft arm angle:  {df['left_arm_angle'].mean():6.1f}° ± {df['left_arm_angle'].std():5.1f}°")
    print(f"Right arm angle: {df['right_arm_angle'].mean():6.1f}° ± {df['right_arm_angle'].std():5.1f}°")
    
    # Show first few rows
    print(f"\nFIRST 5 ROWS:")
    print(df.head().to_string(index=False))
    
    print(f"\nSUCCESS: Combined angles CSV with SKIN-BASED trunk ready!")
    return output_path

def main():
    """Main execution function with vertex choice"""
    
    # Default PKL file
    pkl_file = "help.pkl"
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Available PKL files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
        return
    
    print("\nSKIN-BASED COMBINED ANGLES")
    print("Choose lumbar vertex:")
    print("1. Vertex 5614 (default)")
    print("2. Vertex 4298 (alternative)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        output_file = create_combined_angles_csv_skin(
            pkl_file, 
            "combined_angles_skin_4298.csv",
            lumbar_vertex=4298
        )
    else:
        output_file = create_combined_angles_csv_skin(
            pkl_file, 
            "combined_angles_skin_5614.csv",
            lumbar_vertex=5614
        )
    
    if output_file:
        print(f"\nREADY FOR ANALYSIS:")
        print(f"Import into Excel, Python pandas, or any analysis tool")
        print(f"File: {output_file}")
        print(f"\nCompare columns:")
        print(f"  trunk_angle_skin vs trunk_angle_joints")
        print(f"  neck_angle_skin vs neck_angle_joints")
        print(f"  Both skin versions use vertices on actual skin surface!")

if __name__ == "__main__":
    main()
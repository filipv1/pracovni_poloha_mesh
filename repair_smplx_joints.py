#!/usr/bin/env python3
"""
SMPL-X Joint Repair System
Intelligently fixes invalid joint positions from failed MediaPipe → SMPL-X fitting
Uses temporal interpolation, anatomical constraints, and biomechanical validation
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# SMPL-X joint indices for arms
SMPL_X_JOINT_INDICES = {
    'pelvis': 0,
    'spine1': 3,          # Lower spine (lumbar region)  
    'neck': 12,           # Neck base (cervical region)
    'left_shoulder': 17,  # Left shoulder joint
    'right_shoulder': 16, # Right shoulder joint
    'left_elbow': 19,     # Left elbow joint
    'right_elbow': 18,    # Right elbow joint
}

# Critical joints that need repair
CRITICAL_ARM_JOINTS = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']

class SMPLXJointRepairer:
    """Intelligent SMPL-X joint position repair system"""
    
    def __init__(self):
        # Anatomical constraints (in meters)
        self.MIN_ARM_LENGTH = 0.15    # Minimum realistic upper arm length
        self.MAX_ARM_LENGTH = 0.45    # Maximum realistic upper arm length
        self.MIN_SHOULDER_WIDTH = 0.25  # Minimum shoulder width
        self.MAX_SHOULDER_WIDTH = 0.65  # Maximum shoulder width
        self.MAX_FRAME_DELTA = 0.10   # Maximum joint movement between frames
        
        # Detection thresholds
        self.ZERO_THRESHOLD = 1e-6    # Consider as zero
        self.EXTREME_VALUE_THRESHOLD = 5.0  # Extreme coordinate values
        
    def detect_invalid_joints(self, joints):
        """
        Detect invalid joint positions using multiple criteria
        
        Returns:
            dict: {joint_name: [invalid_frame_indices]}
        """
        
        invalid_joints = {joint: [] for joint in CRITICAL_ARM_JOINTS}
        n_frames = len(joints)
        
        print(f"DETECTING INVALID JOINTS IN {n_frames} FRAMES...")
        
        for frame_idx in range(n_frames):
            frame_joints = joints[frame_idx]
            
            for joint_name in CRITICAL_ARM_JOINTS:
                joint_idx = SMPL_X_JOINT_INDICES[joint_name]
                joint_pos = frame_joints[joint_idx]
                
                is_invalid = False
                reasons = []
                
                # 1. Zero coordinates detection
                if np.allclose(joint_pos, 0, atol=self.ZERO_THRESHOLD):
                    is_invalid = True
                    reasons.append("zero_coords")
                
                # 2. Extreme values detection
                if np.any(np.abs(joint_pos) > self.EXTREME_VALUE_THRESHOLD):
                    is_invalid = True
                    reasons.append("extreme_values")
                
                # 3. Anatomical constraints
                if 'shoulder' in joint_name:
                    # Check shoulder width
                    left_shoulder = frame_joints[SMPL_X_JOINT_INDICES['left_shoulder']]
                    right_shoulder = frame_joints[SMPL_X_JOINT_INDICES['right_shoulder']]
                    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                    
                    if shoulder_width < self.MIN_SHOULDER_WIDTH or shoulder_width > self.MAX_SHOULDER_WIDTH:
                        is_invalid = True
                        reasons.append(f"invalid_shoulder_width_{shoulder_width:.3f}")
                
                elif 'elbow' in joint_name:
                    # Check arm length
                    if 'left' in joint_name:
                        shoulder_pos = frame_joints[SMPL_X_JOINT_INDICES['left_shoulder']]
                    else:
                        shoulder_pos = frame_joints[SMPL_X_JOINT_INDICES['right_shoulder']]
                    
                    arm_length = np.linalg.norm(joint_pos - shoulder_pos)
                    if arm_length < self.MIN_ARM_LENGTH or arm_length > self.MAX_ARM_LENGTH:
                        is_invalid = True
                        reasons.append(f"invalid_arm_length_{arm_length:.3f}")
                
                # 4. Temporal discontinuity (large jumps)
                if frame_idx > 0:
                    prev_joint_pos = joints[frame_idx - 1][joint_idx]
                    delta = np.linalg.norm(joint_pos - prev_joint_pos)
                    
                    if delta > self.MAX_FRAME_DELTA:
                        is_invalid = True
                        reasons.append(f"large_jump_{delta:.3f}")
                
                if is_invalid:
                    invalid_joints[joint_name].append({
                        'frame': frame_idx,
                        'position': joint_pos.copy(),
                        'reasons': reasons
                    })
        
        # Summary
        total_invalid = sum(len(frames) for frames in invalid_joints.values())
        print(f"FOUND {total_invalid} INVALID JOINT POSITIONS:")
        
        for joint_name, invalid_frames in invalid_joints.items():
            if invalid_frames:
                print(f"  {joint_name}: {len(invalid_frames)} frames")
                # Show first few examples
                for i, inv in enumerate(invalid_frames[:3]):
                    reasons_str = ", ".join(inv['reasons'])
                    print(f"    Frame {inv['frame']}: {reasons_str}")
                if len(invalid_frames) > 3:
                    print(f"    ... and {len(invalid_frames) - 3} more")
        
        return invalid_joints
    
    def repair_joints_intelligent(self, joints, invalid_joints):
        """
        Intelligently repair invalid joint positions
        """
        
        print(f"REPAIRING INVALID JOINTS...")
        repaired_joints = [frame.copy() for frame in joints]  # Deep copy
        n_frames = len(joints)
        
        repair_stats = {joint: {'repaired': 0, 'method': []} for joint in CRITICAL_ARM_JOINTS}
        
        for joint_name, invalid_frames in invalid_joints.items():
            if not invalid_frames:
                continue
                
            joint_idx = SMPL_X_JOINT_INDICES[joint_name]
            invalid_frame_indices = [inv['frame'] for inv in invalid_frames]
            
            print(f"  Repairing {joint_name}: {len(invalid_frame_indices)} frames")
            
            # Extract all positions for this joint
            all_positions = np.array([joints[i][joint_idx] for i in range(n_frames)])
            
            # Find valid frames (not in invalid list)
            valid_frame_indices = [i for i in range(n_frames) if i not in invalid_frame_indices]
            
            if len(valid_frame_indices) < 2:
                print(f"    WARNING: Too few valid frames for {joint_name}, using anatomical fallback")
                # Use anatomical fallback (mirror from other arm, etc.)
                repaired_positions = self._anatomical_fallback(joints, joint_name, invalid_frame_indices)
                repair_method = "anatomical_fallback"
                
            else:
                # Use temporal interpolation/extrapolation
                valid_positions = all_positions[valid_frame_indices]
                
                # Create interpolation function
                if len(valid_frame_indices) >= 4:
                    # Use cubic spline for smooth interpolation
                    kind = 'cubic'
                else:
                    # Use linear for few points
                    kind = 'linear'
                
                # Interpolate each coordinate separately
                repaired_positions = np.zeros((len(invalid_frame_indices), 3))
                
                for coord in range(3):  # x, y, z
                    coord_values = valid_positions[:, coord]
                    
                    # Create interpolation function with extrapolation
                    interp_func = interp1d(
                        valid_frame_indices, coord_values,
                        kind=kind, fill_value='extrapolate'
                    )
                    
                    # Repair invalid frames
                    for i, frame_idx in enumerate(invalid_frame_indices):
                        repaired_positions[i, coord] = interp_func(frame_idx)
                
                repair_method = f"temporal_{kind}"
            
            # Apply repairs
            for i, frame_idx in enumerate(invalid_frame_indices):
                repaired_joints[frame_idx][joint_idx] = repaired_positions[i]
                repair_stats[joint_name]['repaired'] += 1
                repair_stats[joint_name]['method'].append(repair_method)
        
        # Apply temporal smoothing to all joints
        print("  Applying temporal smoothing...")
        repaired_joints = self._apply_temporal_smoothing(repaired_joints)
        
        # Print repair summary
        print(f"REPAIR SUMMARY:")
        for joint_name, stats in repair_stats.items():
            if stats['repaired'] > 0:
                methods = set(stats['method'])
                print(f"  {joint_name}: {stats['repaired']} frames repaired using {', '.join(methods)}")
        
        return repaired_joints
    
    def _anatomical_fallback(self, joints, joint_name, invalid_frame_indices):
        """
        Anatomical fallback when not enough valid data for interpolation
        """
        n_invalid = len(invalid_frame_indices)
        fallback_positions = np.zeros((n_invalid, 3))
        
        if 'left_shoulder' in joint_name:
            # Mirror from right shoulder with typical offset
            for i, frame_idx in enumerate(invalid_frame_indices):
                right_shoulder = joints[frame_idx][SMPL_X_JOINT_INDICES['right_shoulder']]
                spine = joints[frame_idx][SMPL_X_JOINT_INDICES['spine1']]
                
                # Mirror across spine
                spine_to_right = right_shoulder - spine
                spine_to_left = np.array([-spine_to_right[0], spine_to_right[1], spine_to_right[2]])
                fallback_positions[i] = spine + spine_to_left
                
        elif 'right_shoulder' in joint_name:
            # Mirror from left shoulder
            for i, frame_idx in enumerate(invalid_frame_indices):
                left_shoulder = joints[frame_idx][SMPL_X_JOINT_INDICES['left_shoulder']]
                spine = joints[frame_idx][SMPL_X_JOINT_INDICES['spine1']]
                
                spine_to_left = left_shoulder - spine
                spine_to_right = np.array([-spine_to_left[0], spine_to_left[1], spine_to_left[2]])
                fallback_positions[i] = spine + spine_to_right
                
        elif 'elbow' in joint_name:
            # Position elbow at typical distance from shoulder
            typical_arm_length = 0.30  # 30cm typical upper arm
            
            for i, frame_idx in enumerate(invalid_frame_indices):
                if 'left' in joint_name:
                    shoulder = joints[frame_idx][SMPL_X_JOINT_INDICES['left_shoulder']]
                    # Typical elbow position (slightly forward and down)
                    elbow_offset = np.array([0.1, -0.25, -0.1])  # Forward, down, slightly in
                else:
                    shoulder = joints[frame_idx][SMPL_X_JOINT_INDICES['right_shoulder']]
                    elbow_offset = np.array([-0.1, -0.25, -0.1])  # Forward, down, slightly in
                
                # Normalize and scale to typical arm length
                elbow_offset = elbow_offset / np.linalg.norm(elbow_offset) * typical_arm_length
                fallback_positions[i] = shoulder + elbow_offset
        
        return fallback_positions
    
    def _apply_temporal_smoothing(self, joints):
        """
        Apply gentle temporal smoothing to reduce jitter
        """
        n_frames = len(joints)
        smoothed_joints = [frame.copy() for frame in joints]
        
        # Apply Gaussian smoothing to each joint trajectory
        sigma = 1.0  # Smoothing strength
        
        for joint_name in CRITICAL_ARM_JOINTS:
            joint_idx = SMPL_X_JOINT_INDICES[joint_name]
            
            # Extract trajectory for this joint
            trajectory = np.array([joints[i][joint_idx] for i in range(n_frames)])
            
            # Smooth each coordinate
            for coord in range(3):
                smoothed_coord = gaussian_filter1d(trajectory[:, coord], sigma=sigma)
                
                # Apply smoothed values
                for frame_idx in range(n_frames):
                    smoothed_joints[frame_idx][joint_idx][coord] = smoothed_coord[frame_idx]
        
        return smoothed_joints
    
    def validate_repair_quality(self, original_joints, repaired_joints):
        """
        Validate the quality of repairs
        """
        print(f"VALIDATING REPAIR QUALITY...")
        
        n_frames = len(original_joints)
        validation_results = {}
        
        for joint_name in CRITICAL_ARM_JOINTS:
            joint_idx = SMPL_X_JOINT_INDICES[joint_name]
            
            # Calculate metrics
            original_traj = np.array([original_joints[i][joint_idx] for i in range(n_frames)])
            repaired_traj = np.array([repaired_joints[i][joint_idx] for i in range(n_frames)])
            
            # Smoothness metric (frame-to-frame variation)
            orig_deltas = np.diff(original_traj, axis=0)
            repaired_deltas = np.diff(repaired_traj, axis=0)
            
            orig_smoothness = np.mean(np.linalg.norm(orig_deltas, axis=1))
            repaired_smoothness = np.mean(np.linalg.norm(repaired_deltas, axis=1))
            
            # Anatomical constraint violations
            if 'elbow' in joint_name:
                if 'left' in joint_name:
                    shoulders = np.array([repaired_joints[i][SMPL_X_JOINT_INDICES['left_shoulder']] for i in range(n_frames)])
                else:
                    shoulders = np.array([repaired_joints[i][SMPL_X_JOINT_INDICES['right_shoulder']] for i in range(n_frames)])
                
                arm_lengths = np.linalg.norm(repaired_traj - shoulders, axis=1)
                invalid_lengths = np.sum((arm_lengths < self.MIN_ARM_LENGTH) | (arm_lengths > self.MAX_ARM_LENGTH))
            else:
                invalid_lengths = 0
            
            validation_results[joint_name] = {
                'original_smoothness': orig_smoothness,
                'repaired_smoothness': repaired_smoothness,
                'smoothness_improvement': orig_smoothness - repaired_smoothness,
                'anatomical_violations': invalid_lengths
            }
            
            print(f"  {joint_name}:")
            print(f"    Smoothness: {orig_smoothness:.4f} -> {repaired_smoothness:.4f} (Delta{repaired_smoothness-orig_smoothness:+.4f})")
            print(f"    Anatomical violations: {invalid_lengths}")
        
        return validation_results

def repair_smplx_pkl_file(input_pkl, output_pkl):
    """
    Main function to repair SMPL-X PKL file
    """
    
    print(f"SMPL-X JOINT REPAIR SYSTEM")
    print(f"Input: {input_pkl}")
    print(f"Output: {output_pkl}")
    print("=" * 60)
    
    # Load PKL file
    with open(input_pkl, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"Loaded {len(meshes)} frames")
    
    # Extract joint data
    joints_data = [mesh['joints'] for mesh in meshes]
    
    # Initialize repairer
    repairer = SMPLXJointRepairer()
    
    # Step 1: Detect invalid joints
    invalid_joints = repairer.detect_invalid_joints(joints_data)
    
    # Step 2: Repair joints
    repaired_joints = repairer.repair_joints_intelligent(joints_data, invalid_joints)
    
    # Step 3: Validate repair quality
    validation = repairer.validate_repair_quality(joints_data, repaired_joints)
    
    # Step 4: Update meshes with repaired joints
    repaired_meshes = []
    for i, mesh in enumerate(meshes):
        repaired_mesh = mesh.copy()
        repaired_mesh['joints'] = repaired_joints[i]
        repaired_meshes.append(repaired_mesh)
    
    # Step 5: Save repaired PKL
    with open(output_pkl, 'wb') as f:
        pickle.dump(repaired_meshes, f)
    
    print(f"\nREPAIR COMPLETE!")
    print(f"  Repaired PKL saved: {output_pkl}")
    print(f"  Original preserved: {input_pkl}")
    
    return output_pkl, validation

if __name__ == "__main__":
    input_file = "fil_vid_meshes.pkl"
    output_file = "fil_vid_meshes_joints_repaired.pkl"
    
    if Path(input_file).exists():
        repaired_file, validation = repair_smplx_pkl_file(input_file, output_file)
        print(f"\nREADY!")
        print(f"Use repaired file: {repaired_file}")
        print(f"For comprehensive visualization, rename to: fil_vid_meshes.pkl")
    else:
        print(f"PKL file not found: {input_file}")
        available_files = list(Path(".").glob("*.pkl"))
        if available_files:
            print("Available PKL files:")
            for pkl in available_files:
                print(f"  - {pkl}")
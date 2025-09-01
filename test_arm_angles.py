#!/usr/bin/env python3
"""
Test script for arm angle calculations
Validates the mathematical correctness with known test cases
"""

import numpy as np
import matplotlib.pyplot as plt
from arm_angle_calculator import calculate_arm_angle_to_trunk_robust, SMPL_X_JOINT_INDICES
from pathlib import Path

def create_test_pose(trunk_angle=0, arm_sagittal=0, arm_frontal=0, side='left'):
    """
    Create synthetic joint positions for testing
    
    Args:
        trunk_angle: trunk tilt in degrees (0=upright, 90=horizontal)
        arm_sagittal: arm angle in sagittal plane (0=along trunk, 90=forward, -90=back)
        arm_frontal: arm angle in frontal plane (0=along trunk, 90=out to side)
        side: 'left' or 'right'
    """
    
    # Create SMPL-X joint array (117, 3)
    joints = np.zeros((117, 3))
    
    # Convert angles to radians
    trunk_rad = np.radians(trunk_angle)
    arm_sag_rad = np.radians(arm_sagittal)
    arm_front_rad = np.radians(arm_frontal)
    
    # Basic body dimensions (in meters)
    trunk_length = 0.3      # 30cm trunk
    arm_length = 0.25       # 25cm arm (shoulder to elbow)
    shoulder_width = 0.4    # 40cm shoulder width
    
    # === TRUNK SETUP ===
    # Lumbar position (base)
    lumbar_pos = np.array([0, 0, 0])
    
    # Cervical position (trunk rotated by trunk_angle)
    trunk_direction = np.array([
        np.sin(trunk_rad),  # forward component when tilted
        0,                  # no side tilt
        np.cos(trunk_rad)   # up component
    ])
    cervical_pos = lumbar_pos + trunk_direction * trunk_length
    
    # === SHOULDERS SETUP ===
    # Shoulder positions relative to cervical (horizontal line)
    shoulder_offset = 0.05 * trunk_length  # shoulders slightly below neck
    
    if side == 'left':
        shoulder_pos = cervical_pos + np.array([0, -shoulder_width/2, -shoulder_offset])
    else:
        shoulder_pos = cervical_pos + np.array([0, +shoulder_width/2, -shoulder_offset])
    
    # Other shoulder for coordinate system
    other_shoulder_pos = cervical_pos + np.array([0, shoulder_width/2 if side == 'left' else -shoulder_width/2, -shoulder_offset])
    
    # === ARM CALCULATION ===
    # Create body coordinate system
    trunk_up = trunk_direction
    shoulder_right = np.array([0, 1, 0])  # Y-axis pointing right
    body_forward = np.cross(shoulder_right, trunk_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    
    # Arm direction in body coordinate system
    # Start with arm along trunk (down)
    arm_base_direction = -trunk_up  # pointing down along trunk
    
    # Rotate by sagittal angle (around shoulder_right axis)
    rotation_axis_sag = shoulder_right
    cos_sag = np.cos(arm_sag_rad)
    sin_sag = np.sin(arm_sag_rad)
    # Rodrigues rotation formula
    arm_after_sag = (arm_base_direction * cos_sag + 
                     np.cross(rotation_axis_sag, arm_base_direction) * sin_sag + 
                     rotation_axis_sag * np.dot(rotation_axis_sag, arm_base_direction) * (1 - cos_sag))
    
    # Rotate by frontal angle (around body_forward axis) 
    rotation_axis_front = body_forward
    cos_front = np.cos(arm_front_rad)
    sin_front = np.sin(arm_front_rad)
    
    side_sign = 1 if side == 'left' else -1
    arm_front_rad_adjusted = arm_front_rad * side_sign
    
    arm_direction = (arm_after_sag * np.cos(arm_front_rad_adjusted) + 
                    np.cross(rotation_axis_front, arm_after_sag) * np.sin(arm_front_rad_adjusted) + 
                    rotation_axis_front * np.dot(rotation_axis_front, arm_after_sag) * (1 - np.cos(arm_front_rad_adjusted)))
    
    # Elbow position
    elbow_pos = shoulder_pos + arm_direction * arm_length
    
    # === POPULATE JOINTS ARRAY ===
    joints[SMPL_X_JOINT_INDICES['pelvis']] = lumbar_pos - np.array([0, 0, 0.1])  # pelvis slightly below
    joints[SMPL_X_JOINT_INDICES['spine1']] = lumbar_pos
    joints[SMPL_X_JOINT_INDICES['neck']] = cervical_pos
    joints[SMPL_X_JOINT_INDICES['left_shoulder']] = shoulder_pos if side == 'left' else other_shoulder_pos
    joints[SMPL_X_JOINT_INDICES['right_shoulder']] = other_shoulder_pos if side == 'left' else shoulder_pos
    joints[SMPL_X_JOINT_INDICES['left_elbow']] = elbow_pos if side == 'left' else shoulder_pos + np.array([0, 0, -0.2])
    joints[SMPL_X_JOINT_INDICES['right_elbow']] = shoulder_pos + np.array([0, 0, -0.2]) if side == 'left' else elbow_pos
    
    return joints, {
        'expected_sagittal': arm_sagittal,
        'expected_frontal': arm_frontal,
        'trunk_angle': trunk_angle,
        'side': side
    }

def test_specific_case(trunk_angle, arm_sagittal, arm_frontal, side='left', tolerance=5.0):
    """Test specific arm angle case"""
    
    joints, expected = create_test_pose(trunk_angle, arm_sagittal, arm_frontal, side)
    result = calculate_arm_angle_to_trunk_robust(joints, side)
    
    if result is None:
        return False, "Calculation failed (None result)"
    
    # Check sagittal angle
    sag_diff = abs(result['sagittal_angle'] - expected['expected_sagittal'])
    # front_diff = abs(result['frontal_angle'] - expected['expected_frontal'])
    
    success = sag_diff <= tolerance
    
    return success, {
        'expected_sag': expected['expected_sagittal'],
        'calculated_sag': result['sagittal_angle'],
        'diff_sag': sag_diff,
        'confidence': result['confidence'],
        'components': result['components']
    }

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    
    print("COMPREHENSIVE ARM ANGLE TESTS")
    print("=" * 60)
    
    test_cases = [
        # (trunk_angle, arm_sagittal, arm_frontal, description)
        (0, 0, 0, "Upright body, arms hanging down"),
        (0, 90, 0, "Upright body, arms forward (shoulder height)"),
        (0, -90, 0, "Upright body, arms backward"),
        (0, 45, 0, "Upright body, arms 45° forward"),
        (0, -45, 0, "Upright body, arms 45° backward"),
        
        # Tilted trunk tests
        (90, 0, 0, "Horizontal trunk, arms hanging (should be -90°)"),
        (90, 90, 0, "Horizontal trunk, arms forward (should be 0°)"),
        (45, 0, 0, "45° tilted trunk, arms hanging"),
        (45, 45, 0, "45° tilted trunk, arms 45° forward"),
        
        # Frontal plane tests (bonus)
        (0, 0, 90, "Upright body, arms to side"),
        (0, 45, 45, "Upright body, arms diagonal"),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for trunk_angle, arm_sagittal, arm_frontal, description in test_cases:
        print(f"\nTest: {description}")
        print(f"   Setup: trunk={trunk_angle}°, arm_sag={arm_sagittal}°, arm_front={arm_frontal}°")
        
        # Test both arms
        for side in ['left', 'right']:
            success, result = test_specific_case(trunk_angle, arm_sagittal, arm_frontal, side)
            
            if success:
                print(f"   PASS {side.upper()}: Expected={result['expected_sag']:.1f}°, "
                      f"Got={result['calculated_sag']:.1f}° (diff={result['diff_sag']:.1f}°, "
                      f"conf={result['confidence']:.2f})")
                passed += 1
            else:
                if isinstance(result, str):
                    print(f"   FAIL {side.upper()}: {result}")
                else:
                    print(f"   FAIL {side.upper()}: Expected={result['expected_sag']:.1f}°, "
                          f"Got={result['calculated_sag']:.1f}° (diff={result['diff_sag']:.1f}°!)")
                failed += 1
            
            results.append({
                'test_case': description,
                'side': side,
                'trunk_angle': trunk_angle,
                'arm_sagittal_expected': arm_sagittal,
                'arm_frontal_expected': arm_frontal,
                'success': success,
                'result': result
            })
    
    print(f"\nTEST SUMMARY:")
    print(f"   Passed: {passed}/{passed + failed} ({passed/(passed + failed)*100:.1f}%)")
    print(f"   Failed: {failed}/{passed + failed}")
    
    return results

def test_with_real_pkl_data():
    """Test with real PKL data if available"""
    
    print(f"\nTESTING WITH REAL PKL DATA")
    print("=" * 60)
    
    pkl_files = list(Path(".").glob("*.pkl"))
    
    if not pkl_files:
        print("No PKL files found for testing")
        return False
    
    pkl_file = pkl_files[0]  # Use first available PKL
    print(f"Using: {pkl_file}")
    
    try:
        from arm_angle_calculator import analyze_arm_movement_sequence
        
        # Test just first few frames for speed
        results = analyze_arm_movement_sequence(pkl_file, "test_arm_analysis")
        
        # Check if we got reasonable results
        valid_results = [r for r in results[:10] if r['left_arm'] and r['right_arm']]
        
        if valid_results:
            print(f"PASS: Successfully processed {len(valid_results)}/10 test frames")
            
            # Show some sample results
            for i, result in enumerate(valid_results[:3]):
                left = result['left_arm']
                right = result['right_arm']
                print(f"   Frame {result['frame']}: "
                      f"L={left['sagittal_angle']:6.1f}° ({left['confidence']:.2f}), "
                      f"R={right['sagittal_angle']:6.1f}° ({right['confidence']:.2f})")
            
            return True
        else:
            print("FAIL: No valid results from PKL processing")
            return False
            
    except Exception as e:
        print(f"FAIL: Error processing PKL: {e}")
        return False

def visualize_test_results():
    """Create visualization of test case geometry"""
    
    print(f"\nCREATING TEST VISUALIZATIONS")
    print("=" * 60)
    
    # Test case: upright body, arm 45° forward
    joints, expected = create_test_pose(trunk_angle=0, arm_sagittal=45, arm_frontal=0, side='left')
    result = calculate_arm_angle_to_trunk_robust(joints, 'left')
    
    if result is None:
        print("FAIL: Cannot create visualization - calculation failed")
        return
    
    # Extract key points
    lumbar = joints[SMPL_X_JOINT_INDICES['spine1']]
    cervical = joints[SMPL_X_JOINT_INDICES['neck']]
    left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
    left_elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw trunk
    ax.plot([lumbar[0], cervical[0]], [lumbar[1], cervical[1]], [lumbar[2], cervical[2]], 
            'r-', linewidth=4, label='Trunk')
    
    # Draw arm
    ax.plot([left_shoulder[0], left_elbow[0]], [left_shoulder[1], left_elbow[1]], [left_shoulder[2], left_elbow[2]], 
            'b-', linewidth=3, label='Left Arm')
    
    # Draw coordinate system
    coord_sys = result['coordinate_system']
    origin = left_shoulder
    scale = 0.1
    
    # Forward (X) - green
    ax.quiver(origin[0], origin[1], origin[2], 
              coord_sys['body_forward'][0], coord_sys['body_forward'][1], coord_sys['body_forward'][2],
              length=scale, color='green', label='Forward')
    
    # Right (Y) - yellow  
    ax.quiver(origin[0], origin[1], origin[2], 
              coord_sys['shoulder_right'][0], coord_sys['shoulder_right'][1], coord_sys['shoulder_right'][2],
              length=scale, color='yellow', label='Right')
    
    # Up (Z) - red
    ax.quiver(origin[0], origin[1], origin[2], 
              coord_sys['trunk_up'][0], coord_sys['trunk_up'][1], coord_sys['trunk_up'][2],
              length=scale, color='red', label='Trunk Up')
    
    # Labels and formatting
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Right)')
    ax.set_zlabel('Z (Up)')
    ax.legend()
    ax.set_title(f'Test Case: 45° Forward Arm\nCalculated: {result["sagittal_angle"]:.1f}° (Expected: 45°)')
    
    # Set equal aspect ratio
    max_range = 0.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    vis_file = "arm_angle_test_visualization.png"
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {vis_file}")
    
    # Print detailed results
    print(f"\nDETAILED TEST RESULTS:")
    print(f"  Expected sagittal: 45.0°")
    print(f"  Calculated sagittal: {result['sagittal_angle']:.1f}°")
    print(f"  Error: {abs(result['sagittal_angle'] - 45.0):.1f}°")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Components: forward={result['components']['forward']:.3f}, up={result['components']['up']:.3f}")

if __name__ == "__main__":
    print("ARM ANGLE CALCULATION TESTING SUITE")
    print("=" * 60)
    
    # Run synthetic tests
    synthetic_results = run_comprehensive_tests()
    
    # Run real data tests
    real_data_success = test_with_real_pkl_data()
    
    # Create visualizations
    visualize_test_results()
    
    print(f"\nTESTING COMPLETE")
    print(f"=" * 60)
    print(f"Synthetic tests: Check results above")
    print(f"Real data test: {'PASSED' if real_data_success else 'FAILED'}")
    print(f"Visualizations: arm_angle_test_visualization.png")
    print(f"\nReady for iteration and improvements!")
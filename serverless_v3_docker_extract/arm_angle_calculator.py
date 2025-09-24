#!/usr/bin/env python3
"""
Robust Arm Angle Calculator - Calculates arm angles relative to trunk
Handles all orientations and positions, anatomically correct
"""

import numpy as np
import pickle
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

# SMPL-X joint indices
SMPL_X_JOINT_INDICES = {
    'pelvis': 0,          # Root/pelvis joint
    'spine1': 3,          # Lower spine (lumbar region)  
    'spine2': 6,          # Mid spine
    'spine3': 9,          # Upper spine
    'neck': 12,           # Neck base (cervical region)
    'head': 15,           # Head
    'left_shoulder': 17,  # Left shoulder joint (FIXED: was 16)
    'right_shoulder': 16, # Right shoulder joint (FIXED: was 17)
    'left_elbow': 19,     # Left elbow joint (FIXED: was 18)
    'right_elbow': 18,    # Right elbow joint (FIXED: was 19)
}

def calculate_arm_angle_to_trunk_robust(joints, arm_side='left'):
    """
    Robustní výpočet úhlu paže vůči trupu v sagitální rovině
    
    Args:
        joints: SMPL-X joint positions (117, 3)
        arm_side: 'left' nebo 'right'
    
    Returns:
        dict with sagittal_angle, frontal_angle, confidence, components
    """
    
    # === ANATOMICKÉ BODY ===
    lumbar = joints[SMPL_X_JOINT_INDICES['spine1']]      # L3/L4
    cervical = joints[SMPL_X_JOINT_INDICES['neck']]       # C7/T1
    left_shoulder = joints[SMPL_X_JOINT_INDICES['left_shoulder']]
    right_shoulder = joints[SMPL_X_JOINT_INDICES['right_shoulder']]
    
    if arm_side == 'left':
        shoulder = left_shoulder
        elbow = joints[SMPL_X_JOINT_INDICES['left_elbow']]
        side_sign = 1  # pro levou ruku
    else:
        shoulder = right_shoulder
        elbow = joints[SMPL_X_JOINT_INDICES['right_elbow']]
        side_sign = -1  # pro pravou ruku (zrcadlení)
    
    # === ZÁKLADNÍ VEKTORY ===
    trunk_vector = cervical - lumbar
    arm_vector = elbow - shoulder  
    shoulder_width_vector = left_shoulder - right_shoulder  # FIXED: left->right direction
    
    # === VALIDACE DÉLKY VEKTORŮ ===
    trunk_length = np.linalg.norm(trunk_vector)
    arm_length = np.linalg.norm(arm_vector)
    shoulder_width = np.linalg.norm(shoulder_width_vector)
    
    # Minimální prahové hodnoty
    MIN_TRUNK_LENGTH = 0.05   # 5cm
    MIN_ARM_LENGTH = 0.03     # 3cm  
    MIN_SHOULDER_WIDTH = 0.08 # 8cm
    
    confidence = 1.0
    
    if trunk_length < MIN_TRUNK_LENGTH:
        return None  # Nevalidní trunk
    if arm_length < MIN_ARM_LENGTH:
        return None  # Paže příliš stažená
    if shoulder_width < MIN_SHOULDER_WIDTH:
        confidence *= 0.5  # Snížená spolehlivost
    
    # === ANATOMICKÝ KOORDINÁTNÍ SYSTÉM ===
    # Z-axis: "nahoru" podél trupu
    trunk_up = trunk_vector / trunk_length
    
    # Y-axis: "doprava" (levé → pravé rameno)
    shoulder_right = shoulder_width_vector / shoulder_width
    
    # X-axis: "dopředu" (cross produkt)
    body_forward_unnorm = np.cross(shoulder_right, trunk_up)
    body_forward_length = np.linalg.norm(body_forward_unnorm)
    
    if body_forward_length < 1e-8:
        confidence *= 0.3  # Ramena rovnoběžná s trupem
        # Fallback: použij jiný směr
        body_forward = np.array([1, 0, 0])  # defaultní forward
    else:
        body_forward = body_forward_unnorm / body_forward_length
    
    # Rekalkulace right axis pro ortogonalitu
    shoulder_right = np.cross(trunk_up, body_forward) 
    
    # === TRANSFORMACE ARM VEKTORU ===
    arm_norm = arm_vector / arm_length
    
    # Projekce na anatomické osy
    arm_forward_comp = np.dot(arm_norm, body_forward)    # dopředu/dozadu
    arm_up_comp = np.dot(arm_norm, trunk_up)            # nahoru/dolů  
    arm_right_comp = np.dot(arm_norm, shoulder_right)    # doprava/doleva
    
    # === SAGITÁLNÍ ÚHEL ===
    # We want: 0° = podél trupu (down), +90° = forward, -90° = backward
    # arm_up_comp: positive when arm points up along trunk
    # arm_forward_comp: positive when arm points forward
    
    # Use atan2 to get angle from "down along trunk" direction
    # When arm hangs down: arm_up_comp = -1, arm_forward_comp = 0 -> angle = 0°
    # When arm forward: arm_up_comp = 0, arm_forward_comp = 1 -> angle = 90°
    # When arm backward: arm_up_comp = 0, arm_forward_comp = -1 -> angle = -90°
    
    sagittal_angle_rad = np.arctan2(arm_forward_comp, -arm_up_comp)
    sagittal_angle_deg = np.degrees(sagittal_angle_rad)
    
    # === FRONTÁLNÍ ÚHEL (bonus) ===
    frontal_angle_rad = np.arctan2(arm_right_comp * side_sign, arm_up_comp) 
    frontal_angle_deg = np.degrees(frontal_angle_rad)
    
    # === CONFIDENCE SCORING ===
    # Penalizace za extrémní úhly nebo nestabilitu
    if abs(sagittal_angle_deg) > 135:  # Velmi extrémní pozice
        confidence *= 0.7
        
    # Penalizace za velmi krátkou paži nebo malý trunk
    length_factor = min(arm_length / 0.15, trunk_length / 0.3, 1.0)
    confidence *= length_factor
    
    # === NÁVRATOVÉ HODNOTY ===
    return {
        'sagittal_angle': sagittal_angle_deg,
        'frontal_angle': frontal_angle_deg,
        'confidence': max(0.0, min(1.0, confidence)),
        'components': {
            'forward': arm_forward_comp,
            'up': arm_up_comp, 
            'right': arm_right_comp
        },
        'coordinate_system': {
            'trunk_up': trunk_up,
            'body_forward': body_forward,
            'shoulder_right': shoulder_right
        },
        'measurements': {
            'trunk_length': trunk_length,
            'arm_length': arm_length,
            'shoulder_width': shoulder_width
        }
    }


def calculate_bilateral_arm_angles(joints):
    """Výpočet pro obě paže současně"""
    
    left_result = calculate_arm_angle_to_trunk_robust(joints, 'left')
    right_result = calculate_arm_angle_to_trunk_robust(joints, 'right')
    
    return {
        'left_arm': left_result,
        'right_arm': right_result,
        'timestamp': time.time()
    }


def analyze_arm_movement_sequence(meshes_pkl_file, output_dir=None):
    """Analýza celé sekvence pohybu paží"""
    
    print(f"ANALYZUJI ARM ANGLES Z: {meshes_pkl_file}")
    print("=" * 60)
    
    with open(meshes_pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    results = []
    valid_frames = 0
    
    for frame_idx, mesh_data in enumerate(meshes):
        joints = mesh_data['joints']  # (117, 3)
        
        frame_result = calculate_bilateral_arm_angles(joints)
        frame_result['frame'] = frame_idx
        
        results.append(frame_result)
        
        # Debug výpis a validace
        left_valid = frame_result['left_arm'] is not None
        right_valid = frame_result['right_arm'] is not None
        
        if left_valid and right_valid:
            valid_frames += 1
            left_sag = frame_result['left_arm']['sagittal_angle']
            right_sag = frame_result['right_arm']['sagittal_angle'] 
            left_conf = frame_result['left_arm']['confidence']
            right_conf = frame_result['right_arm']['confidence']
            
            # Verbose output every 10 frames
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx:3d}: "
                      f"L_sag={left_sag:6.1f}° ({left_conf:.2f}), "
                      f"R_sag={right_sag:6.1f}° ({right_conf:.2f})")
        else:
            print(f"Frame {frame_idx:3d}: INVALID - L_valid={left_valid}, R_valid={right_valid}")
    
    print(f"\nVÝSLEDKY ANALÝZY:")
    print(f"  Celkem snímků: {len(meshes)}")
    print(f"  Validní snímky: {valid_frames}")
    print(f"  Úspěšnost: {valid_frames/len(meshes)*100:.1f}%")
    
    # Export výsledků
    if output_dir:
        export_arm_analysis_results(results, output_dir)
    
    return results


def export_arm_analysis_results(results, output_dir):
    """Export výsledků do CSV a grafů"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Připravit data pro CSV
    csv_data = []
    
    for result in results:
        frame = result['frame']
        
        row = {'frame': frame}
        
        # Left arm data
        if result['left_arm']:
            row['left_sagittal'] = result['left_arm']['sagittal_angle']
            row['left_frontal'] = result['left_arm']['frontal_angle']
            row['left_confidence'] = result['left_arm']['confidence']
            row['left_trunk_length'] = result['left_arm']['measurements']['trunk_length']
            row['left_arm_length'] = result['left_arm']['measurements']['arm_length']
        else:
            row.update({
                'left_sagittal': None, 'left_frontal': None, 'left_confidence': None,
                'left_trunk_length': None, 'left_arm_length': None
            })
        
        # Right arm data  
        if result['right_arm']:
            row['right_sagittal'] = result['right_arm']['sagittal_angle']
            row['right_frontal'] = result['right_arm']['frontal_angle']
            row['right_confidence'] = result['right_arm']['confidence']
            row['right_trunk_length'] = result['right_arm']['measurements']['trunk_length']
            row['right_arm_length'] = result['right_arm']['measurements']['arm_length']
        else:
            row.update({
                'right_sagittal': None, 'right_frontal': None, 'right_confidence': None,
                'right_trunk_length': None, 'right_arm_length': None
            })
        
        csv_data.append(row)
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "arm_angles_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"  CSV export: {csv_file}")
    
    # Create plots
    create_arm_angle_plots(df, output_dir)
    
    # Create statistics
    create_arm_statistics(df, output_dir)


def create_arm_angle_plots(df, output_dir):
    """Vytvořit grafy úhlů paží"""
    
    plt.style.use('default')
    
    # Filter valid data
    valid_left = df.dropna(subset=['left_sagittal'])
    valid_right = df.dropna(subset=['right_sagittal'])
    
    # Plot 1: Sagittal angles over time
    plt.figure(figsize=(12, 6))
    
    if not valid_left.empty:
        plt.plot(valid_left['frame'], valid_left['left_sagittal'], 
                'b-', label='Left Arm Sagittal', alpha=0.7)
    
    if not valid_right.empty:
        plt.plot(valid_right['frame'], valid_right['right_sagittal'], 
                'r-', label='Right Arm Sagittal', alpha=0.7)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (0°)')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Forward flexion (90°)')
    plt.axhline(y=-90, color='orange', linestyle='--', alpha=0.5, label='Backward extension (-90°)')
    
    plt.xlabel('Frame')
    plt.ylabel('Sagittal Angle (degrees)')
    plt.title('Arm Angles in Sagittal Plane (relative to trunk)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / "arm_sagittal_angles.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_file}")
    
    # Plot 2: Confidence scores
    plt.figure(figsize=(12, 4))
    
    if not valid_left.empty:
        plt.plot(valid_left['frame'], valid_left['left_confidence'], 
                'b-', label='Left Arm Confidence', alpha=0.7)
    
    if not valid_right.empty:
        plt.plot(valid_right['frame'], valid_right['right_confidence'], 
                'r-', label='Right Arm Confidence', alpha=0.7)
    
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High confidence (>0.8)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium confidence (>0.5)')
    
    plt.xlabel('Frame')
    plt.ylabel('Confidence Score')
    plt.title('Calculation Confidence Over Time')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    conf_plot = output_dir / "confidence_scores.png"
    plt.savefig(conf_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {conf_plot}")


def create_arm_statistics(df, output_dir):
    """Vytvořit statistiky úhlů"""
    
    stats_file = output_dir / "arm_angle_statistics.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("ARM ANGLE ANALYSIS STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        # Valid frames
        total_frames = len(df)
        valid_left_frames = df['left_sagittal'].notna().sum()
        valid_right_frames = df['right_sagittal'].notna().sum()
        
        f.write(f"DATA VALIDITY:\n")
        f.write(f"  Total frames: {total_frames}\n")
        f.write(f"  Valid left arm: {valid_left_frames} ({valid_left_frames/total_frames*100:.1f}%)\n")
        f.write(f"  Valid right arm: {valid_right_frames} ({valid_right_frames/total_frames*100:.1f}%)\n\n")
        
        # Left arm statistics
        if valid_left_frames > 0:
            left_sag = df['left_sagittal'].dropna()
            f.write(f"LEFT ARM SAGITTAL ANGLES:\n")
            f.write(f"  Mean: {left_sag.mean():.1f}°\n")
            f.write(f"  Median: {left_sag.median():.1f}°\n")
            f.write(f"  Std Dev: {left_sag.std():.1f}°\n")
            f.write(f"  Min: {left_sag.min():.1f}°\n")
            f.write(f"  Max: {left_sag.max():.1f}°\n")
            f.write(f"  Range: {left_sag.max() - left_sag.min():.1f}°\n\n")
        
        # Right arm statistics
        if valid_right_frames > 0:
            right_sag = df['right_sagittal'].dropna()
            f.write(f"RIGHT ARM SAGITTAL ANGLES:\n")
            f.write(f"  Mean: {right_sag.mean():.1f}°\n")
            f.write(f"  Median: {right_sag.median():.1f}°\n")
            f.write(f"  Std Dev: {right_sag.std():.1f}°\n")
            f.write(f"  Min: {right_sag.min():.1f}°\n")
            f.write(f"  Max: {right_sag.max():.1f}°\n")
            f.write(f"  Range: {right_sag.max() - right_sag.min():.1f}°\n\n")
        
        # Confidence statistics
        if valid_left_frames > 0:
            left_conf = df['left_confidence'].dropna()
            f.write(f"LEFT ARM CONFIDENCE:\n")
            f.write(f"  Mean: {left_conf.mean():.3f}\n")
            f.write(f"  Min: {left_conf.min():.3f}\n")
            f.write(f"  High confidence (>0.8): {(left_conf > 0.8).sum()} frames\n\n")
        
        if valid_right_frames > 0:
            right_conf = df['right_confidence'].dropna()
            f.write(f"RIGHT ARM CONFIDENCE:\n")
            f.write(f"  Mean: {right_conf.mean():.3f}\n")
            f.write(f"  Min: {right_conf.min():.3f}\n")
            f.write(f"  High confidence (>0.8): {(right_conf > 0.8).sum()} frames\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("  0° = Arms hanging along trunk\n")
        f.write("  +90° = Arms forward (flexion)\n")
        f.write("  -90° = Arms backward (extension)\n")
        f.write("  Angles are relative to trunk orientation\n")
    
    print(f"  Statistics: {stats_file}")


if __name__ == "__main__":
    # Test with existing PKL file
    pkl_file = "arm_meshes.pkl"
    output_dir = "arm_angle_analysis_results"
    
    if Path(pkl_file).exists():
        results = analyze_arm_movement_sequence(pkl_file, output_dir)
        print(f"\nAnalysis complete! Results saved in: {output_dir}")
    else:
        print(f"PKL file not found: {pkl_file}")
        print("Available files:")
        for pkl in Path(".").glob("*.pkl"):
            print(f"  - {pkl}")
#!/usr/bin/env python3
"""
Trunk Angle Calculator from PKL mesh data
Calculates spine flexion angle from lumbar to cervical vertebrae
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class TrunkAngleCalculator:
    """Calculate trunk flexion angles from SMPL-X joint positions"""
    
    def __init__(self):
        """Initialize calculator with SMPL-X joint mappings for trunk angle only"""
        
        # SMPL-X spine joint indices (only trunk-relevant joints)
        self.joint_names = {
            0: 'pelvis',
            3: 'spine1',  # L3/L4 lumbar vertebrae
            6: 'spine2',  # T12/L1 thoracolumbar junction
            9: 'spine3',  # T1/T2 upper thoracic
            12: 'neck',   # C7/T1 cervical spine
        }
        
        # Primary spine points for trunk angle
        self.lumbar_joint = 3  # spine1 - lower back
        self.cervical_joint = 12  # neck - upper spine
        
        # Alternative more stable measurement
        self.pelvis_joint = 0
        self.upper_spine_joint = 9  # spine3
        
        print("TrunkAngleCalculator initialized")
        print(f"Primary measurement: Joint {self.lumbar_joint} (spine1) to Joint {self.cervical_joint} (neck)")
        print("Focus: Trunk angle measurement only")
    
    def load_pkl_data(self, pkl_path: str) -> List[Dict]:
        """Load mesh data from PKL file"""
        
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"\nLoading PKL file: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            mesh_data = pickle.load(f)
        
        print(f"Loaded {len(mesh_data)} frames")
        
        # Validate data structure
        if mesh_data and len(mesh_data) > 0:
            first_frame = mesh_data[0]
            if 'joints' not in first_frame:
                raise ValueError("PKL file does not contain 'joints' data")
            
            joint_shape = first_frame['joints'].shape
            print(f"Joint data shape: {joint_shape}")
            
            if joint_shape[0] < 22:
                print(f"WARNING: Only {joint_shape[0]} joints found (expected 22+)")
        
        return mesh_data
    
    def calculate_trunk_angle(self, lumbar_point: np.ndarray, cervical_point: np.ndarray) -> float:
        """
        Calculate trunk flexion angle between lumbar and cervical spine points
        
        Returns:
            Angle in degrees:
            - 0° = upright posture (spine vertical)
            - 90° = full forward bend (spine horizontal)
            - Negative = backward bend (extension)
        """
        
        # Vector from lumbar to cervical (spine direction)
        spine_vector = cervical_point - lumbar_point
        
        # Normalize
        spine_length = np.linalg.norm(spine_vector)
        if spine_length < 0.001:  # Avoid division by zero
            return 0.0
        
        spine_unit = spine_vector / spine_length
        
        # Vertical reference vector (Y-up in SMPL-X after our fix)
        # After Y-flip fix in run_production_simple.py, Y is up
        vertical = np.array([0, 1, 0])
        
        # Calculate angle using dot product
        cos_angle = np.dot(spine_unit, vertical)
        
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Angle in radians
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        # Determine if forward or backward bend
        # If spine vector has positive Z component, person is leaning forward
        if spine_vector[2] > 0:
            # Forward bend - keep positive angle
            return angle_deg
        else:
            # Could be backward bend or sideways
            # For now, keep as positive (can enhance later)
            return angle_deg
    
    
    
    
    def process_frames(self, mesh_data: List[Dict]) -> pd.DataFrame:
        """Process all frames and calculate trunk angles with 3D coordinates"""
        
        results = []
        
        print("\nProcessing frames...")
        print("-" * 60)
        
        for frame_idx, frame_data in enumerate(mesh_data):
            
            # Extract joints
            joints = frame_data['joints']
            
            # Get trunk anatomical points only
            lumbar_point = joints[self.lumbar_joint]      # joint 3
            cervical_point = joints[self.cervical_joint]  # joint 12
            
            # Calculate trunk angle only
            trunk_angle = self.calculate_trunk_angle(lumbar_point, cervical_point)
            
            # Store results with trunk-related 3D coordinates + angle
            result = {
                'frame_number': frame_idx,
                
                # 3D coordinates of trunk anatomical points
                'lumbar_x': lumbar_point[0], 'lumbar_y': lumbar_point[1], 'lumbar_z': lumbar_point[2],
                'cervical_x': cervical_point[0], 'cervical_y': cervical_point[1], 'cervical_z': cervical_point[2],
                
                # Calculated trunk angle
                'trunk_angle_degrees': trunk_angle
            }
            
            results.append(result)
            
            # Print progress with trunk-only format
            print(f"Frame {frame_idx:3d}: Trunk={trunk_angle:6.2f}°")
            
            # Detailed output for first few frames  
            if frame_idx < 3:
                print(f"  Lumbar (spine1):   [{lumbar_point[0]:6.3f}, {lumbar_point[1]:6.3f}, {lumbar_point[2]:6.3f}]")
                print(f"  Cervical (neck):   [{cervical_point[0]:6.3f}, {cervical_point[1]:6.3f}, {cervical_point[2]:6.3f}]")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def validate_results(self, df: pd.DataFrame) -> Dict:
        """Validate trunk angle measurements for realism"""
        
        validation = {
            'total_frames': len(df),
            'trunk': {
                'mean_angle': df['trunk_angle_degrees'].mean(),
                'std_angle': df['trunk_angle_degrees'].std(),
                'min_angle': df['trunk_angle_degrees'].min(),
                'max_angle': df['trunk_angle_degrees'].max()
            },
            'warnings': [],
            'checks_passed': []
        }
        
        print("\n" + "=" * 70)
        print("TRUNK ANGLE VALIDATION REPORT")
        print("=" * 70)
        
        # TRUNK ANGLE VALIDATION
        print("\nTRUNK ANGLE VALIDATION")
        print("-" * 40)
        
        trunk_stats = validation['trunk']
        if trunk_stats['min_angle'] < -30:
            validation['warnings'].append(f"Extreme trunk backward bend: {trunk_stats['min_angle']:.1f}°")
        else:
            validation['checks_passed'].append("Trunk minimum angle within normal range")
        
        if trunk_stats['max_angle'] > 120:
            validation['warnings'].append(f"Extreme trunk forward bend: {trunk_stats['max_angle']:.1f}°")
        else:
            validation['checks_passed'].append("Trunk maximum angle within normal range")
        
        # FRAME-TO-FRAME CONTINUITY
        if len(df) > 1:
            trunk_diffs = df['trunk_angle_degrees'].diff().abs()
            max_trunk_jump = trunk_diffs.max()
            
            if max_trunk_jump > 30:
                validation['warnings'].append(f"Large trunk angle jump: {max_trunk_jump:.1f}°")
            else:
                validation['checks_passed'].append("Good frame-to-frame continuity for trunk measurements")
        
        # PRINT DETAILED REPORT
        print(f"Total frames analyzed: {validation['total_frames']}")
        
        print(f"\nTrunk Statistics:")
        print(f"  Mean angle: {trunk_stats['mean_angle']:.2f}° (±{trunk_stats['std_angle']:.2f}°)")
        print(f"  Range: [{trunk_stats['min_angle']:.2f}°, {trunk_stats['max_angle']:.2f}°]")
        
        print("\nValidation Results:")
        print(f"  Checks passed: {len(validation['checks_passed'])}")
        for check in validation['checks_passed']:
            print(f"    + {check}")
        
        if validation['warnings']:
            print(f"\n  Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"    ! {warning}")
        
        return validation
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save trunk angle results to CSV file"""
        
        output_path = Path(output_path)
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\nResults saved to: {output_path}")
        
        # Also save summary statistics for trunk angles only
        stats_path = output_path.with_suffix('.stats.json')
        stats = {
            'total_frames': len(df),
            'trunk_angle_stats': {
                'mean_angle': float(df['trunk_angle_degrees'].mean()),
                'std_angle': float(df['trunk_angle_degrees'].std()),
                'min_angle': float(df['trunk_angle_degrees'].min()),
                'max_angle': float(df['trunk_angle_degrees'].max()),
                'median_angle': float(df['trunk_angle_degrees'].median())
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")
    
    def visualize_angles(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """Create visualization of trunk angles over time"""
        
        fig = plt.figure(figsize=(12, 8))
        
        # Plot 1: Trunk angle over time
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(df['frame_number'], df['trunk_angle_degrees'], 'b-', linewidth=2)
        ax1.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright (0°)')
        ax1.axhline(y=30, color='y', linestyle='--', alpha=0.5, label='Mild flexion (30°)')
        ax1.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='High flexion (60°)')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Trunk Angle (degrees)')
        ax1.set_title('Trunk Flexion Angle Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Trunk angle histogram
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(df['trunk_angle_degrees'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=df['trunk_angle_degrees'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {df["trunk_angle_degrees"].mean():.1f}°')
        ax2.set_xlabel('Trunk Angle (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Trunk Angles')
        ax2.legend()
        
        # Plot 3: Trunk angle statistics box plot
        ax3 = plt.subplot(2, 2, 3)
        ax3.boxplot(df['trunk_angle_degrees'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Trunk Angle (degrees)')
        ax3.set_title('Trunk Angle Distribution Summary')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trunk angle differences (frame-to-frame changes)
        ax4 = plt.subplot(2, 2, 4)
        if len(df) > 1:
            angle_diffs = df['trunk_angle_degrees'].diff().fillna(0)
            ax4.plot(df['frame_number'], angle_diffs, 'g-', linewidth=1, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('Angle Change (degrees)')
            ax4.set_title('Frame-to-Frame Trunk Angle Changes')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Not enough frames\nfor difference plot', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Frame-to-Frame Changes (N/A)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Trunk angle visualization saved to: {output_path}")
        
        plt.close()


def main():
    """Test the trunk angle calculator"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate trunk angles from PKL mesh data')
    parser.add_argument('pkl_file', help='Path to PKL file with mesh data')
    parser.add_argument('--output', default='trunk_angles.csv', help='Output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Create trunk angle visualization')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = TrunkAngleCalculator()
    
    # Load data
    mesh_data = calculator.load_pkl_data(args.pkl_file)
    
    # Process frames (trunk angles only)
    df = calculator.process_frames(mesh_data)
    
    # Validate trunk angles
    validation = calculator.validate_results(df)
    
    # Save results
    calculator.save_results(df, args.output)
    
    # Visualize if requested
    if args.visualize:
        viz_path = Path(args.output).with_suffix('.png')
        calculator.visualize_angles(df, str(viz_path))
    
    print("\nTrunk angle processing complete!")
    
    return df, validation


if __name__ == "__main__":
    df, validation = main()
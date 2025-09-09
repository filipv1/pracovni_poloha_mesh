#!/usr/bin/env python3
"""
Trunk Angle Calculator using SKIN VERTICES
Uses actual skin surface points instead of internal joint positions
Vertex 2151 (cervical/neck area on skin) and 5614 (lumbar/lower back on skin)
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# SKIN VERTEX IDs (on actual skin surface)
CERVICAL_SKIN_VERTEX = 2151  # Neck area on skin
LUMBAR_SKIN_VERTEX = 5614    # Lower back on skin
# Alternative lumbar: 4298

class TrunkAngleCalculatorSkin:
    """Calculate trunk flexion angles using skin surface vertices"""
    
    def __init__(self, lumbar_vertex=5614, cervical_vertex=2151):
        """
        Initialize calculator with skin vertex IDs
        
        Args:
            lumbar_vertex: Vertex ID for lower back on skin (default 5614, alt: 4298)
            cervical_vertex: Vertex ID for neck area on skin (default 2151)
        """
        self.lumbar_vertex = lumbar_vertex
        self.cervical_vertex = cervical_vertex
        
        print("TrunkAngleCalculatorSkin initialized")
        print(f"Using SKIN SURFACE vertices:")
        print(f"  Lumbar (lower back): Vertex {self.lumbar_vertex}")
        print(f"  Cervical (neck): Vertex {self.cervical_vertex}")
        print("These are actual points on the skin, not internal joints!")
    
    def load_pkl_data(self, pkl_path: str) -> List[Dict]:
        """Load mesh data from PKL file"""
        
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"\nLoading PKL file: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Handle both old and new PKL format
        if isinstance(pkl_data, dict) and 'mesh_sequence' in pkl_data:
            # New format with metadata
            meshes = pkl_data['mesh_sequence']
            metadata = pkl_data.get('metadata', {})
            print(f"  New PKL format detected with metadata")
            if 'fps' in metadata:
                print(f"  FPS from PKL: {metadata['fps']:.2f}")
            if 'video_filename' in metadata:
                print(f"  Original video: {metadata['video_filename']}")
        else:
            # Old format - just mesh sequence
            meshes = pkl_data
            print(f"  Old PKL format detected")
        
        print(f"Loaded {len(meshes)} frames")
        
        # Verify vertices exist
        if meshes:
            first_frame = meshes[0]
            num_vertices = len(first_frame['vertices'])
            print(f"Mesh has {num_vertices} vertices")
            
            if self.lumbar_vertex >= num_vertices:
                print(f"WARNING: Lumbar vertex {self.lumbar_vertex} out of range!")
            if self.cervical_vertex >= num_vertices:
                print(f"WARNING: Cervical vertex {self.cervical_vertex} out of range!")
        
        return meshes
    
    def calculate_trunk_angle(self, vertices: np.ndarray) -> Dict:
        """
        Calculate trunk angle using skin vertices
        
        Args:
            vertices: Vertex positions array
            
        Returns:
            Dictionary with angle data
        """
        
        # Get skin surface points
        lumbar_point = vertices[self.lumbar_vertex]
        cervical_point = vertices[self.cervical_vertex]
        
        # Calculate trunk vector (from lower back to neck on skin)
        trunk_vector = cervical_point - lumbar_point
        trunk_length = np.linalg.norm(trunk_vector)
        
        if trunk_length < 0.01:  # Too short
            return None
        
        # Normalize
        trunk_unit = trunk_vector / trunk_length
        
        # Calculate angle to vertical (Y-axis in SMPL-X)
        vertical = np.array([0, 1, 0])
        
        # Dot product for angle
        cos_angle = np.dot(trunk_unit, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # Calculate forward/backward component
        forward_component = trunk_vector[2] / trunk_length  # Z is forward/back
        
        return {
            'angle': angle_deg,
            'trunk_length': trunk_length,
            'forward_lean': forward_component > 0,
            'lumbar_pos': lumbar_point.tolist(),
            'cervical_pos': cervical_point.tolist(),
            'trunk_vector': trunk_vector.tolist()
        }
    
    def calculate_angles_from_pkl(self, pkl_path: str) -> pd.DataFrame:
        """Calculate trunk angles for all frames"""
        
        meshes = self.load_pkl_data(pkl_path)
        
        print(f"\nCalculating trunk angles using skin vertices...")
        
        results = []
        
        for frame_idx, mesh_data in enumerate(meshes):
            vertices = mesh_data['vertices']
            
            angle_data = self.calculate_trunk_angle(vertices)
            
            if angle_data:
                results.append({
                    'frame': frame_idx,
                    'trunk_angle': angle_data['angle'],
                    'trunk_length': angle_data['trunk_length'],
                    'forward_lean': angle_data['forward_lean'],
                    'lumbar_x': angle_data['lumbar_pos'][0],
                    'lumbar_y': angle_data['lumbar_pos'][1],
                    'lumbar_z': angle_data['lumbar_pos'][2],
                    'cervical_x': angle_data['cervical_pos'][0],
                    'cervical_y': angle_data['cervical_pos'][1],
                    'cervical_z': angle_data['cervical_pos'][2]
                })
            else:
                results.append({
                    'frame': frame_idx,
                    'trunk_angle': 0,
                    'trunk_length': 0,
                    'forward_lean': False
                })
            
            if frame_idx % 50 == 0 or frame_idx < 5:
                if angle_data:
                    print(f"Frame {frame_idx:3d}: Angle = {angle_data['angle']:.1f}°")
        
        df = pd.DataFrame(results)
        
        print(f"\nProcessed {len(df)} frames")
        print(f"Average trunk angle: {df['trunk_angle'].mean():.1f}°")
        print(f"Range: {df['trunk_angle'].min():.1f}° to {df['trunk_angle'].max():.1f}°")
        
        return df
    
    def export_statistics(self, df: pd.DataFrame, output_file: str = "trunk_angle_skin_statistics.txt"):
        """Export detailed statistics"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TRUNK ANGLE STATISTICS (SKIN-BASED)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Using skin vertices:\n")
            f.write(f"  Lumbar vertex: {self.lumbar_vertex}\n")
            f.write(f"  Cervical vertex: {self.cervical_vertex}\n\n")
            f.write(f"Total frames analyzed: {len(df)}\n\n")
            
            f.write("ANGLE STATISTICS:\n")
            f.write(f"  Mean: {df['trunk_angle'].mean():.2f}°\n")
            f.write(f"  Std Dev: {df['trunk_angle'].std():.2f}°\n")
            f.write(f"  Min: {df['trunk_angle'].min():.2f}°\n")
            f.write(f"  Max: {df['trunk_angle'].max():.2f}°\n")
            f.write(f"  25th percentile: {df['trunk_angle'].quantile(0.25):.2f}°\n")
            f.write(f"  50th percentile (median): {df['trunk_angle'].quantile(0.50):.2f}°\n")
            f.write(f"  75th percentile: {df['trunk_angle'].quantile(0.75):.2f}°\n\n")
            
            # Posture classification
            f.write("POSTURE DISTRIBUTION:\n")
            upright = len(df[df['trunk_angle'] < 15])
            slight_bend = len(df[(df['trunk_angle'] >= 15) & (df['trunk_angle'] < 30)])
            moderate_bend = len(df[(df['trunk_angle'] >= 30) & (df['trunk_angle'] < 60)])
            severe_bend = len(df[df['trunk_angle'] >= 60])
            
            total = len(df)
            f.write(f"  Upright (<15°): {upright} frames ({100*upright/total:.1f}%)\n")
            f.write(f"  Slight bend (15-30°): {slight_bend} frames ({100*slight_bend/total:.1f}%)\n")
            f.write(f"  Moderate bend (30-60°): {moderate_bend} frames ({100*moderate_bend/total:.1f}%)\n")
            f.write(f"  Severe bend (≥60°): {severe_bend} frames ({100*severe_bend/total:.1f}%)\n\n")
            
            # Forward/backward lean
            forward = len(df[df['forward_lean'] == True])
            backward = len(df[df['forward_lean'] == False])
            f.write("LEAN DIRECTION:\n")
            f.write(f"  Forward lean: {forward} frames ({100*forward/total:.1f}%)\n")
            f.write(f"  Backward lean: {backward} frames ({100*backward/total:.1f}%)\n")
        
        print(f"\nStatistics saved to: {output_file}")
    
    def save_to_csv(self, df: pd.DataFrame, output_file: str = "trunk_angles_skin.csv"):
        """Save angles to CSV"""
        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")

def main():
    """Main execution with both vertex options"""
    
    import sys
    
    # Default PKL file
    pkl_file = "arm_meshes.pkl"
    
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    
    if not Path(pkl_file).exists():
        print(f"ERROR: PKL file not found: {pkl_file}")
        print("Usage: python trunk_angle_calculator_skin.py [pkl_file]")
        return
    
    print("\nSKIN-BASED TRUNK ANGLE CALCULATOR")
    print("=" * 50)
    print("Choose lumbar vertex:")
    print("1. Vertex 5614 (default)")
    print("2. Vertex 4298 (alternative)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        calculator = TrunkAngleCalculatorSkin(lumbar_vertex=4298, cervical_vertex=2151)
        suffix = "_4298"
    else:
        calculator = TrunkAngleCalculatorSkin(lumbar_vertex=5614, cervical_vertex=2151)
        suffix = "_5614"
    
    # Calculate angles
    df = calculator.calculate_angles_from_pkl(pkl_file)
    
    # Export results
    calculator.export_statistics(df, f"trunk_angle_skin_statistics{suffix}.txt")
    calculator.save_to_csv(df, f"trunk_angles_skin{suffix}.csv")
    
    print("\n" + "=" * 50)
    print("SKIN-BASED ANALYSIS COMPLETE!")
    print(f"Files created:")
    print(f"  - trunk_angle_skin_statistics{suffix}.txt")
    print(f"  - trunk_angles_skin{suffix}.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analýza frame skip z PKL souboru a statistik
"""

import pickle
import json
from pathlib import Path

def analyze_frame_skip(pkl_file, stats_file=None):
    """Analyzuj frame skip z dostupných dat"""
    
    print("ANALYZING FRAME SKIP")
    print("=" * 40)
    
    # Load PKL data
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"Meshes in PKL: {len(meshes)}")
    
    # Load stats if available
    if stats_file and Path(stats_file).exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        frames_processed = stats.get('frames_processed', 0)
        meshes_generated = stats.get('meshes_generated', len(meshes))
        
        print(f"Stats data:")
        print(f"  Total frames processed: {frames_processed}")
        print(f"  Successful meshes: {meshes_generated}")
        print(f"  Success rate: {meshes_generated/frames_processed*100:.1f}%")
        
        # Estimate frame skip
        if meshes_generated > 0 and frames_processed > 0:
            estimated_skip = frames_processed / meshes_generated
            print(f"\nESTIMATED FRAME SKIP: {estimated_skip:.1f}")
            
            # Common frame skip values
            common_skips = [1, 2, 5, 10, 15, 20, 25, 30, 50, 60]
            closest_skip = min(common_skips, key=lambda x: abs(x - estimated_skip))
            print(f"Closest common skip: {closest_skip}")
            
        return estimated_skip if 'estimated_skip' in locals() else None
    else:
        print("No stats file available - cannot determine frame skip precisely")
        print(f"Only know: {len(meshes)} successful frames generated")
        return None

def analyze_temporal_spacing(pkl_file):
    """Analyzuj časové rozestupy mezi framy (pokud jsou dostupné metadata)"""
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"\nTEMPORAL ANALYSIS")
    print("-" * 30)
    
    # Check if we have any frame indices or timestamps
    for i, mesh in enumerate(meshes[:5]):  # First 5 frames
        fitting_error = mesh.get('fitting_error', 0)
        print(f"Frame {i:2d}: error={fitting_error:.6f}")
        
        # Look for any frame-related metadata
        for key, value in mesh.items():
            if 'frame' in key.lower() or 'time' in key.lower():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # Analyze your data
    pkl_file = "simple_results/test_meshes.pkl"
    stats_file = "simple_results/test_stats.json"
    
    if Path(pkl_file).exists():
        frame_skip = analyze_frame_skip(pkl_file, stats_file)
        analyze_temporal_spacing(pkl_file)
        
        print(f"\nSUMMARY:")
        print(f"  PKL file: {pkl_file}")
        print(f"  Frames in sequence: {len(pickle.load(open(pkl_file, 'rb')))}")
        if frame_skip:
            print(f"  Estimated frame skip: {frame_skip:.1f}")
    else:
        print(f"PKL file not found: {pkl_file}")
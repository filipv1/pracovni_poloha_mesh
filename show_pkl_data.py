#!/usr/bin/env python3
"""
Zobrazení obsahu PKL souboru s mesh daty
"""

import pickle
import numpy as np

def show_pkl_content(pkl_file):
    """Zobraz obsah PKL souboru"""
    
    print(f"LOADING: {pkl_file}")
    print("=" * 50)
    
    with open(pkl_file, 'rb') as f:
        meshes = pickle.load(f)
    
    print(f"TOTAL FRAMES: {len(meshes)}")
    print()
    
    # Zobraz první frame detailně
    if meshes:
        mesh = meshes[0]
        print("FRAME 0 STRUCTURE:")
        print("-" * 30)
        
        for key, value in mesh.items():
            if hasattr(value, 'shape'):
                print(f"  {key:15}: shape={value.shape}, dtype={value.dtype}")
                
                # Ukaz náhled dat
                if len(value.shape) == 2 and value.shape[0] <= 5:
                    print(f"    Preview: {value}")
                elif len(value.shape) == 2:
                    print(f"    First 3 rows:\n{value[:3]}")
                elif len(value.shape) == 1:
                    print(f"    Values: {value}")
                    
            elif isinstance(value, dict):
                print(f"  {key:15}: dict with keys {list(value.keys())}")
                for subkey, subval in value.items():
                    if hasattr(subval, 'shape'):
                        print(f"    {subkey:12}: shape={subval.shape}")
            else:
                print(f"  {key:15}: {type(value).__name__} = {value}")
        
        print()
        
        # Zobraz summary všech framů
        print("ALL FRAMES SUMMARY:")
        print("-" * 30)
        
        for i, mesh in enumerate(meshes[:10]):  # Prvních 10 framů
            error = mesh.get('fitting_error', 0)
            vertex_count = mesh.get('vertex_count', 0)
            print(f"  Frame {i:2d}: {vertex_count:5d} vertices, error={error:.6f}")
        
        if len(meshes) > 10:
            print(f"  ... and {len(meshes) - 10} more frames")
    
    print()
    print("DONE!")

if __name__ == "__main__":
    show_pkl_content("simple_results/test_meshes.pkl")
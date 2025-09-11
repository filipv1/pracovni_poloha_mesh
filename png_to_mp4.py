#!/usr/bin/env python3
"""
PNG sequence to MP4 converter using OpenCV
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def png_sequence_to_mp4(png_pattern, output_path, fps=25):
    """Convert PNG sequence to MP4 using OpenCV"""
    
    # Find all PNG files matching pattern
    pattern_path = Path(png_pattern)
    parent_dir = pattern_path.parent
    
    if "%" in str(pattern_path):
        # Handle format like temp_frame_%04d.png
        base_name = str(pattern_path.name).replace("%04d", "")
        png_files = sorted([f for f in parent_dir.glob("*.png") if base_name.split('%')[0] in f.name])
    else:
        png_files = sorted(parent_dir.glob(pattern_path.name))
    
    if not png_files:
        print(f"ERROR: No PNG files found matching {png_pattern}")
        return False
    
    print(f"Found {len(png_files)} PNG files")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(png_files[0]))
    if first_img is None:
        print(f"ERROR: Cannot read {png_files[0]}")
        return False
    
    height, width, channels = first_img.shape
    print(f"Video resolution: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("ERROR: Cannot create video writer")
        return False
    
    # Process each PNG
    for i, png_file in enumerate(png_files):
        img = cv2.imread(str(png_file))
        if img is not None:
            writer.write(img)
            print(f"  Added frame {i+1}/{len(png_files)}")
        else:
            print(f"WARNING: Cannot read {png_file}")
    
    # Release resources
    writer.release()
    
    # Check output file
    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Video created: {output_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[ERROR] Output file not created: {output_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PNG sequence to MP4")
    parser.add_argument("pattern", help="PNG pattern (e.g., temp_frame_%04d.png)")
    parser.add_argument("output", help="Output MP4 file")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    
    args = parser.parse_args()
    
    success = png_sequence_to_mp4(args.pattern, args.output, args.fps)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
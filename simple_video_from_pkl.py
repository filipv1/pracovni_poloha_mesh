#!/usr/bin/env python3
"""
Simplified PKL to video converter
Creates video directly from mesh vertices without OBJ export
"""

import numpy as np
import pickle
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse

def load_pkl_data(pkl_path):
    """Load PKL data"""
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # Handle both formats
    if isinstance(pkl_data, dict) and 'mesh_sequence' in pkl_data:
        meshes = pkl_data['mesh_sequence']
        metadata = pkl_data.get('metadata', {})
        fps = metadata.get('fps', 30.0)
    else:
        meshes = pkl_data
        fps = 30.0
    
    return meshes, fps

def create_simple_video(pkl_file, output_mp4, quality='medium'):
    """Create video using matplotlib instead of Blender"""
    
    print(f"Loading {pkl_file}...")
    meshes, fps = load_pkl_data(pkl_file)
    num_frames = len(meshes)
    print(f"Loaded {num_frames} frames at {fps} FPS")
    
    # Quality settings
    quality_settings = {
        'low': {'dpi': 50, 'figsize': (6, 6)},
        'medium': {'dpi': 80, 'figsize': (8, 8)},
        'high': {'dpi': 100, 'figsize': (10, 10)}
    }
    settings = quality_settings[quality]
    
    # Create figure
    fig = plt.figure(figsize=settings['figsize'])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    # Setup axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    
    def update_frame(frame_idx):
        ax.clear()
        
        # Get mesh data
        mesh = meshes[frame_idx]
        vertices = mesh['vertices']
        joints = mesh['joints']
        
        # Plot mesh points (subsample for speed)
        step = 50  # Show every 50th vertex
        ax.scatter(vertices[::step, 0], 
                  vertices[::step, 1], 
                  vertices[::step, 2], 
                  c='gray', s=1, alpha=0.3)
        
        # Plot joints
        ax.scatter(joints[:, 0], 
                  joints[:, 1], 
                  joints[:, 2], 
                  c='red', s=10)
        
        # Add trunk vector
        lumbar = joints[3]  # spine1
        cervical = joints[12]  # neck
        ax.plot([lumbar[0], cervical[0]],
               [lumbar[1], cervical[1]],
               [lumbar[2], cervical[2]], 
               'r-', linewidth=3)
        
        # Title
        ax.set_title(f'Frame {frame_idx}/{num_frames}')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])
        
        return ax
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update_frame, frames=min(num_frames, 100), 
                        interval=1000/fps, blit=False)
    
    # Save as MP4
    print(f"Saving to {output_mp4}...")
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(output_mp4, writer=writer, dpi=settings['dpi'])
    
    plt.close()
    print(f"Done! Video saved to {output_mp4}")

def main():
    parser = argparse.ArgumentParser(description="Simple PKL to video converter")
    parser.add_argument("pkl_file", help="Input PKL file")
    parser.add_argument("output_mp4", help="Output MP4 file")
    parser.add_argument("--quality", choices=['low', 'medium', 'high'], 
                       default='medium', help="Quality preset")
    
    args = parser.parse_args()
    
    create_simple_video(args.pkl_file, args.output_mp4, args.quality)

if __name__ == "__main__":
    main()
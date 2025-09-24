#!/usr/bin/env python3
"""
Generate 4 Synchronized Videos from PKL Mesh Data
Uses pre-computed mesh data from PKL file to generate videos
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import pickle
import json
import mediapipe as mp
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, List, Tuple


class VideoGeneratorFromPKL:
    """Generate 4 synchronized videos from PKL mesh data"""
    
    def __init__(self):
        """Initialize video generator"""
        print("INITIALIZING VIDEO GENERATOR")
        print("=" * 50)
        
        # Initialize MediaPipe for videos 1 and 2
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("MediaPipe initialized")
        print("Ready to generate videos!")
    
    def generate_videos(self, input_video_path, pkl_path, output_dir, max_frames=None):
        """Generate 4 videos from input video and PKL mesh data"""
        
        input_path = Path(input_video_path)
        pkl_path = Path(pkl_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGENERATING 4 VIDEOS")
        print(f"Input video: {input_path}")
        print(f"Mesh data: {pkl_path}")
        print(f"Output directory: {output_dir}")
        
        # Load mesh data from PKL
        with open(pkl_path, 'rb') as f:
            mesh_data_list = pickle.load(f)
        print(f"Loaded {len(mesh_data_list)} meshes from PKL")
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Prepare output videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Video 1: Original
        video1_path = output_dir / f"{input_path.stem}_1_original.mp4"
        video1 = cv2.VideoWriter(str(video1_path), fourcc, fps, (width, height))
        
        # Video 2: MediaPipe skeleton overlay
        video2_path = output_dir / f"{input_path.stem}_2_mediapipe.mp4"
        video2 = cv2.VideoWriter(str(video2_path), fourcc, fps, (width, height))
        
        # Video 3: 3D mesh in white space
        video3_path = output_dir / f"{input_path.stem}_3_mesh3d.mp4"
        video3 = cv2.VideoWriter(str(video3_path), fourcc, fps, (width, height))
        
        # Video 4: 3D mesh overlay on original
        video4_path = output_dir / f"{input_path.stem}_4_mesh_overlay.mp4"
        video4 = cv2.VideoWriter(str(video4_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        mesh_idx = 0
        
        print("\nPROCESSING FRAMES...")
        print("-" * 30)
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Frame {frame_idx}/{total_frames}")
            
            # Video 1: Original frame
            video1.write(frame)
            
            # Process with MediaPipe for Video 2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Video 2: MediaPipe overlay
                frame2 = frame.copy()
                self.mp_drawing.draw_landmarks(
                    frame2, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2
                    )
                )
                video2.write(frame2)
            else:
                video2.write(frame)
            
            # Videos 3 and 4: Use mesh data from PKL
            if mesh_idx < len(mesh_data_list):
                mesh_data = mesh_data_list[mesh_idx]
                
                # Video 3: 3D mesh in white space
                frame3 = self._render_3d_mesh_whitespace(mesh_data, width, height)
                video3.write(frame3)
                
                # Video 4: 3D mesh overlay on original
                frame4 = self._render_mesh_overlay(frame, mesh_data, width, height)
                video4.write(frame4)
                
                mesh_idx += 1
            else:
                # No more mesh data - use white frame for video 3, original for video 4
                video3.write(np.ones((height, width, 3), dtype=np.uint8) * 255)
                video4.write(frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        video1.release()
        video2.release()
        video3.release()
        video4.release()
        
        print("\n4 VIDEOS GENERATED:")
        print(f"  1. Original: {video1_path}")
        print(f"  2. MediaPipe: {video2_path}")
        print(f"  3. 3D Mesh: {video3_path}")
        print(f"  4. Mesh Overlay: {video4_path}")
        print(f"Processed {frame_idx} frames")
        
        return {
            'video1': video1_path,
            'video2': video2_path,
            'video3': video3_path,
            'video4': video4_path,
            'frames_processed': frame_idx
        }
    
    def _render_3d_mesh_whitespace(self, mesh_data, width, height):
        """Render 3D mesh in white space"""
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Get vertices and faces
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Use natural orientation from SMPL-X (no forced rotation)
        # Just swap Y and Z for matplotlib 3D (Y is up in SMPL-X, Z is up in matplotlib)
        vertices_natural = vertices.copy()
        vertices_natural[:, [1, 2]] = vertices_natural[:, [2, 1]]  # Swap Y and Z for 3D view
        
        # Center and scale vertices
        vertices_centered = vertices_natural - vertices_natural.mean(axis=0)
        scale = 2.0 / (vertices_centered.max() - vertices_centered.min())
        vertices_scaled = vertices_centered * scale
        
        # Create mesh collection
        mesh_faces = []
        for face in faces:
            face_vertices = vertices_scaled[face]
            mesh_faces.append(face_vertices)
        
        # Create collection and add to plot
        mesh_collection = Poly3DCollection(
            mesh_faces,
            facecolors='lightblue',
            edgecolors='gray',
            linewidths=0.1,
            alpha=0.8
        )
        ax.add_collection3d(mesh_collection)
        
        # Also plot joints if available
        if 'joints' in mesh_data:
            joints = mesh_data['joints']
            # Apply same transformation as vertices (no forced rotation)
            joints_natural = joints.copy()
            joints_natural[:, [1, 2]] = joints_natural[:, [2, 1]]
            joints_centered = joints_natural - joints_natural.mean(axis=0)
            joints_scaled = joints_centered * scale
            ax.scatter(joints_scaled[:, 0], joints_scaled[:, 1], joints_scaled[:, 2],
                      c='red', s=20, alpha=0.9)
        
        # Set view - slightly angled to see 3D shape better
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=10, azim=45)  # Slight angle to see orientation
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')
        
        # Render to array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize to video dimensions
        img_resized = cv2.resize(img_array, (width, height))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    
    def _render_mesh_overlay(self, frame, mesh_data, width, height):
        """Render 3D mesh overlay on original frame"""
        
        # Create matplotlib figure with transparent background
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Make background transparent
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        
        # Get vertices and faces
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Use natural orientation (no forced rotation)
        vertices_transformed = vertices.copy()
        # Swap Y and Z for matplotlib 3D
        vertices_transformed[:, [1, 2]] = vertices_transformed[:, [2, 1]]
        
        # Center the mesh
        vertices_transformed = vertices_transformed - vertices_transformed.mean(axis=0)
        
        # Scale and position to match video
        scale = 3.0  # Adjust scale to match person size
        vertices_transformed *= scale
        vertices_transformed[:, 1] -= 0.5  # Move down a bit to align with person
        vertices_transformed[:, 2] += 5   # Move forward in Z
        
        # Create mesh collection
        mesh_faces = []
        for face in faces:
            face_vertices = vertices_transformed[face]
            mesh_faces.append(face_vertices)
        
        # Create semi-transparent mesh
        mesh_collection = Poly3DCollection(
            mesh_faces,
            facecolors='cyan',
            edgecolors='blue',
            linewidths=0.1,
            alpha=0.3
        )
        ax.add_collection3d(mesh_collection)
        
        # Set view to match camera - looking straight at the person
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([2, 8])
        ax.view_init(elev=0, azim=0)
        
        # Remove all axes and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')
        
        # Render to array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize mesh render
        img_resized = cv2.resize(img_array, (width, height))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        # Create mask for non-white pixels (the mesh)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Apply mesh only where mask is active
        result = frame.copy()
        mesh_pixels = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        result = cv2.addWeighted(result, 0.7, mesh_pixels, 0.3, 0)
        
        return result


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 4 videos from PKL mesh data')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('pkl_file', help='PKL file with mesh data')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = VideoGeneratorFromPKL()
    
    # Generate videos
    results = generator.generate_videos(
        args.input_video,
        args.pkl_file,
        args.output_dir,
        max_frames=args.max_frames
    )
    
    print("\nSUCCESS!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()

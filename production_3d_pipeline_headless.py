#!/usr/bin/env python3
"""
RUNPOD HEADLESS VERSION - 3D Human Mesh Pipeline
Optimized for server environments without display
"""

# Copy the fixed production_3d_pipeline_clean.py content here
# This ensures RunPod compatibility

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
from pathlib import Path
import json
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Set environment for headless operation
os.environ['DISPLAY'] = ':99'
os.environ['MPLBACKEND'] = 'Agg'

print("🚀 RUNPOD HEADLESS 3D PIPELINE READY")
print("=" * 60)

def main():
    """RunPod optimized main function"""
    
    print("📍 Running in headless mode (server environment)")
    
    # Check if we're on RunPod
    if os.path.exists('/workspace'):
        print("✅ RunPod environment detected")
        os.chdir('/workspace/pracovni_poloha_mesh')
    
    # Import the main pipeline (will use fixed visualization)
    exec(open('production_3d_pipeline_clean.py').read())

if __name__ == "__main__":
    main()
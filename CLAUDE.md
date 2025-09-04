# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Environment Setup
```bash
# Always activate the conda environment first
conda activate trunk_analysis

# For quick testing of the 3D pipeline
python quick_test_3_frames.py

# For trunk analysis (2D pose detection)
python pracovni_poloha2/main.py input.mp4 output.mp4
```

### Testing Commands
```bash
# Test 3D mesh pipeline (production-ready)
python production_3d_pipeline_clean.py

# Test minimal MediaPipe functionality
python quick_mediapipe_test.py

# Validate complete pipeline
python validate_complete_pipeline.py

# Test specific components
python test_pipeline_init.py
python test_fitting_only.py
python test_render_only.py

# 3D Arm Angle Analysis
python export_arm_analysis_with_angles.py

# Debug arm angle calculations
python debug_arm_sides.py
python test_arm_angles.py

# Combined angle CSV generation
python create_combined_angles_csv.py fil_vid_meshes.pkl combined_angles.csv

# Neck angle analysis with stable algorithm
python neck_angle_calculator_like_arm.py
```

### RunPod Deployment
```bash
# Automated GPU setup for RunPod
python setup_runpod.py

# Alternative conda setup
python setup_runpod_conda.py
```

## Architecture Overview

This repository contains three main systems:

### 1. 3D Human Mesh Pipeline (Production)
- **Entry Point**: `production_3d_pipeline_clean.py` (main pipeline)
- **Technology**: SMPL-X + MediaPipe + Open3D + PyTorch
- **Purpose**: Convert MediaPipe 33 3D landmarks → SMPL-X 3D human mesh → Professional visualization
- **Output**: 3D mesh animations, high-quality renders, mesh data export

**Key Components:**
- `PreciseMediaPipeConverter` - Converts MediaPipe landmarks to SMPL-X format
- `SMPLXFittingOptimizer` - 3-stage optimization (global pose → body pose → refinement)
- `Open3DRenderer` - Professional visualization and animation generation
- `MasterPipeline` - Orchestrates the complete workflow

### 2. Trunk Analysis System (2D Analysis)
- **Entry Point**: `pracovni_poloha2/main.py`
- **Technology**: MediaPipe + OpenCV + numpy
- **Purpose**: Analyze trunk bending angles from video using 2D pose estimation
- **Output**: Annotated videos with angle measurements, CSV export

**Key Components:**
- `TrunkAnalysisProcessor` - Main processing pipeline
- `PoseDetector` - MediaPipe pose detection wrapper
- `TrunkAngleCalculator` - Angle computation and analysis
- `SkeletonVisualizer` - Video overlay visualization

### 3. Advanced Angle Analysis System (NEW)
- **Bilateral Arm Angles**: `arm_angle_calculator.py` - Robust anatomical angle calculation
- **Neck Angles**: `neck_angle_calculator_like_arm.py` - Stable neck angle using arm algorithm
- **Combined Export**: `create_combined_angles_csv.py` - Simple CSV with all angles
- **Enhanced Visualization**: `export_arm_analysis_with_angles.py` - OBJ with angle data

**Key Features:**
- Anatomically correct coordinate systems
- Handles all body orientations robustly
- Sagittal and frontal plane measurements
- Frame-stable calculations with confidence scoring

## Dependencies

### Core Requirements
- Python 3.9 (conda environment: `trunk_analysis`)
- MediaPipe 0.10.8
- OpenCV 4.8.1.78
- NumPy 1.24.3

### 3D Pipeline Additional Requirements
- PyTorch 2.0+ (GPU: CUDA 11.8+)
- SMPL-X 0.1.28+
- Open3D 0.18.0+
- Trimesh 4.0.0+

### SMPL-X Models (Required for 3D Pipeline)
Download from SMPL-X official site and place in `models/smplx/`:
- `SMPLX_NEUTRAL.npz`
- `SMPLX_MALE.npz` 
- `SMPLX_FEMALE.npz`

## Processing Pipeline Flow

### 3D Mesh Generation
```
Input Video → MediaPipe (33 3D landmarks) → SMPL-X Fitting → Open3D Rendering → Export (PKL/OBJ)
```

### Trunk Analysis (2D)
```
Input Video → MediaPipe (33 2D landmarks) → Angle Calculation → Visualization → Output
```

### 3D Angle Analysis
```
PKL Mesh Data → Joint Extraction → Anatomical Coordinate System → Multi-angle Calculation → CSV/OBJ Export
```

## Performance Characteristics

### Hardware Requirements
- **CPU Processing**: Intel GPU (33s/frame for 3D pipeline)
- **GPU Processing**: RTX 4090 (2-3s/frame), RTX 3090 (3-4s/frame)
- **Memory**: 8GB+ RAM, 6GB+ VRAM (for GPU processing)

### Quality Modes (3D Pipeline)
- `ultra`: Maximum accuracy, longest processing time
- `high`: Balanced accuracy and speed
- `medium`: Faster processing, good quality

## Common Development Patterns

### Pipeline Initialization
```python
from production_3d_pipeline_clean import MasterPipeline

pipeline = MasterPipeline(device='cuda')  # or 'cpu'
results = pipeline.execute_full_pipeline(
    'input_video.mp4',
    output_dir='results',
    quality='ultra'
)
```

### Trunk Analysis Usage (2D Pipeline)
```python
from pracovni_poloha2.main import parse_arguments
from pracovni_poloha2.src.trunk_analyzer import TrunkAnalysisProcessor

processor = TrunkAnalysisProcessor(
    input_path='video.mp4',
    output_path='output.mp4',
    export_csv=True
)
```

### Angle Analysis (3D Pipeline)
```python
# Trunk angles
from trunk_angle_calculator import TrunkAngleCalculator
calculator = TrunkAngleCalculator()
angles_data = calculator.calculate_angles_from_pkl('meshes.pkl')
calculator.export_statistics('trunk_angle_statistics.txt')

# Combined angles (trunk + neck + arms)
from create_combined_angles_csv import create_combined_angles_csv
create_combined_angles_csv('fil_vid_meshes.pkl', 'combined_angles.csv')

# Bilateral arm angles
from arm_angle_calculator import calculate_bilateral_arm_angles
result = calculate_bilateral_arm_angles(joints_3d)
left_angle = result['left_arm']['sagittal_angle']  # 0°=hanging, +90°=forward, -90°=backward
right_angle = result['right_arm']['sagittal_angle']
```

### Export and Analysis Tools
```bash
# Calculate trunk angles from existing mesh data
python trunk_angle_calculator.py

# Create combined CSV with all angles
python create_combined_angles_csv.py fil_vid_meshes.pkl

# Export to Blender with enhanced visualization
python export_trunk_vectors_with_angle_to_blender.py
python export_to_blender.py

# Blender import scripts in blender_export/:
# - side_by_side_arm_and_trunk_sequence.py
# - combined_mesh_and_trunk_sequence.py
# - trunk_vector_sequence.py

# Interactive 3D visualization
python interactive_3d_viewer.py

# Analyze frame skip performance
python analyze_frame_skip.py
```

### Data Export Capabilities
The pipeline supports multiple export formats:
- **PKL files**: Complete mesh sequence data (`*_meshes.pkl`)
- **OBJ sequences**: Individual frame meshes (`trunk_analysis_export/trunk_analysis_XXXX.obj`)
- **Blender integration**: Direct export to Blender format with animation
- **CSV data**: Angle measurements and statistics
- **Visualization**: Professional 3D renders and animations

## Important Notes

- Always use `conda activate trunk_analysis` before running any scripts
- SMPL-X models must be manually downloaded and placed in `models/smplx/`
- For GPU processing, ensure CUDA 11.8+ compatibility
- Use RunPod RTX 4090 for optimal 3D pipeline performance
- The repository includes extensive documentation in multiple `.md` files for specific use cases
- Large OBJ export sequences (400+ files) are generated in `trunk_analysis_export/` directory
- Trunk angle statistics are saved as `trunk_angle_statistics.txt`
- Combined angle data (trunk+neck+arms) exported as CSV for analysis

## SMPL-X Joint Indices Reference
Key joints used in calculations:
- 0: pelvis (root)
- 3: spine1 (lumbar L3/L4)
- 6: spine2 (mid spine)
- 9: spine3 (upper spine)
- 12: neck (cervical C7/T1)
- 15: head
- 16: right_shoulder
- 17: left_shoulder
- 18: right_elbow
- 19: left_elbow
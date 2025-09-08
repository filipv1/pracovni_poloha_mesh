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

# Simple production run
python run_production_simple.py
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

### Skin-based Analysis Commands (NEW)
```bash
# Skin-based trunk angle calculation
python trunk_angle_calculator_skin.py

# Skin-based neck angle calculation
python neck_angle_calculator_skin.py

# Combined skin-based angles CSV
python create_combined_angles_csv_skin.py fil_vid_meshes.pkl skin_angles.csv

# Export all vectors with skin vertices to Blender
python export_all_vectors_skin_to_blender.py

# Blender skin vectors animation
python blender_skin_vectors_animation.py
```

### RunPod Deployment
```bash
# Automated GPU setup for RunPod
python setup_runpod.py

# Alternative conda setup
python setup_runpod_conda.py
```

## Architecture Overview

This repository contains four main systems:

### 1. 3D Human Mesh Pipeline (Production)
- **Entry Point**: `production_3d_pipeline_clean.py` (main pipeline)
- **Simple Runner**: `run_production_simple.py` (simplified execution)
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

### 3. Advanced Angle Analysis System (Joint-based)
- **Bilateral Arm Angles**: `arm_angle_calculator.py` - Robust anatomical angle calculation
- **Neck Angles**: `neck_angle_calculator_like_arm.py` - Stable neck angle using arm algorithm
- **Combined Export**: `create_combined_angles_csv.py` - Simple CSV with all angles
- **Enhanced Visualization**: `export_arm_analysis_with_angles.py` - OBJ with angle data

**Key Features:**
- Anatomically correct coordinate systems
- Handles all body orientations robustly
- Sagittal and frontal plane measurements
- Frame-stable calculations with confidence scoring

### 4. Skin-based Analysis System (NEW)
- **Purpose**: Calculate angles using actual skin vertices instead of internal joints
- **Trunk Angles**: `trunk_angle_calculator_skin.py` - Uses surface vertices for trunk
- **Neck Angles**: `neck_angle_calculator_skin.py` - Uses head and neck skin vertices
- **Combined Export**: `create_combined_angles_csv_skin.py` - All skin-based angles
- **Blender Export**: `export_all_vectors_skin_to_blender.py` - Visualize vectors in Blender

**Key Advantages:**
- More accurate representation of visible body posture
- Better correlation with visual appearance
- Includes skin deformation effects
- Enhanced visualization capabilities

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

### 3D Angle Analysis (Joint-based)
```
PKL Mesh Data → Joint Extraction → Anatomical Coordinate System → Multi-angle Calculation → CSV/OBJ Export
```

### 3D Angle Analysis (Skin-based)
```
PKL Mesh Data → Skin Vertex Selection → Vector Calculation → Angle Computation → CSV/Blender Export
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

### Joint-based Angle Analysis (3D Pipeline)
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

### Skin-based Angle Analysis (3D Pipeline)
```python
# Skin-based trunk angles
from trunk_angle_calculator_skin import TrunkAngleCalculatorSkin
calculator = TrunkAngleCalculatorSkin()
angles_data = calculator.calculate_angles_from_pkl('meshes.pkl')

# Skin-based neck angles
from neck_angle_calculator_skin import NeckAngleCalculatorSkin
calculator = NeckAngleCalculatorSkin()
angles_data = calculator.calculate_angles_from_pkl('meshes.pkl')

# Combined skin-based angles
from create_combined_angles_csv_skin import create_combined_angles_csv_skin
create_combined_angles_csv_skin('meshes.pkl', 'skin_angles.csv')
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
python export_all_vectors_to_blender.py  # All vectors (joint-based)
python export_all_vectors_skin_to_blender.py  # All vectors (skin-based)

# Blender import scripts in blender_export/:
# - side_by_side_arm_and_trunk_sequence.py
# - combined_mesh_and_trunk_sequence.py
# - trunk_vector_sequence.py

# Blender animation scripts:
# - blender_fixed_animation.py  # Fixed animation with shape consistency
# - blender_skin_vectors_animation.py  # Skin-based vector animation
# - blender_ultra_fast_animation.py  # Optimized fast animation

# Interactive 3D visualization
python interactive_3d_viewer.py

# Analyze frame skip performance
python analyze_frame_skip.py

# Show PKL data structure
python show_pkl_data.py

# Repair SMPL-X joints if needed
python repair_smplx_joints.py
```

### Data Export Capabilities
The pipeline supports multiple export formats:
- **PKL files**: Complete mesh sequence data (`*_meshes.pkl`)
- **OBJ sequences**: Individual frame meshes (`trunk_analysis_export/trunk_analysis_XXXX.obj`)
- **Blender integration**: Direct export to Blender format with animation
- **CSV data**: Angle measurements and statistics (joint-based and skin-based)
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
- Skin-based analysis provides more visually accurate angle measurements

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

## Blender Integration Notes
- Blender scripts require Blender 3.0+ with Python API
- Animation scripts handle both joint-based and skin-based visualizations
- Use `blender_ultra_fast_animation.py` for large datasets (optimized performance)
- Shape consistency maintained across all frames in newer scripts (`blender_fixed_animation.py`)
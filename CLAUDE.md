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

# Debug and validate data
python show_pkl_data.py
python explore_joints.py
```

### Linting and Quality Checks
```bash
# No specific lint commands found - add as needed for code quality
# Consider adding: ruff check, black, mypy when implementing
```

### RunPod Deployment
```bash
# Automated GPU setup for RunPod
python setup_runpod.py

# Alternative conda setup
python setup_runpod_conda.py
```

## Architecture Overview

This repository contains two main systems:

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

### 3. 3D Arm Angle Analysis System (NEW)
- **Entry Point**: `export_arm_analysis_with_angles.py`
- **Technology**: SMPL-X joint data + anatomical coordinate systems + 3D vector math
- **Purpose**: Calculate bilateral arm angles relative to trunk orientation with high precision
- **Output**: Enhanced OBJ sequences with angle data, comprehensive statistics, Blender-ready visualization

**Key Components:**
- `arm_angle_calculator.py` - Robust anatomical angle calculation engine
- `export_arm_analysis_with_angles.py` - Enhanced export with trunk+arm vector combinations
- `debug_arm_sides.py` - Joint assignment verification tool
- `test_arm_angles.py` - Synthetic pose validation suite
- `neck_angle_calculator_like_arm.py` - Stable neck angle calculator using arm calculator logic

### 4. Data Analysis and Visualization Tools
- **Entry Points**: Multiple analysis and export scripts
- **Technology**: NumPy + Matplotlib + Blender integration + CSV export
- **Purpose**: Extract insights from processed mesh data and create visualizations
- **Output**: Statistics, interactive viewers, Blender animations, CSV reports

**Key Components:**
- `trunk_angle_calculator.py` - Extract trunk bending statistics from mesh data
- `create_combined_angles_csv.py` - Combine multiple angle data sources into unified CSV
- `interactive_3d_viewer.py` - Real-time 3D mesh visualization
- `analyze_frame_skip.py` - Performance analysis for optimization

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

### 3D Trunk Angle Analysis
```
PKL Mesh Data → Joint Extraction → Trunk Vector Calculation → Angle Analysis → Statistics Export
```

### 3D Arm Angle Analysis (NEW)
```
PKL Mesh Data → Joint Extraction → Anatomical Coordinate System → Arm-to-Trunk Angle Calculation → Enhanced Visualization Export
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

### Trunk Angle Analysis (3D Pipeline)
```python
from trunk_angle_calculator import TrunkAngleCalculator

calculator = TrunkAngleCalculator()
angles_data = calculator.calculate_angles_from_pkl('meshes.pkl')
calculator.export_statistics('trunk_angle_statistics.txt')
```

### 3D Arm Angle Analysis (NEW)
```python
from arm_angle_calculator import calculate_bilateral_arm_angles
from export_arm_analysis_with_angles import create_enhanced_arm_analysis_export

# Calculate arm angles for single frame
result = calculate_bilateral_arm_angles(joints_3d)
left_angle = result['left_arm']['sagittal_angle']  # 0°=hanging, +90°=forward, -90°=backward
right_angle = result['right_arm']['sagittal_angle']

# Export complete enhanced analysis
create_enhanced_arm_analysis_export('arm_meshes.pkl', 'enhanced_arm_analysis_export')
```

### Export and Analysis Tools
```bash
# Calculate trunk angles from existing mesh data
python trunk_angle_calculator.py

# Calculate bilateral arm angles with trunk reference
python export_arm_analysis_with_angles.py

# Combine multiple angle data sources into unified CSV
python create_combined_angles_csv.py

# Export to Blender with enhanced visualization
python export_trunk_vectors_with_angle_to_blender.py
python export_to_blender.py

# Blender import for enhanced arm+trunk visualization
# Load: blender_export/side_by_side_arm_and_trunk_sequence.py

# Interactive 3D visualization
python interactive_3d_viewer.py

# Analyze frame skip performance
python analyze_frame_skip.py

# Generate mesh video outputs
python generate_mesh_video.py
python generate_4videos_from_pkl.py
```

### Data Export Capabilities
The pipeline supports multiple export formats:
- **PKL files**: Complete mesh sequence data (`*_meshes.pkl`)
- **OBJ sequences**: Individual frame meshes (`trunk_analysis_export/trunk_analysis_XXXX.obj`)
- **Blender integration**: Direct export to Blender format with animation
- **CSV data**: Angle measurements and statistics
- **Visualization**: Professional 3D renders and animations

### Blender Integration Workflow
```
PKL Mesh Data → Trunk Vector Analysis → Blender Export → Professional Animation
```

## File Structure and Key Locations

### Main Pipeline Files
- `production_3d_pipeline_clean.py` - Primary 3D mesh generation pipeline (production-ready)
- `pracovni_poloha2/main.py` - 2D trunk analysis entry point
- `arm_angle_calculator.py` - Core arm angle calculation engine
- `trunk_angle_calculator.py` - 3D trunk angle analysis tool

### Test and Validation Scripts
- `quick_test_3_frames.py` - Fast pipeline validation (3 frames)
- `validate_complete_pipeline.py` - Full system validation
- `test_arm_angles.py` - Arm angle calculation validation

### Setup and Deployment
- `setup_runpod.py` - Automated RunPod GPU environment setup
- `setup_runpod_conda.py` - Conda-based RunPod setup (recommended)
- `requirements_runpod.txt` - GPU-optimized dependencies

### Export and Analysis
- `create_combined_angles_csv.py` - Unified CSV export for all angle data
- `interactive_3d_viewer.py` - Real-time 3D mesh visualization
- `blender_export/` - Blender integration scripts directory

### Documentation
- `README.md` - Project overview and quick start
- `RUNPOD_DEPLOYMENT_GUIDE.md` - Comprehensive GPU deployment guide
- `FINAL_IMPLEMENTATION_REPORT.md` - Technical implementation details
- `TROUBLESHOOTING.md` - Common issues and solutions

## Important Notes

- Always use `conda activate trunk_analysis` before running any scripts
- SMPL-X models must be manually downloaded and placed in `models/smplx/`
- For GPU processing, ensure CUDA 11.8+ compatibility
- Use RunPod RTX 4090 for optimal 3D pipeline performance (2-3s/frame)
- Large OBJ export sequences (400+ files) are generated in `trunk_analysis_export/` directory
- Trunk angle statistics are saved as `trunk_angle_statistics.txt`
- Enhanced arm analysis creates `enhanced_arm_analysis_export/` directory with combined visualizations
- All PKL files contain complete mesh sequence data and can be analyzed independently
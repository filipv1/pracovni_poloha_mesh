# Quick Start Guide: EasyMoCap + PyTorch3D + SMPL-X Pipeline

## Overview
This guide provides step-by-step instructions to set up and run the maximum accuracy 3D human mesh fitting pipeline combining EasyMoCap, PyTorch3D, and SMPL-X models.

## Prerequisites

### Hardware Requirements
- **GPU**: RTX 3060 (8GB) or better (RTX 4090 recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB free space
- **OS**: Windows 10/11, Ubuntu 18.04+, or macOS 10.15+

### Software Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- [Git](https://git-scm.com/downloads)
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-toolkit) (for GPU acceleration)

## Step 1: Automated Setup

### Option A: Full GPU Setup (Recommended)
```bash
# Clone or download the pipeline files
cd your_project_directory

# Run automated setup
python setup_environment.py --mode full

# This will:
# - Create conda environment 'trunk_analysis'
# - Install all dependencies with GPU support
# - Download and configure EasyMoCap
# - Create configuration files
# - Set up directory structure
```

### Option B: CPU-Only Setup (for testing/limited hardware)
```bash
python setup_environment.py --mode cpu-only
```

### Option C: Minimal Setup (essential components only)
```bash
python setup_environment.py --mode minimal
```

## Step 2: Manual Model Download

The setup script will create `models/download_instructions.json` with download links. You must manually download model files due to licensing requirements:

### Required Downloads:
1. **SMPL-X Models**: Register at https://smpl-x.is.tue.mpg.de/
   - Download: SMPLX_NEUTRAL.pkl, SMPLX_MALE.pkl, SMPLX_FEMALE.pkl
   - Place in: `models/smplx/`

2. **FLAME Model** (optional, for face details): Register at https://flame.is.tue.mpg.de/
   - Download: FLAME_NEUTRAL.pkl
   - Place in: `models/flame/`

3. **MANO Models** (optional, for hands): Register at https://mano.is.tue.mpg.de/
   - Download: MANO_LEFT.pkl, MANO_RIGHT.pkl  
   - Place in: `models/mano/`

### Directory Structure After Download:
```
your_project/
├── models/
│   ├── smplx/
│   │   ├── SMPLX_NEUTRAL.pkl
│   │   ├── SMPLX_MALE.pkl
│   │   └── SMPLX_FEMALE.pkl
│   ├── flame/
│   │   └── FLAME_NEUTRAL.pkl
│   └── mano/
│       ├── MANO_LEFT.pkl
│       └── MANO_RIGHT.pkl
├── configs/
├── scripts/
└── implementation_architecture.py
```

## Step 3: Test Installation

```bash
# Activate the conda environment
conda activate trunk_analysis

# Run installation test
python scripts/test_installation.py

# Expected output:
# === Installation Verification Tests ===
# ✓ OpenCV imported successfully
# ✓ MediaPipe imported successfully
# ✓ PyTorch setup working
# ✓ PyTorch3D functional
# ✓ SMPL-X library available
# === Test Results: 5/5 tests passed ===
```

## Step 4: Process Your First Video

### Prepare Input Video
- **Format**: MP4, AVI, MOV
- **Resolution**: 720p to 4K (1080p recommended)
- **Duration**: Start with 10-30 seconds for testing
- **Content**: Single person, clearly visible, good lighting
- **Frame rate**: 30 FPS recommended

### Run Processing
```bash
# Activate environment
conda activate trunk_analysis

# Run simple example
python scripts/simple_example.py

# Or use the full implementation directly
python -c "
from implementation_architecture import CompletePipeline, PipelineConfig

config = PipelineConfig(
    input_video_path='your_video.mp4',
    output_dir='results',
    smplx_model_path='models/smplx',
    render_resolution=1024,
    use_gpu=True
)

pipeline = CompletePipeline(config)
results = pipeline.process_video(config.input_video_path)
print('Results:', results)
"
```

### Expected Processing Time
- **30-second video on RTX 4090**: 5-10 minutes
- **30-second video on RTX 3060**: 15-25 minutes  
- **CPU-only processing**: 60-120 minutes

## Step 5: Review Results

After processing completes, check the output directory:

```
results/
├── keypoints.json          # Extracted 2D keypoints
├── smplx_params.pt         # Optimized SMPL-X parameters  
├── rendered_mesh.mp4       # Final rendered video
└── meshes/                 # Sample mesh files (.obj)
    ├── mesh_frame_0000.obj
    ├── mesh_frame_0010.obj
    └── ...
```

### Quality Assessment
- **Rendered video**: Should show smooth, realistic human motion
- **Mesh files**: Can be opened in Blender, MeshLab, or other 3D software
- **SMPL-X parameters**: Can be used for further analysis or animation

## Troubleshooting Common Issues

### 1. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA issues persist, fall back to CPU
python setup_environment.py --mode cpu-only
```

### 2. Memory Issues
- Reduce `batch_size` in config (16 → 8 → 4)
- Lower `render_resolution` (1024 → 512)
- Process shorter video segments

### 3. EasyMoCap Import Errors
- The pipeline includes fallback implementations
- Check `troubleshooting_guide.md` for detailed solutions

### 4. Model File Issues
- Ensure all .pkl files are in correct directories
- Check file sizes (SMPLX_NEUTRAL.pkl should be ~100MB)
- Verify downloads completed successfully

## Performance Optimization

### For Better Speed:
```python
config = PipelineConfig(
    batch_size=32,              # Larger batches (if GPU memory allows)
    render_resolution=512,      # Lower resolution for speed
    temporal_smoothing=False,   # Disable for faster processing
)
```

### For Better Quality:
```python
config = PipelineConfig(
    render_resolution=2048,     # Higher resolution
    temporal_smoothing=True,    # Enable smoothing
    mediapipe_confidence=0.8,   # Higher confidence threshold
)
```

## Advanced Usage

### Custom Configuration
Edit `configs/pipeline_config.json` to customize:
- Processing parameters
- Quality settings  
- Output formats
- Optimization weights

### Batch Processing
```python
# Process multiple videos
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
for video in videos:
    config.input_video_path = video
    config.output_dir = f'results_{Path(video).stem}'
    results = pipeline.process_video(video)
```

### Integration with Other Tools
- Export meshes to Blender for animation
- Use SMPL-X parameters in Unity/Unreal Engine
- Convert to other formats (FBX, GLTF) using Blender scripts

## Next Steps

1. **Experiment with different videos** to understand system capabilities
2. **Adjust parameters** based on your specific use case
3. **Explore advanced features** in the full implementation
4. **Check the troubleshooting guide** for optimization tips
5. **Consider cloud processing** for larger projects

## Support and Resources

- **Troubleshooting**: See `troubleshooting_guide.md`
- **Implementation details**: Review `implementation_architecture.py`
- **Model comparisons**: Check `smplx_comparison.md`
- **GPU optimization**: See `pytorch3d_analysis.md`

## Estimated Costs

### Hardware Costs (if upgrading):
- **RTX 4090**: $1,600 (recommended for professional use)
- **RTX 4070**: $600 (good balance of price/performance)
- **RTX 3060**: $300 (minimum for GPU acceleration)

### Cloud Processing Costs:
- **RunPod RTX 4090**: ~$0.50/hour
- **Google Colab Pro**: $10/month (limited GPU hours)
- **AWS P3 instances**: ~$3/hour

### Processing Cost Estimates:
- **30-second video**: $0.10-0.25 on cloud GPU
- **2-minute video**: $0.25-0.75 on cloud GPU
- **1-hour video**: $5-15 on cloud GPU (batch processing recommended)
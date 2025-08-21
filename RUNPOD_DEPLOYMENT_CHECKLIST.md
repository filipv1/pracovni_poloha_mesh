# 🚀 RUNPOD DEPLOYMENT CHECKLIST
## MediaPipe → 3D Human Mesh Pipeline

Complete checklist for deploying your 3D mesh pipeline on RunPod GPU.

---

## ✅ PRE-DEPLOYMENT PREPARATION (LOCAL)

### Step 1: Local Validation
Run these scripts locally before RunPod deployment:

```bash
# 1. Complete pipeline validation
python validate_complete_pipeline.py

# 2. Quick MediaPipe test  
python quick_mediapipe_test.py

# 3. Test existing functionality
cd pracovni_poloha2
python test_simple.py
```

**✅ Requirements:**
- [ ] All validation tests pass
- [ ] MediaPipe detection success rate > 60%
- [ ] Input video ready and tested locally
- [ ] All dependencies identified

### Step 2: SMPL-X Models Preparation
**✅ Download Requirements:**
- [ ] Register at https://smpl-x.is.tue.mpg.de/
- [ ] Download SMPL-X v1.1 models:
  - [ ] SMPLX_NEUTRAL.npz
  - [ ] SMPLX_MALE.npz  
  - [ ] SMPLX_FEMALE.npz
- [ ] Verify files are not corrupted (can load with numpy)
- [ ] Prepare for upload to RunPod

### Step 3: Code Repository
**✅ GitHub Repository Ready:**
- [ ] All code pushed to GitHub
- [ ] pracovni_poloha2 folder complete (not empty)
- [ ] All documentation files present
- [ ] Repository clone URL ready

---

## 🎮 RUNPOD SETUP PROCESS

### Step 1: RunPod Instance Creation
**✅ Instance Configuration:**
- [ ] GPU Instance selected (RTX 4090 or A40 recommended)
- [ ] PyTorch template selected
- [ ] Sufficient disk space (20GB+ recommended)
- [ ] Network ports configured if needed

### Step 2: Initial RunPod Setup
**✅ First Commands:**
```bash
# 1. Update system
apt update

# 2. Clone repository
git clone https://github.com/filipv1/pracovni_poloha_mesh.git
cd pracovni_poloha_mesh

# 3. Verify conda is available
conda --version

# 4. Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**✅ Verification:**
- [ ] Repository cloned successfully
- [ ] Conda is available and working
- [ ] CUDA is detected by PyTorch
- [ ] pracovni_poloha2 folder has content

### Step 3: Environment Setup
**✅ Automated Setup:**
```bash
# Run the automated setup script
python setup_runpod_conda.py

# If successful, activate environment
./activate_mesh_env.sh

# Verify installation
python -c "import mediapipe, torch, open3d, smplx; print('All packages available')"
```

**✅ Manual Setup (if automated fails):**
```bash
# Create environment manually
conda create -n mesh_env python=3.9 -y
conda activate mesh_env

# Install packages
pip install mediapipe opencv-python numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open3d==0.18.0
pip install smplx matplotlib pillow
```

**✅ Environment Verification:**
- [ ] mesh_env environment created
- [ ] All packages installed without errors
- [ ] CUDA available in environment
- [ ] Python version is 3.9.x

---

## 📁 FILE UPLOAD & ORGANIZATION

### Step 1: SMPL-X Models Upload
**✅ Upload Process:**
```bash
# Create models directory
mkdir -p models/smplx

# Upload SMPL-X models (use RunPod file upload or scp)
# Place files in: models/smplx/
# - SMPLX_NEUTRAL.npz
# - SMPLX_MALE.npz
# - SMPLX_FEMALE.npz
```

**✅ Verification:**
```bash
# Check model files
ls -la models/smplx/
python -c "import numpy as np; print(list(np.load('models/smplx/SMPLX_NEUTRAL.npz').keys()))"
```

- [ ] All 3 SMPL-X model files present
- [ ] Files can be loaded without errors
- [ ] File sizes are reasonable (50-100MB each)

### Step 2: Test Video Upload
**✅ Video Preparation:**
```bash
# Upload your test video as input_video.mp4
# Or use scp/rsync for large files
```

**✅ Video Verification:**
```bash
# Check video properties
python -c "
import cv2
cap = cv2.VideoCapture('input_video.mp4')
print(f'Resolution: {int(cap.get(3))}x{int(cap.get(4))}')
print(f'FPS: {cap.get(5):.2f}')
print(f'Frames: {int(cap.get(7))}')
cap.release()
"
```

- [ ] Input video uploaded successfully
- [ ] Video can be opened by OpenCV
- [ ] Video properties are reasonable

---

## 🧪 PIPELINE TESTING ON RUNPOD

### Step 1: Quick Functionality Test
**✅ Basic Tests:**
```bash
# Activate environment
./activate_mesh_env.sh

# Quick MediaPipe test
python quick_mediapipe_test.py

# 3-frame mesh test
python quick_test_3_frames.py
```

**✅ Success Criteria:**
- [ ] MediaPipe detects poses successfully
- [ ] 3D mesh generation works without errors
- [ ] GPU is being utilized (check nvidia-smi)
- [ ] Output files are created

### Step 2: Full Pipeline Test
**✅ Production Pipeline:**
```bash
# Run complete pipeline
python production_3d_pipeline_clean.py

# Monitor GPU usage
nvidia-smi -l 1  # In separate terminal
```

**✅ Expected Outputs:**
- [ ] input_video_3d_animation.mp4 (3D mesh video)
- [ ] input_video_meshes.pkl (mesh data)
- [ ] input_video_final_mesh.png (visualization)
- [ ] input_video_stats.json (processing stats)

### Step 3: Export Validation
**✅ Mesh Export Test:**
```bash
# Test mesh export functionality
python export_mesh_formats.py

# Check exported files
ls -la outputs/mesh_exports/
```

**✅ Export Verification:**
- [ ] OBJ files created successfully
- [ ] NumPy arrays exported
- [ ] Files have reasonable sizes
- [ ] No corruption errors

---

## ⚡ PERFORMANCE OPTIMIZATION

### Step 1: GPU Utilization Check
**✅ Performance Monitoring:**
```bash
# Monitor during processing
nvidia-smi -l 1
htop  # CPU usage
```

**✅ Optimization Settings:**
- [ ] GPU utilization > 80%
- [ ] No CUDA out of memory errors  
- [ ] Processing speed acceptable (check FPS)
- [ ] Memory usage stable

### Step 2: Processing Settings
**✅ Configuration Tuning:**
```python
# Edit production_3d_pipeline_clean.py if needed:
config = {
    'quality': 'ultra',        # For best results
    'frame_skip': 1,          # Process every frame
    'batch_size': 16,         # Adjust based on GPU memory
    'device': 'cuda'          # Ensure GPU usage
}
```

**✅ Performance Targets:**
- [ ] Processing speed: > 5 FPS on RTX 4090
- [ ] Memory usage: < 90% of GPU memory
- [ ] No memory leaks over time
- [ ] Stable processing without crashes

---

## 🎯 PRODUCTION DEPLOYMENT

### Step 1: Batch Processing Setup
**✅ Multiple Video Processing:**
```bash
# For multiple videos, create batch script
mkdir -p batch_inputs
# Place multiple videos in batch_inputs/

# Process all videos
for video in batch_inputs/*.mp4; do
    echo "Processing: $video"
    python production_3d_pipeline_clean.py --input "$video"
done
```

### Step 2: Output Management
**✅ Results Organization:**
```bash
# Create organized output structure
mkdir -p results/{videos,meshes,exports}

# Move outputs to organized folders
mv *_3d_animation.mp4 results/videos/
mv *_meshes.pkl results/meshes/
mv outputs/mesh_exports/* results/exports/
```

### Step 3: Monitoring & Logs
**✅ Process Monitoring:**
```bash
# Enable logging
export PYTHONUNBUFFERED=1

# Run with logging
python production_3d_pipeline_clean.py 2>&1 | tee processing.log

# Monitor system resources
watch -n 5 'nvidia-smi; echo "---"; df -h'
```

---

## 🚨 TROUBLESHOOTING CHECKLIST

### If Something Goes Wrong:
**✅ Diagnostic Steps:**
1. [ ] Check troubleshooting guide: `cat TROUBLESHOOTING.md`
2. [ ] Run validation script: `python validate_complete_pipeline.py`
3. [ ] Check GPU memory: `nvidia-smi`
4. [ ] Verify environment: `conda list`
5. [ ] Check disk space: `df -h`
6. [ ] Review error logs: `tail -n 50 processing.log`

### Common Issues:
**✅ Quick Fixes:**
- [ ] **CUDA out of memory**: Reduce batch_size, add `torch.cuda.empty_cache()`
- [ ] **Import errors**: Reinstall conda environment
- [ ] **Video errors**: Check FFmpeg installation: `ffmpeg -version`
- [ ] **Model errors**: Verify SMPL-X files integrity
- [ ] **Permission errors**: Check file permissions: `chmod +x *.py`

---

## ✅ DEPLOYMENT COMPLETION

### Final Verification:
**✅ Success Criteria:**
- [ ] Complete pipeline runs without errors
- [ ] Output quality is satisfactory
- [ ] Processing speed meets expectations
- [ ] All exports work correctly
- [ ] System is stable under load

### Documentation:
**✅ Results Documentation:**
- [ ] Processing logs saved
- [ ] Performance metrics recorded  
- [ ] Output samples validated
- [ ] Any issues documented for future reference

### Next Steps:
**✅ Post-Deployment:**
- [ ] Set up regular monitoring
- [ ] Plan for scaling if needed
- [ ] Consider automation for batch processing
- [ ] Document any custom modifications

---

## 🎉 SUCCESS! 

Your MediaPipe → 3D Human Mesh pipeline is now fully deployed on RunPod!

**Ready for Production:**
- Full video processing capability
- GPU-accelerated 3D mesh generation  
- Professional visualization output
- Computational analysis export
- Scalable batch processing

**Pipeline Outputs:**
- 3D mesh videos (MP4)
- Mesh data (PKL/NumPy)
- Analysis formats (OBJ/PLY)
- Processing statistics (JSON)

**Your complete workflow is now operational! 🚀**
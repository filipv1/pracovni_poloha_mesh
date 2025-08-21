# 🔧 TROUBLESHOOTING GUIDE
## MediaPipe → 3D Human Mesh Pipeline

This guide covers common issues and their solutions for the complete pipeline.

---

## 🐍 PYTHON & ENVIRONMENT ISSUES

### Issue: Python Version Compatibility
**Error:** `ImportError: No module named 'mediapipe'` or version conflicts

**Solution:**
```bash
# Check Python version
python --version

# MediaPipe requires Python 3.8-3.11 (3.9 recommended)
# If wrong version, create conda environment:
conda create -n mesh_env python=3.9
conda activate mesh_env

# Install requirements
pip install mediapipe opencv-python numpy torch torchvision
pip install open3d matplotlib smplx pillow
```

### Issue: Open3D DLL Loading Errors on Windows
**Error:** `ImportError: DLL load failed while importing open3d`

**Solution:**
```bash
# Install specific Open3D version for Python 3.9
pip uninstall open3d
pip install open3d==0.18.0

# Alternative: Use conda
conda install -c open3d-admin open3d
```

### Issue: CUDA/GPU Detection Problems
**Error:** `RuntimeError: CUDA out of memory` or `torch.cuda.is_available() returns False`

**Solutions:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# For RunPod: Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU fallback, edit production_3d_pipeline_clean.py:
# Change: device='cuda' to device='cpu'
```

---

## 📹 VIDEO & MEDIAPIPE ISSUES

### Issue: Video File Cannot Be Opened
**Error:** `Cannot open input video` or `VideoCapture returns False`

**Solutions:**
1. **Check file format:** Ensure video is in supported format (MP4, AVI, MOV)
   ```bash
   # Convert video if needed
   ffmpeg -i input.mov -c:v libx264 -c:a aac input.mp4
   ```

2. **Check file path:** Use absolute paths or place video in project root
   ```python
   # Correct path examples:
   input_path = "C:/full/path/to/video.mp4"  # Windows absolute
   input_path = "./input_video.mp4"          # Relative to current directory
   ```

3. **Check file corruption:** Try playing video in media player first

### Issue: Low MediaPipe Detection Success Rate
**Error:** `Detection success rate: 20%` or frequent `No pose detected`

**Solutions:**
1. **Improve video quality:**
   - Ensure good lighting
   - Person should be clearly visible
   - Avoid heavy motion blur
   - Minimum resolution: 640x480

2. **Adjust detection parameters:**
   ```python
   detector = PoseDetector(
       model_complexity=2,        # Increase from 1 to 2
       min_detection_confidence=0.3,  # Decrease from 0.5
       min_tracking_confidence=0.3    # Decrease from 0.5
   )
   ```

3. **Check pose visibility:**
   - Full body should be visible in frame
   - Person should face camera (not profile)
   - Avoid occlusions

### Issue: Unicode Encoding Errors
**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:**
```python
# Add to top of Python scripts:
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Or run with encoding:
python -X utf8 script.py
```

---

## 🎭 SMPL-X & 3D MESH ISSUES

### Issue: Missing SMPL-X Model Files
**Error:** `FileNotFoundError: SMPL-X model file not found`

**Solutions:**
1. **Download SMPL-X models:**
   - Visit: https://smpl-x.is.tue.mpg.de/
   - Register and download models
   - Place in `models/smplx/` directory

2. **Required files:**
   ```
   models/smplx/
   ├── SMPLX_NEUTRAL.npz
   ├── SMPLX_MALE.npz
   └── SMPLX_FEMALE.npz
   ```

3. **Verify file integrity:**
   ```python
   import numpy as np
   model = np.load('models/smplx/SMPLX_NEUTRAL.npz')
   print(list(model.keys()))  # Should show model parameters
   ```

### Issue: 3D Mesh Fitting Fails
**Error:** `RuntimeError during mesh optimization` or poor mesh quality

**Solutions:**
1. **Reduce optimization complexity:**
   ```python
   # In production_3d_pipeline_clean.py, reduce iterations:
   optimization_config = {
       'num_iterations': [50, 50, 50],  # Reduce from [100, 100, 100]
       'learning_rate': 1e-2
   }
   ```

2. **Check landmark quality:**
   - Run `quick_mediapipe_test.py` first
   - Ensure consistent pose detection
   - Verify landmark stability

3. **Memory issues:**
   ```python
   # Process in smaller batches
   batch_size = 10  # Reduce if memory errors
   ```

---

## 💾 OUTPUT & EXPORT ISSUES

### Issue: Video Output Not Created
**Error:** `Failed to create output video` or empty output file

**Solutions:**
1. **Install FFmpeg:**
   ```bash
   # Windows: Download from https://ffmpeg.org/
   # Or use conda:
   conda install ffmpeg
   
   # Linux:
   sudo apt-get install ffmpeg
   
   # Verify installation:
   ffmpeg -version
   ```

2. **Check output permissions:**
   ```python
   import os
   os.makedirs('outputs', exist_ok=True)
   # Ensure write permissions in output directory
   ```

3. **Fallback to image sequence:**
   ```python
   # If video fails, export as PNG sequence
   for i, frame in enumerate(frames):
       cv2.imwrite(f'outputs/frame_{i:04d}.png', frame)
   ```

### Issue: Mesh Export Fails
**Error:** `Cannot export mesh to OBJ format`

**Solutions:**
1. **Check output directory:**
   ```bash
   mkdir -p outputs/mesh_exports
   ```

2. **Verify mesh data:**
   ```python
   print(f"Vertices shape: {vertices.shape}")
   print(f"Faces shape: {faces.shape}")
   # Should be: (10475, 3) and (20908, 3) for SMPL-X
   ```

3. **Use alternative export:**
   ```python
   # Save as NumPy if OBJ fails
   np.save('outputs/vertices.npy', vertices)
   np.save('outputs/faces.npy', faces)
   ```

---

## 🚀 RUNPOD DEPLOYMENT ISSUES

### Issue: Conda Environment Setup Fails on RunPod
**Error:** `conda: command not found` or environment creation fails

**Solutions:**
1. **Initialize conda:**
   ```bash
   # On RunPod, initialize conda first
   conda init bash
   source ~/.bashrc
   
   # Then run setup
   python setup_runpod_conda.py
   ```

2. **Manual environment setup:**
   ```bash
   # Create environment manually if script fails
   conda create -n mesh_env python=3.9 -y
   conda activate mesh_env
   
   # Install packages one by one
   pip install mediapipe
   pip install opencv-python
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install open3d==0.18.0
   pip install smplx matplotlib pillow numpy
   ```

### Issue: GPU Memory Errors on RunPod
**Error:** `CUDA out of memory` during processing

**Solutions:**
1. **Reduce batch processing:**
   ```python
   # Process fewer frames at once
   frame_skip = 2  # Process every 2nd frame
   batch_size = 5  # Reduce batch size
   ```

2. **Use gradient checkpointing:**
   ```python
   # Add to model initialization
   model.gradient_checkpointing = True
   ```

3. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## 🔍 DEBUGGING UTILITIES

### Quick Diagnosis Script
Run this to quickly identify issues:

```python
# Save as debug_pipeline.py
import sys, os, subprocess

def quick_debug():
    print("=== QUICK PIPELINE DEBUG ===")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Check critical imports
    imports = ['cv2', 'mediapipe', 'torch', 'open3d', 'numpy']
    for pkg in imports:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - MISSING")
    
    # Check CUDA
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except:
        print("✗ CUDA check failed")
    
    # Check files
    files = ['input_video.mp4', 'models/smplx/SMPLX_NEUTRAL.npz']
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")

if __name__ == "__main__":
    quick_debug()
```

---

## 📋 COMMON ERROR PATTERNS

### Pattern 1: Import Errors
```
ImportError: No module named 'xyz'
→ Install missing package: pip install xyz
→ Check Python environment is activated
```

### Pattern 2: File Path Errors
```
FileNotFoundError: [Errno 2] No such file or directory
→ Use absolute paths
→ Check file exists: os.path.exists(path)
→ Create directories: os.makedirs(dir, exist_ok=True)
```

### Pattern 3: Memory Errors
```
RuntimeError: CUDA out of memory
→ Reduce batch size
→ Clear GPU cache: torch.cuda.empty_cache()
→ Use CPU fallback: device='cpu'
```

### Pattern 4: Encoding Errors
```
UnicodeEncodeError: 'charmap' codec
→ Set environment: os.environ['PYTHONIOENCODING'] = 'utf-8'
→ Use UTF-8 encoding in file operations
```

---

## 🆘 GETTING HELP

### Before Asking for Help:
1. Run `validate_complete_pipeline.py`
2. Run `quick_mediapipe_test.py` 
3. Check this troubleshooting guide
4. Enable debug logging
5. Note exact error messages

### Information to Include:
- Python version and OS
- Complete error message and stack trace
- Video specifications (resolution, length, format)
- Hardware (CPU/GPU, RAM)
- Which script was running when error occurred

**Remember:** Most issues are environment-related. When in doubt, try the conda environment setup from scratch.
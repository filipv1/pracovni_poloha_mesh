# 🐍 Python Version Compatibility Guide
# MediaPipe and SMPL-X Requirements

## 🎯 RECOMMENDED SETUP FOR RUNPOD

### Primary Choice: **Python 3.9 with Conda**

**Why Python 3.9:**
- ✅ **MediaPipe fully supported** - stable and tested
- ✅ **SMPL-X compatibility** - all features work
- ✅ **Open3D stable** - no DLL issues
- ✅ **PyTorch CUDA support** - excellent GPU acceleration
- ✅ **All dependencies available** - no compatibility conflicts

### Setup Command:
```bash
python setup_runpod_conda.py  # Creates Python 3.9 environment
```

---

## 📊 PYTHON VERSION COMPATIBILITY MATRIX

| Python Version | MediaPipe | SMPL-X | Open3D | PyTorch | Status |
|---------------|-----------|---------|---------|---------|---------|
| **3.9** | ✅ Excellent | ✅ Perfect | ✅ Stable | ✅ Full CUDA | **RECOMMENDED** |
| 3.10 | ✅ Good | ✅ Good | ⚠️ Some issues | ✅ Full CUDA | OK |
| 3.11 | ⚠️ Limited | ✅ Good | ⚠️ Some issues | ✅ Full CUDA | Risky |
| 3.12 | ❌ Unstable | ⚠️ Limited | ❌ Issues | ⚠️ Limited | Not Recommended |
| 3.13 | ❌ Not supported | ❌ Issues | ❌ Major issues | ❌ No support | Avoid |

---

## 🔧 RUNPOD TEMPLATE CONSIDERATIONS

### RTX 4090 Templates Analysis:

**✅ RECOMMENDED: "PyTorch 2.0" Template**
- Base: Ubuntu 22.04
- Python: Usually 3.10 (we override to 3.9)
- CUDA: 11.8 pre-installed
- Conda: Available
- **Action**: Run `setup_runpod_conda.py` to create Python 3.9 env

**✅ GOOD: "CUDA Development" Template**  
- Base: Ubuntu 22.04
- Python: Variable (we install 3.9)
- CUDA: 11.8+ 
- More control over setup
- **Action**: Run `setup_runpod_conda.py` for full setup

**❌ AVOID: "Python 3.12" Templates**
- MediaPipe compatibility issues
- SMPL-X installation problems
- Open3D DLL conflicts

---

## 🚀 DEPLOYMENT STRATEGY

### Method 1: Conda Environment (RECOMMENDED)

```bash
# 1. Clone repository
git clone https://github.com/filipv1/pracovni_poloha_mesh.git
cd pracovni_poloha_mesh

# 2. Run conda setup (creates Python 3.9 environment)
python setup_runpod_conda.py

# 3. Activate optimized environment
./activate_mesh_env.sh

# 4. Test
mesh-test
```

**Advantages:**
- ✅ Guaranteed Python 3.9
- ✅ Isolated from system conflicts  
- ✅ MediaPipe fully compatible
- ✅ Easy activation/deactivation
- ✅ Reproducible across different templates

### Method 2: System Python (Fallback)

```bash
# Only if conda fails - check Python version first!
python --version  # Must be 3.9 or 3.10

# If wrong version, STOP and use conda method
python setup_runpod.py  # Risky if wrong Python version
```

---

## ⚠️ COMMON ISSUES & SOLUTIONS

### Issue 1: MediaPipe Installation Fails
```bash
# Cause: Wrong Python version (usually 3.12+)
# Solution: Use conda method
python setup_runpod_conda.py
```

### Issue 2: Open3D DLL Errors
```bash
# Cause: Python 3.11+ or missing libraries
# Solution: Conda with Python 3.9
conda activate mesh_pipeline
python -c "import open3d; print('OK')"
```

### Issue 3: SMPL-X Import Errors
```bash
# Cause: Missing dependencies or wrong Python version
# Solution: Full conda setup
./activate_mesh_env.sh
python -c "import smplx; print('OK')"
```

### Issue 4: CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch in conda environment
conda activate mesh_pipeline
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## 🎯 ENVIRONMENT VALIDATION

### Quick Test Script:
```python
#!/usr/bin/env python3
import sys
print(f"Python: {sys.version}")

# Critical imports test
try:
    import mediapipe as mp
    print("✅ MediaPipe: OK")
except ImportError as e:
    print(f"❌ MediaPipe: {e}")

try:
    import smplx
    print("✅ SMPL-X: OK") 
except ImportError as e:
    print(f"❌ SMPL-X: {e}")

try:
    import open3d as o3d
    print(f"✅ Open3D: OK (v{o3d.__version__})")
except ImportError as e:
    print(f"❌ Open3D: {e}")

try:
    import torch
    print(f"✅ PyTorch: OK (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
```

### Success Criteria:
- ✅ Python 3.9.x
- ✅ All imports successful
- ✅ CUDA available: True
- ✅ No error messages

---

## 📋 RUNPOD SETUP CHECKLIST

- [ ] **Template**: PyTorch 2.0 or CUDA Development
- [ ] **GPU**: RTX 4090/3090 (24GB+ VRAM)
- [ ] **Storage**: 50GB+ for models and outputs
- [ ] **Setup Method**: `python setup_runpod_conda.py`
- [ ] **Environment**: Python 3.9 conda environment  
- [ ] **Activation**: `./activate_mesh_env.sh`
- [ ] **Validation**: All critical imports working
- [ ] **SMPL-X Models**: Downloaded and placed
- [ ] **GPU Test**: `mesh-test` successful

**With Python 3.9 conda environment, you get maximum compatibility and stability for production 3D mesh processing! 🚀**
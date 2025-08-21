#!/usr/bin/env python3
"""
Setup script for 3D Human Mesh dependencies
Installs EasyMoCap, PyTorch3D, SMPL-X and related libraries for accurate human mesh fitting
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
import json
from pathlib import Path

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"\n{'='*50}")
    print(f"Installing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description}")
        if result.stdout:
            print("STDOUT:", result.stdout[:500])
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing {description}: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[:500])
        if e.stderr:
            print("STDERR:", e.stderr[:500])
        return False

def check_gpu_support():
    """Check if CUDA is available for GPU acceleration"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch not found")
        return False

def install_pytorch3d():
    """Install PyTorch3D for 3D mesh processing"""
    print("Installing PyTorch3D...")
    
    # Try conda install first (recommended)
    conda_cmd = "conda install pytorch3d -c pytorch3d -y"
    if run_command(conda_cmd, "PyTorch3D via conda"):
        return True
    
    # Fallback to pip
    pip_cmd = "pip install pytorch3d"
    if run_command(pip_cmd, "PyTorch3D via pip"):
        return True
    
    # Try pre-compiled wheels
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "cpu"
        
        wheel_cmd = f"pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt1130/download.html"
        if run_command(wheel_cmd, "PyTorch3D pre-compiled wheels"):
            return True
    except:
        pass
    
    print("WARNING: PyTorch3D installation failed. Will use fallback rendering.")
    return False

def install_smplx():
    """Install SMPL-X body model"""
    print("Installing SMPL-X...")
    
    # Install smplx library
    if not run_command("pip install smplx[all]", "SMPL-X library"):
        return False
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nSMPL-X library installed successfully!")
    print("IMPORTANT: You need to manually download SMPL-X model files:")
    print("1. Visit: https://smpl-x.is.tue.mpg.de/")
    print("2. Register and download SMPL-X models")
    print("3. Place files in: ./models/smplx/")
    print("   - SMPLX_NEUTRAL.npz")
    print("   - SMPLX_MALE.npz") 
    print("   - SMPLX_FEMALE.npz")
    
    return True

def install_easymocap():
    """Install EasyMoCap for high-accuracy SMPL fitting"""
    print("Installing EasyMoCap...")
    
    # Install from git
    git_cmd = "pip install git+https://github.com/zju3dv/EasyMocap.git"
    if run_command(git_cmd, "EasyMoCap from git"):
        return True
    
    # Alternative: clone and install locally
    print("Trying local installation...")
    if run_command("git clone https://github.com/zju3dv/EasyMocap.git", "Clone EasyMoCap"):
        if run_command("cd EasyMocap && pip install -e .", "Install EasyMoCap locally"):
            return True
    
    print("WARNING: EasyMoCap installation failed. Will use fallback SMPL fitting.")
    return False

def install_additional_deps():
    """Install additional dependencies for 3D mesh processing"""
    deps = [
        ("trimesh[easy]", "Trimesh for mesh operations"),
        ("open3d>=0.16.0", "Open3D for 3D processing"),
        ("scipy", "SciPy for optimization"),
        ("scikit-image", "Image processing utilities"),
        ("imageio[ffmpeg]", "Video I/O with FFmpeg"),
        ("chumpy", "Chumpy for SMPL operations"),
        ("chamferdist", "Chamfer distance for mesh evaluation")
    ]
    
    success_count = 0
    for package, description in deps:
        if run_command(f"pip install {package}", description):
            success_count += 1
    
    print(f"\nInstalled {success_count}/{len(deps)} additional dependencies")
    return success_count > len(deps) // 2  # Success if > 50% installed

def create_test_script():
    """Create a test script to validate installation"""
    test_code = '''
import sys
import importlib

def test_import(module_name, description):
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}: OK")
        return True
    except ImportError as e:
        print(f"✗ {description}: FAILED - {e}")
        return False

def main():
    print("Testing 3D Mesh Dependencies Installation")
    print("="*50)
    
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("trimesh", "Trimesh"),
        ("smplx", "SMPL-X"),
        ("pytorch3d", "PyTorch3D"),
        ("open3d", "Open3D"),
        ("easymocap", "EasyMoCap")
    ]
    
    success_count = 0
    for module, description in tests:
        if test_import(module, description):
            success_count += 1
    
    print(f"\\nInstallation Test Results: {success_count}/{len(tests)} modules OK")
    
    if success_count >= len(tests) - 2:  # Allow 2 failures
        print("✓ Installation appears successful!")
        return True
    else:
        print("✗ Installation has significant issues. Check error messages above.")
        return False

if __name__ == "__main__":
    main()
'''
    
    with open("test_mesh_installation.py", "w", encoding='utf-8') as f:
        f.write(test_code)
    
    print("Created test_mesh_installation.py - run this to validate installation")

def main():
    print("3D Human Mesh Dependencies Setup")
    print("=" * 50)
    
    # Check current environment
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check GPU support
    gpu_available = check_gpu_support()
    
    # Install dependencies
    installations = []
    installations.append(("PyTorch3D", install_pytorch3d()))
    installations.append(("SMPL-X", install_smplx()))
    installations.append(("EasyMoCap", install_easymocap()))
    installations.append(("Additional Dependencies", install_additional_deps()))
    
    # Create test script
    create_test_script()
    
    # Summary
    print("\\n" + "=" * 50)
    print("INSTALLATION SUMMARY")
    print("=" * 50)
    
    successful = 0
    for name, success in installations:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name:25} {status}")
        if success:
            successful += 1
    
    print(f"\\nOverall: {successful}/{len(installations)} components installed successfully")
    
    if gpu_available:
        print("\\n🚀 GPU acceleration available for RunPod deployment!")
    else:
        print("\\n⚠️  CPU-only mode. Performance will be limited on Intel GPU.")
    
    print("\\nNext steps:")
    print("1. Run: python test_mesh_installation.py")
    print("2. Download SMPL-X model files (see instructions above)")
    print("3. Test with short video segment")
    
    return successful >= len(installations) - 1

if __name__ == "__main__":
    main()
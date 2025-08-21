#!/usr/bin/env python3
"""
Install 3D mesh dependencies in conda trunk_analysis environment
"""
import subprocess
import sys

def run_in_conda(cmd, description):
    """Run command in conda environment"""
    full_cmd = f"C:/Users/vaclavik/miniconda3/envs/trunk_analysis/Scripts/pip.exe {cmd}"
    print(f"\n{'='*50}")
    print(f"Installing: {description}")
    print(f"Command: {full_cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(full_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description}")
        if result.stdout:
            print("STDOUT:", result.stdout[:1000])
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing {description}: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[:1000])
        if e.stderr:
            print("STDERR:", e.stderr[:1000])
        return False

def main():
    print("Installing dependencies in trunk_analysis conda environment")
    
    # Check current environment
    result = subprocess.run("C:/Users/vaclavik/miniconda3/envs/trunk_analysis/python.exe --version", 
                           shell=True, capture_output=True, text=True)
    print(f"Python version: {result.stdout.strip()}")
    
    # Install packages
    packages = [
        ("install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt251/download.html", "PyTorch3D for CUDA 12.1"),
        ("install smplx[all]", "SMPL-X body models"),
        ("install 'git+https://github.com/zju3dv/EasyMocap.git'", "EasyMoCap framework"),
        ("install trimesh[easy]", "Trimesh mesh processing"),
        ("install open3d>=0.16.0", "Open3D 3D processing"),
        ("install scipy scikit-image imageio[ffmpeg]", "Additional processing libraries"),
        ("install chumpy", "SMPL utilities")
    ]
    
    success_count = 0
    for cmd, desc in packages:
        if run_in_conda(cmd, desc):
            success_count += 1
    
    print(f"\n\nInstalled {success_count}/{len(packages)} packages successfully")
    
    # Test installation
    test_cmd = "C:/Users/vaclavik/miniconda3/envs/trunk_analysis/python.exe -c \"import torch; print('CUDA available:', torch.cuda.is_available()); import smplx; print('SMPL-X OK');\""
    print(f"\nTesting installation:")
    subprocess.run(test_cmd, shell=True)

if __name__ == "__main__":
    main()
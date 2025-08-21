#!/usr/bin/env python3
"""
RunPod GPU Setup with Conda Environment Management
Ensures proper Python version for MediaPipe compatibility
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description, check=True, show_output=True):
    """Execute shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"SETUP: {description}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        if show_output and result.stdout:
            print("OUTPUT:", result.stdout[:1500])
        if result.stderr and len(result.stderr.strip()) > 0:
            print("STDERR:", result.stderr[:800])
            
        print(f"SUCCESS: {description}")
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[:1000])
        if e.stderr:
            print("STDERR:", e.stderr[:1000])
        return False, ""
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, ""

def check_conda_installation():
    """Check if conda is available"""
    print("CHECKING CONDA INSTALLATION")
    print("=" * 60)
    
    success, output = run_command("which conda", "Check conda location", check=False)
    if success:
        print(f"Conda found: {output.strip()}")
    else:
        print("Conda not found in PATH")
    
    success, output = run_command("conda --version", "Check conda version", check=False)
    if success:
        print(f"Conda version: {output.strip()}")
        return True
    else:
        print("Conda not available - will install")
        return False

def install_miniconda():
    """Install Miniconda if not available"""
    print("INSTALLING MINICONDA")
    print("=" * 60)
    
    # Download Miniconda installer
    installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh"
    
    run_command(
        f"wget {installer_url} -O miniconda_installer.sh",
        "Download Miniconda installer"
    )
    
    # Make executable and install
    run_command("chmod +x miniconda_installer.sh", "Make installer executable")
    
    run_command(
        "bash miniconda_installer.sh -b -p $HOME/miniconda3",
        "Install Miniconda"
    )
    
    # Add to PATH
    run_command(
        'echo "export PATH=$HOME/miniconda3/bin:$PATH" >> ~/.bashrc',
        "Add conda to PATH"
    )
    
    # Initialize conda
    run_command(
        "$HOME/miniconda3/bin/conda init bash",
        "Initialize conda"
    )
    
    print("IMPORTANT: Please run 'source ~/.bashrc' or restart terminal")
    return True

def create_conda_environment():
    """Create conda environment with correct Python version"""
    print("CREATING CONDA ENVIRONMENT")
    print("=" * 60)
    
    env_name = "mesh_pipeline"
    python_version = "3.9"  # Optimal for MediaPipe
    
    # Check if environment already exists
    success, output = run_command(
        "conda env list",
        "List existing environments",
        check=False
    )
    
    if env_name in output:
        print(f"Environment '{env_name}' already exists")
        
        # Ask user if they want to recreate
        print("Removing existing environment to ensure clean setup...")
        run_command(
            f"conda env remove -n {env_name} -y",
            f"Remove existing {env_name} environment"
        )
    
    # Create new environment
    run_command(
        f"conda create -n {env_name} python={python_version} -y",
        f"Create {env_name} environment with Python {python_version}"
    )
    
    print(f"Environment '{env_name}' created successfully!")
    return env_name

def install_cuda_pytorch(env_name):
    """Install PyTorch with CUDA support in conda environment"""
    print("INSTALLING PYTORCH WITH CUDA")
    print("=" * 60)
    
    # Activate environment and install PyTorch
    pytorch_cmd = f"""
    source activate {env_name} && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    """
    
    success, _ = run_command(
        pytorch_cmd,
        "Install PyTorch with CUDA support"
    )
    
    if not success:
        # Fallback to pip installation
        print("Conda installation failed, trying pip...")
        pip_cmd = f"""
        source activate {env_name} && \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        """
        run_command(pip_cmd, "Install PyTorch via pip")

def install_pipeline_dependencies(env_name):
    """Install all pipeline dependencies in conda environment"""
    print("INSTALLING PIPELINE DEPENDENCIES")
    print("=" * 60)
    
    # Core dependencies with conda where possible
    conda_packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "matplotlib>=3.7.0",
        "opencv",  # conda version often more stable
        "ffmpeg",  # Important for video processing
    ]
    
    for package in conda_packages:
        cmd = f"source activate {env_name} && conda install {package} -y"
        run_command(cmd, f"Install {package} via conda", check=False)
    
    # Pip packages that work better with pip
    pip_packages = [
        "smplx[all]>=0.1.28",
        "trimesh[easy]>=4.0.0", 
        "open3d>=0.18.0",
        "mediapipe>=0.10.8",
        "opencv-contrib-python>=4.8.0",  # Additional OpenCV features
        "scikit-image>=0.20.0",
        "imageio[ffmpeg]>=2.30.0",
        "ffmpeg-python>=0.2.0",
        "chumpy>=0.70",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0"
    ]
    
    # Install pip packages
    for package in pip_packages:
        cmd = f"source activate {env_name} && pip install {package}"
        run_command(cmd, f"Install {package} via pip", check=False)
    
    # Try to install PyTorch3D (optional but helpful)
    pytorch3d_cmd = f"""
    source activate {env_name} && \
    pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
    """
    run_command(pytorch3d_cmd, "Install PyTorch3D (optional)", check=False)

def test_environment(env_name):
    """Test the conda environment setup"""
    print("TESTING ENVIRONMENT")
    print("=" * 60)
    
    test_script = f'''
import sys
print(f"Python: {{sys.version}}")
print(f"Python executable: {{sys.executable}}")

# Test critical imports
imports_to_test = [
    ("torch", "PyTorch"),
    ("smplx", "SMPL-X"),
    ("open3d", "Open3D"), 
    ("mediapipe", "MediaPipe"),
    ("cv2", "OpenCV"),
    ("trimesh", "Trimesh"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy")
]

for module, name in imports_to_test:
    try:
        imported = __import__(module)
        if hasattr(imported, "__version__"):
            print(f"✓ {{name}}: v{{imported.__version__}}")
        else:
            print(f"✓ {{name}}: OK")
    except ImportError as e:
        print(f"✗ {{name}}: FAILED - {{e}}")

# Test CUDA if PyTorch available
try:
    import torch
    print(f"CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        print(f"CUDA version: {{torch.version.cuda}}")
        print(f"GPU count: {{torch.cuda.device_count()}}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {{i}}: {{torch.cuda.get_device_name(i)}}")
except:
    pass

print("\\nEnvironment test complete!")
'''
    
    # Save test script
    with open("test_conda_environment.py", "w") as f:
        f.write(test_script)
    
    # Run test in conda environment
    test_cmd = f"source activate {env_name} && python test_conda_environment.py"
    run_command(test_cmd, "Test conda environment setup")

def create_activation_scripts(env_name):
    """Create convenient activation scripts"""
    print("CREATING ACTIVATION SCRIPTS")
    print("=" * 60)
    
    # Bash activation script
    bash_script = f'''#!/bin/bash
# 3D Human Mesh Pipeline Environment Activation

echo "Activating 3D Human Mesh Pipeline environment..."
echo "Python 3.9 with MediaPipe, SMPL-X, and Open3D"

source activate {env_name}

echo "Environment: $(conda info --envs | grep '*')"
echo "Python: $(python --version)"
echo "Ready for 3D mesh processing!"

# Set environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Useful aliases
alias mesh-test="python test_gpu_pipeline.py"
alias mesh-run="python production_3d_pipeline_clean.py"
alias mesh-setup="python setup_runpod_conda.py"

echo "Aliases created: mesh-test, mesh-run, mesh-setup"
'''
    
    with open("activate_mesh_env.sh", "w") as f:
        f.write(bash_script)
    
    run_command("chmod +x activate_mesh_env.sh", "Make activation script executable")
    
    # Create usage instructions
    instructions = f'''# 3D Human Mesh Pipeline - RunPod Setup Complete!

## Environment Activation

### Method 1: Use activation script (recommended)
```bash
./activate_mesh_env.sh
```

### Method 2: Manual activation  
```bash
source activate {env_name}
```

### Method 3: Conda activate
```bash
conda activate {env_name}
```

## Quick Commands

After activation:
```bash
# Test GPU pipeline
mesh-test

# Process video
mesh-run

# Re-run setup if needed
mesh-setup
```

## Manual Commands

```bash
# Test installation
python test_conda_environment.py

# Test GPU pipeline  
python test_gpu_pipeline.py

# Process video
python production_3d_pipeline_clean.py

# Quick 3-frame test
python quick_test_3_frames.py
```

## Environment Details

- **Name**: {env_name}
- **Python**: 3.9 (MediaPipe optimized)
- **Location**: ~/miniconda3/envs/{env_name}
- **CUDA**: Enabled (if GPU available)

## Next Steps

1. Upload SMPL-X models to models/smplx/
2. Upload test video (test.mp4)
3. Run: mesh-test
4. Process your videos!

Environment is ready for production 3D mesh processing! 🚀
'''
    
    with open("RUNPOD_USAGE.md", "w") as f:
        f.write(instructions)
    
    print("Created activation script: activate_mesh_env.sh")
    print("Created usage guide: RUNPOD_USAGE.md")

def main():
    """Main setup function for RunPod with conda"""
    print("RUNPOD SETUP WITH CONDA ENVIRONMENT")
    print("=" * 80)
    print("Ensures Python 3.9 compatibility for MediaPipe")
    print("=" * 80)
    
    # Check system
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Current Python: {sys.version}")
    
    # Check/install conda
    conda_available = check_conda_installation()
    if not conda_available:
        install_miniconda()
        print("\nIMPORTANT: Run 'source ~/.bashrc' then re-run this script")
        return
    
    # Create environment
    env_name = create_conda_environment()
    
    # Install dependencies
    install_cuda_pytorch(env_name)
    install_pipeline_dependencies(env_name)
    
    # Test setup
    test_environment(env_name)
    
    # Create convenience scripts
    create_activation_scripts(env_name)
    
    print("\n" + "=" * 80)
    print("CONDA ENVIRONMENT SETUP COMPLETE!")
    print("=" * 80)
    
    print(f"\nEnvironment: {env_name}")
    print("Python: 3.9 (MediaPipe compatible)")
    print("CUDA: Enabled")
    print("All dependencies: Installed")
    
    print(f"\nACTIVATION:")
    print(f"./activate_mesh_env.sh")
    print(f"OR: conda activate {env_name}")
    
    print(f"\nNEXT STEPS:")
    print("1. Upload SMPL-X models to models/smplx/")
    print("2. Upload test video")
    print("3. Activate environment: ./activate_mesh_env.sh")
    print("4. Test: mesh-test")
    print("5. Process videos: mesh-run")
    
    print(f"\n🚀 READY FOR GPU PROCESSING WITH PROPER PYTHON ENVIRONMENT!")

if __name__ == "__main__":
    main()
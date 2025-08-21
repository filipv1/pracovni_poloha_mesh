#!/usr/bin/env python3
"""
RunPod GPU Setup Script for 3D Human Mesh Pipeline
Automated installation and configuration for Ubuntu/Linux environment
"""

import subprocess
import sys
import os
import platform
import torch

def run_command(cmd, description, check=True):
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
        
        if result.stdout:
            print("OUTPUT:", result.stdout[:1000])
        if result.stderr and len(result.stderr.strip()) > 0:
            print("STDERR:", result.stderr[:500])
            
        print(f"SUCCESS: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[:1000])
        if e.stderr:
            print("STDERR:", e.stderr[:1000])
        return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def check_system():
    """Check system information and requirements"""
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check NVIDIA GPU
    gpu_available = run_command("nvidia-smi", "Check NVIDIA GPU", check=False)
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA: Available ({torch.version.cuda})")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA: Not available")
    except ImportError:
        print("PyTorch: Not installed yet")
        cuda_available = False
    
    return gpu_available and cuda_available

def install_system_dependencies():
    """Install system-level dependencies"""
    print("\nINSTALLING SYSTEM DEPENDENCIES")
    print("=" * 60)
    
    # Update package lists
    run_command("apt update", "Update package lists")
    
    # Essential tools
    run_command(
        "apt install -y git curl wget htop tree vim nano ffmpeg",
        "Install essential tools"
    )
    
    # Development libraries
    run_command(
        "apt install -y build-essential cmake pkg-config",
        "Install build tools"
    )
    
    # Graphics and multimedia
    run_command(
        "apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6",
        "Install graphics libraries"
    )

def install_python_dependencies():
    """Install Python packages optimized for GPU"""
    print("\nINSTALLING PYTHON DEPENDENCIES")
    print("=" * 60)
    
    # Upgrade pip
    run_command(
        "pip install --upgrade pip setuptools wheel",
        "Upgrade pip and tools"
    )
    
    # Install PyTorch with CUDA support
    pytorch_cmd = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    run_command(pytorch_cmd, "Install PyTorch with CUDA 11.8")
    
    # Install from requirements file
    if os.path.exists("requirements_runpod.txt"):
        run_command(
            "pip install -r requirements_runpod.txt",
            "Install requirements from file"
        )
    else:
        # Fallback manual installation
        packages = [
            "smplx[all]",
            "trimesh[easy]", 
            "open3d>=0.18.0",
            "mediapipe>=0.10.8",
            "opencv-python opencv-contrib-python",
            "numpy scipy scikit-image matplotlib",
            "imageio[ffmpeg] ffmpeg-python",
            "chumpy tqdm Pillow",
            "jupyter ipywidgets notebook"
        ]
        
        for package in packages:
            run_command(f"pip install {package}", f"Install {package}")
    
    # Try to install PyTorch3D (optional)
    pytorch3d_cmd = (
        "pip install pytorch3d -f "
        "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html"
    )
    run_command(pytorch3d_cmd, "Install PyTorch3D (optional)", check=False)

def test_installation():
    """Test critical components"""
    print("\nTESTING INSTALLATION")
    print("=" * 60)
    
    test_code = '''
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"PyTorch error: {e}")

try:
    import smplx
    print(f"SMPL-X: OK")
except ImportError as e:
    print(f"SMPL-X error: {e}")

try:
    import open3d as o3d
    print(f"Open3D: OK (v{o3d.__version__})")
except ImportError as e:
    print(f"Open3D error: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe: OK")
except ImportError as e:
    print(f"MediaPipe error: {e}")

try:
    import trimesh
    print(f"Trimesh: OK")
except ImportError as e:
    print(f"Trimesh error: {e}")

try:
    import cv2
    print(f"OpenCV: OK (v{cv2.__version__})")
except ImportError as e:
    print(f"OpenCV error: {e}")

print("\\nInstallation test complete!")
'''
    
    # Save test script
    with open("test_installation.py", "w") as f:
        f.write(test_code)
    
    # Run test
    run_command("python test_installation.py", "Test installation")

def setup_models_directory():
    """Create and prepare models directory"""
    print("\nSETTING UP MODELS DIRECTORY")
    print("=" * 60)
    
    os.makedirs("models/smplx", exist_ok=True)
    
    print("Models directory created: models/smplx/")
    print("\nIMPORTANT: Upload SMPL-X model files manually:")
    print("  - SMPLX_NEUTRAL.npz")
    print("  - SMPLX_MALE.npz") 
    print("  - SMPLX_FEMALE.npz")
    print("\nUpload methods:")
    print("  1. SCP: scp -P [PORT] models/smplx/* root@[IP]:/workspace/project/models/smplx/")
    print("  2. Jupyter: Use file upload interface")
    print("  3. Wget: Download from your cloud storage")

def create_gpu_test_script():
    """Create GPU-specific test script"""
    test_script = '''#!/usr/bin/env python3
"""
GPU Test Script for RunPod 3D Human Mesh Pipeline
"""

import torch
import time
import os
from pathlib import Path

def test_gpu_pipeline():
    print("GPU PIPELINE TEST")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    
    # Test tensor operations
    device = torch.device('cuda')
    
    # Memory test
    start_time = time.time()
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"GPU Matrix Multiply: {gpu_time:.3f} seconds")
    
    # Test pipeline components
    try:
        from production_3d_pipeline_clean import MasterPipeline
        print("Pipeline import: OK")
        
        # Initialize with GPU
        pipeline = MasterPipeline(device='cuda')
        print("Pipeline GPU init: OK")
        
        return True
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False

def test_with_sample_video():
    """Test with sample video if available"""
    test_videos = ['test.mp4', 'sample.mp4', 'demo.mp4']
    
    for video in test_videos:
        if Path(video).exists():
            print(f"\\nTesting with {video}...")
            
            from production_3d_pipeline_clean import MasterPipeline
            pipeline = MasterPipeline(device='cuda')
            
            results = pipeline.execute_full_pipeline(
                video,
                output_dir="gpu_test_output",
                max_frames=6,
                frame_skip=2,
                quality='high'
            )
            
            if results:
                print(f"SUCCESS: {len(results['mesh_sequence'])} meshes generated")
                return True
            else:
                print(f"FAILED: No meshes generated")
    
    print("No test videos found")
    return False

if __name__ == "__main__":
    success = test_gpu_pipeline()
    
    if success:
        print("\\nAdvanced test with video...")
        test_with_sample_video()
    
    print("\\nGPU test complete!")
'''
    
    with open("test_gpu_pipeline.py", "w") as f:
        f.write(test_script)
    
    print("Created GPU test script: test_gpu_pipeline.py")

def main():
    """Main setup function"""
    print("RUNPOD GPU SETUP FOR 3D HUMAN MESH PIPELINE")
    print("=" * 80)
    
    # Check system
    system_ok = check_system()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies  
    install_python_dependencies()
    
    # Test installation
    test_installation()
    
    # Setup directories
    setup_models_directory()
    
    # Create test scripts
    create_gpu_test_script()
    
    print("\\n" + "=" * 80)
    print("RUNPOD SETUP COMPLETE!")
    print("=" * 80)
    
    print("\\nNEXT STEPS:")
    print("1. Upload SMPL-X models to models/smplx/")
    print("2. Upload test video (test.mp4)")
    print("3. Run: python test_gpu_pipeline.py")
    print("4. Start processing with production_3d_pipeline_clean.py")
    
    print("\\nTROUBLESHOOTING:")
    print("- If CUDA issues: nvidia-smi && pip install torch --upgrade")
    print("- If memory issues: reduce batch size in pipeline")
    print("- If model issues: check file permissions in models/smplx/")
    
    if system_ok:
        print("\\n🚀 SYSTEM READY FOR GPU PROCESSING!")
    else:
        print("\\n⚠️  MANUAL GPU SETUP MAY BE REQUIRED")

if __name__ == "__main__":
    main()
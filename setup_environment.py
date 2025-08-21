#!/usr/bin/env python3
"""
Complete Environment Setup Script for EasyMoCap + PyTorch3D + SMPL-X Pipeline
Automated installation and configuration for maximum accuracy 3D human mesh fitting

Usage:
    python setup_environment.py --mode [full|minimal|cpu-only]
    
Modes:
    full: Complete GPU setup with all dependencies
    minimal: Essential dependencies only
    cpu-only: CPU-only processing setup
"""

import os
import sys
import subprocess
import argparse
import platform
import json
import urllib.request
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self, mode='full', base_dir=None):
        self.mode = mode
        self.base_dir = Path(base_dir or os.getcwd())
        self.conda_env_name = 'trunk_analysis'
        self.python_version = '3.8'
        
        # Create directories
        self.models_dir = self.base_dir / 'models'
        self.configs_dir = self.base_dir / 'configs'
        self.scripts_dir = self.base_dir / 'scripts'
        
        for dir_path in [self.models_dir, self.configs_dir, self.scripts_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def check_system_requirements(self):
        """Check system requirements and CUDA availability"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 7:
            raise RuntimeError(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
        
        # Check CUDA availability (if GPU mode)
        cuda_available = False
        if self.mode != 'cpu-only':
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                cuda_available = result.returncode == 0
                if cuda_available:
                    logger.info("CUDA GPU detected")
                else:
                    logger.warning("No CUDA GPU detected, falling back to CPU mode")
                    self.mode = 'cpu-only'
            except FileNotFoundError:
                logger.warning("nvidia-smi not found, assuming no GPU")
                self.mode = 'cpu-only'
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"System memory: {memory_gb:.1f} GB")
            if memory_gb < 16:
                logger.warning("Less than 16GB RAM detected - may cause memory issues")
        except ImportError:
            logger.info("Cannot check memory - install psutil for memory monitoring")
        
        return {
            'python_version': f"{python_version.major}.{python_version.minor}",
            'cuda_available': cuda_available,
            'platform': platform.system(),
            'mode': self.mode
        }
    
    def setup_conda_environment(self):
        """Create and setup conda environment"""
        logger.info(f"Setting up conda environment: {self.conda_env_name}")
        
        # Check if conda is available
        try:
            subprocess.run(['conda', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Conda not found. Please install Miniconda or Anaconda first.")
        
        # Create environment
        conda_commands = [
            f"conda create -n {self.conda_env_name} python={self.python_version} -y",
            f"conda activate {self.conda_env_name}",
        ]
        
        # Add package installations based on mode
        if self.mode == 'full':
            conda_commands.extend([
                "conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y",
                "conda install -c conda-forge -c fvcore -c iopath pytorch3d -y",
                "conda install numpy scipy matplotlib opencv -y",
            ])
        elif self.mode == 'minimal':
            conda_commands.extend([
                "conda install pytorch torchvision cpuonly -c pytorch -y",
                "conda install numpy scipy matplotlib opencv -y",
            ])
        else:  # cpu-only
            conda_commands.extend([
                "conda install pytorch torchvision cpuonly -c pytorch -y",
                "conda install numpy scipy matplotlib opencv -y",
            ])
        
        # Execute conda commands
        for cmd in conda_commands:
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed: {cmd}")
                raise e
    
    def install_python_packages(self):
        """Install Python packages via pip"""
        logger.info("Installing Python packages...")
        
        # Determine pip packages based on mode
        base_packages = [
            "mediapipe==0.8.11",
            "open3d>=0.16.0",
            "trimesh>=3.15.0",
            "smplx>=0.1.28",
            "chumpy>=0.70",
            "scipy",
            "scikit-learn",
            "tqdm",
            "pillow",
            "psutil",
        ]
        
        gpu_packages = [
            "pytorch3d",  # Will try conda first, fallback to pip
        ]
        
        # Install packages
        pip_cmd = f"conda run -n {self.conda_env_name} pip install"
        
        # Install base packages
        for package in base_packages:
            try:
                subprocess.run(f"{pip_cmd} {package}", shell=True, check=True)
                logger.info(f"Installed: {package}")
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install: {package}")
        
        # Install GPU packages if needed
        if self.mode == 'full':
            for package in gpu_packages:
                try:
                    subprocess.run(f"{pip_cmd} {package}", shell=True, check=True)
                    logger.info(f"Installed: {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to install {package} via pip")
    
    def setup_easymocap(self):
        """Clone and install EasyMoCap"""
        logger.info("Setting up EasyMoCap...")
        
        easymocap_dir = self.base_dir / 'EasyMocap'
        
        if not easymocap_dir.exists():
            # Clone repository
            git_cmd = f"git clone https://github.com/zju3dv/EasyMocap.git {easymocap_dir}"
            subprocess.run(git_cmd, shell=True, check=True)
        
        # Install EasyMoCap
        pip_cmd = f"conda run -n {self.conda_env_name} pip install -e {easymocap_dir}"
        try:
            subprocess.run(pip_cmd, shell=True, check=True)
            logger.info("EasyMoCap installed successfully")
        except subprocess.CalledProcessError:
            logger.warning("EasyMoCap installation failed - will provide fallback implementation")
    
    def download_model_files(self):
        """Download and setup model files"""
        logger.info("Setting up model files...")
        
        # Create model directory structure
        smplx_dir = self.models_dir / 'smplx'
        flame_dir = self.models_dir / 'flame'
        mano_dir = self.models_dir / 'mano'
        
        for dir_path in [smplx_dir, flame_dir, mano_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Create download instructions
        download_instructions = {
            'smplx': {
                'url': 'https://smpl-x.is.tue.mpg.de/',
                'files': [
                    'SMPLX_NEUTRAL.pkl',
                    'SMPLX_MALE.pkl', 
                    'SMPLX_FEMALE.pkl'
                ],
                'directory': str(smplx_dir)
            },
            'flame': {
                'url': 'https://flame.is.tue.mpg.de/',
                'files': [
                    'FLAME_NEUTRAL.pkl',
                    'flame_static_embedding.pkl'
                ],
                'directory': str(flame_dir)
            },
            'mano': {
                'url': 'https://mano.is.tue.mpg.de/',
                'files': [
                    'MANO_LEFT.pkl',
                    'MANO_RIGHT.pkl'
                ],
                'directory': str(mano_dir)
            }
        }
        
        # Save download instructions
        instructions_file = self.models_dir / 'download_instructions.json'
        with open(instructions_file, 'w') as f:
            json.dump(download_instructions, f, indent=2)
        
        logger.info(f"Model download instructions saved to: {instructions_file}")
        logger.info("Please download model files manually (requires registration)")
        
        return instructions_file
    
    def create_configuration_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        # EasyMoCap configuration
        easymocap_config = {
            'model': {
                'body_model': 'smplx',
                'gender': 'neutral',
                'model_path': str(self.models_dir / 'smplx'),
                'use_face_keypoints': False,
                'use_hand_keypoints': False,
            },
            'optimize': {
                'stages': [
                    {
                        'iterations': 100,
                        'optimize': ['shapes', 'poses', 'Rh', 'Th'],
                        'weights': {
                            'keypoints2d': 1.0,
                            'pose_reg': 0.1,
                            'shape_reg': 0.01,
                            'smooth_pose': 0.1,
                            'smooth_shape': 0.1,
                        }
                    }
                ]
            },
            'dataset': {
                'ranges': [0, -1],
                'step': 1,
            }
        }
        
        config_file = self.configs_dir / 'easymocap_config.yml'
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(easymocap_config, f, default_flow_style=False)
        
        # Pipeline configuration
        pipeline_config = {
            'processing': {
                'batch_size': 16 if self.mode == 'full' else 4,
                'render_resolution': 1024 if self.mode == 'full' else 512,
                'temporal_smoothing': True,
                'use_gpu': self.mode != 'cpu-only',
            },
            'mediapipe': {
                'confidence': 0.7,
                'tracking_confidence': 0.5,
                'model_complexity': 2,
            },
            'output': {
                'save_intermediate': True,
                'export_meshes': True,
                'render_video': True,
            }
        }
        
        pipeline_config_file = self.configs_dir / 'pipeline_config.json'
        with open(pipeline_config_file, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
        
        logger.info(f"Configuration files created in: {self.configs_dir}")
        
        return config_file, pipeline_config_file
    
    def create_test_script(self):
        """Create test script to verify installation"""
        test_script = '''#!/usr/bin/env python3
"""
Test script to verify installation of EasyMoCap + PyTorch3D + SMPL-X pipeline
"""

import sys
import torch
import numpy as np

def test_basic_imports():
    """Test basic library imports"""
    print("Testing basic imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV import failed")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError:
        print("✗ MediaPipe import failed")
        return False
    
    try:
        import trimesh
        print("✓ Trimesh imported successfully")
    except ImportError:
        print("✗ Trimesh import failed")
        return False
    
    try:
        import open3d as o3d
        print("✓ Open3D imported successfully")
    except ImportError:
        print("✗ Open3D import failed")
        return False
    
    return True

def test_pytorch_setup():
    """Test PyTorch and CUDA setup"""
    print("\\nTesting PyTorch setup...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(3, 3)
        if torch.cuda.is_available():
            x = x.cuda()
        y = torch.mm(x, x.t())
        print("✓ Basic tensor operations work")
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def test_pytorch3d():
    """Test PyTorch3D installation"""
    print("\\nTesting PyTorch3D...")
    
    try:
        import pytorch3d
        print(f"✓ PyTorch3D version: {pytorch3d.__version__}")
        
        # Test basic functionality
        from pytorch3d.renderer import FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras()
        print("✓ PyTorch3D basic functionality works")
        return True
        
    except ImportError:
        print("✗ PyTorch3D import failed")
        return False
    except Exception as e:
        print(f"✗ PyTorch3D test failed: {e}")
        return False

def test_smplx():
    """Test SMPL-X model loading"""
    print("\\nTesting SMPL-X...")
    
    try:
        import smplx
        print("✓ SMPL-X library imported")
        
        # Note: Actual model loading requires downloaded model files
        print("Note: Model file testing requires manual model download")
        return True
        
    except ImportError:
        print("✗ SMPL-X import failed")
        return False

def test_easymocap():
    """Test EasyMoCap installation"""
    print("\\nTesting EasyMoCap...")
    
    try:
        import easymocap
        print("✓ EasyMoCap imported successfully")
        return True
    except ImportError:
        print("✗ EasyMoCap import failed (fallback implementation available)")
        return False

def main():
    """Run all tests"""
    print("=== Installation Verification Tests ===\\n")
    
    tests = [
        test_basic_imports,
        test_pytorch_setup,
        test_pytorch3d,
        test_smplx,
        test_easymocap,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! Installation successful.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        test_script_file = self.scripts_dir / 'test_installation.py'
        with open(test_script_file, 'w') as f:
            f.write(test_script)
        
        # Make executable
        os.chmod(test_script_file, 0o755)
        
        logger.info(f"Test script created: {test_script_file}")
        return test_script_file
    
    def create_usage_examples(self):
        """Create usage example scripts"""
        logger.info("Creating usage examples...")
        
        # Simple usage example
        simple_example = '''#!/usr/bin/env python3
"""
Simple usage example for the 3D human mesh fitting pipeline
"""

from pathlib import Path
import sys

# Add the implementation to Python path
sys.path.append(str(Path(__file__).parent.parent))

from implementation_architecture import CompletePipeline, PipelineConfig

def main():
    # Configure pipeline
    config = PipelineConfig(
        input_video_path="input_video.mp4",  # Replace with your video
        output_dir="output_results",
        smplx_model_path="models/smplx",
        render_resolution=1024,
        temporal_smoothing=True,
        use_gpu=True  # Set to False for CPU-only processing
    )
    
    # Create and run pipeline
    pipeline = CompletePipeline(config)
    
    try:
        results = pipeline.process_video(config.input_video_path)
        
        print("Processing completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
        
        example_file = self.scripts_dir / 'simple_example.py'
        with open(example_file, 'w') as f:
            f.write(simple_example)
        
        os.chmod(example_file, 0o755)
        
        logger.info(f"Usage example created: {example_file}")
        return example_file
    
    def generate_setup_report(self):
        """Generate setup completion report"""
        logger.info("Generating setup report...")
        
        report = {
            'setup_mode': self.mode,
            'conda_environment': self.conda_env_name,
            'python_version': self.python_version,
            'base_directory': str(self.base_dir),
            'directories_created': {
                'models': str(self.models_dir),
                'configs': str(self.configs_dir),
                'scripts': str(self.scripts_dir),
            },
            'next_steps': [
                f"Activate environment: conda activate {self.conda_env_name}",
                "Download model files (see models/download_instructions.json)",
                "Run test script: python scripts/test_installation.py",
                "Try example: python scripts/simple_example.py",
            ],
            'important_files': {
                'main_implementation': 'implementation_architecture.py',
                'test_script': 'scripts/test_installation.py',
                'configuration': 'configs/pipeline_config.json',
                'troubleshooting': 'troubleshooting_guide.md',
            }
        }
        
        report_file = self.base_dir / 'setup_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\\n" + "="*60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Mode: {self.mode}")
        print(f"Environment: {self.conda_env_name}")
        print(f"Base directory: {self.base_dir}")
        print("\\nNext steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print(f"\\nSetup report saved to: {report_file}")
        
        return report_file


def main():
    parser = argparse.ArgumentParser(description='Setup EasyMoCap + PyTorch3D + SMPL-X environment')
    parser.add_argument('--mode', choices=['full', 'minimal', 'cpu-only'], 
                       default='full', help='Installation mode')
    parser.add_argument('--base-dir', help='Base directory for installation')
    parser.add_argument('--skip-conda', action='store_true', 
                       help='Skip conda environment creation')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only run installation tests')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Just run the test script
        test_script = Path(__file__).parent / 'scripts' / 'test_installation.py'
        if test_script.exists():
            subprocess.run([sys.executable, str(test_script)])
        else:
            print("Test script not found. Run full setup first.")
        return
    
    try:
        # Create setup instance
        setup = EnvironmentSetup(mode=args.mode, base_dir=args.base_dir)
        
        # Check system requirements
        system_info = setup.check_system_requirements()
        logger.info(f"System check completed: {system_info}")
        
        # Setup conda environment
        if not args.skip_conda:
            setup.setup_conda_environment()
        
        # Install Python packages
        setup.install_python_packages()
        
        # Setup EasyMoCap
        setup.setup_easymocap()
        
        # Download model files
        setup.download_model_files()
        
        # Create configuration files
        setup.create_configuration_files()
        
        # Create test script
        setup.create_test_script()
        
        # Create usage examples
        setup.create_usage_examples()
        
        # Generate final report
        setup.generate_setup_report()
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
# Troubleshooting Guide and Alternative Solutions

## Common Issues and Solutions

### 1. Dependency Conflicts in Conda Environment

#### Issue: Version conflicts between PyTorch, PyTorch3D, and EasyMoCap
```bash
# Common error messages:
# "RuntimeError: CUDA version mismatch"
# "ImportError: cannot import name 'rasterize_meshes'"
# "ModuleNotFoundError: No module named 'pytorch3d._C'"
```

#### Solutions:
```bash
# Option A: Use conda-forge for consistent builds
conda create -n trunk_analysis python=3.8
conda activate trunk_analysis
conda install -c conda-forge pytorch torchvision pytorch-cuda=11.8
conda install -c conda-forge -c fvcore -c iopath pytorch3d

# Option B: Build PyTorch3D from source (if binary installation fails)
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

# Option C: Use Docker container (most reliable)
docker pull pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
# Mount your data directory and install additional dependencies
```

#### Alternative: Simplified Environment Setup
```python
# minimal_requirements.txt
torch==1.13.0+cu116
torchvision==0.14.0+cu116
numpy==1.21.6
opencv-python==4.6.0.66
mediapipe==0.8.11
trimesh==3.15.8
open3d==0.16.0
chumpy==0.70
smplx==0.1.28
```

### 2. GPU Compatibility Requirements

#### Issue: CUDA compatibility problems
```python
# Check CUDA compatibility
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU devices: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

#### Solutions:
1. **CUDA Version Mismatch**:
```bash
# Install matching CUDA toolkit version
conda install cudatoolkit=11.8
# Or use CPU-only version for testing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **Insufficient GPU Memory**:
```python
# Memory optimization strategies
def optimize_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # Reduces memory usage
        torch.backends.cudnn.enabled = False    # Fallback for compatibility
```

3. **Alternative: CPU-Only Processing**:
```python
class CPUFallbackConfig:
    """Configuration for CPU-only processing"""
    def __init__(self):
        self.device = 'cpu'
        self.batch_size = 1  # Smaller batches for CPU
        self.render_resolution = 512  # Lower resolution
        self.use_simplified_renderer = True
        
    def setup_cpu_renderer(self):
        # Use Open3D for CPU rendering instead of PyTorch3D
        import open3d as o3d
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.render_resolution, height=self.render_resolution, visible=False)
        return vis
```

### 3. EasyMoCap Installation Issues

#### Issue: EasyMoCap import errors or missing dependencies
```python
# Common errors:
# "ImportError: No module named 'easymocap'"
# "AttributeError: module 'easymocap' has no attribute 'bodymodels'"
```

#### Solutions:
1. **Manual Installation**:
```bash
# Clone and install from source
git clone https://github.com/zju3dv/EasyMocap.git
cd EasyMocap

# Install dependencies first
pip install torch torchvision
pip install opencv-python
pip install open3d
pip install chumpy
pip install smplx

# Install EasyMoCap
python setup.py develop
# OR
pip install -e .
```

2. **Alternative: Direct SMPL-X Implementation**:
```python
"""
Simplified SMPL-X fitting without EasyMoCap dependency
Uses direct optimization with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from smplx import SMPLX

class DirectSMPLXFitter:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.body_model = SMPLX(
            model_path=model_path,
            gender='neutral',
            use_face_contour=False,
            use_hands=False,
        ).to(device)
        
    def fit_sequence(self, keypoints_2d, camera_params, num_iterations=100):
        """Fit SMPL-X to 2D keypoints sequence"""
        batch_size = len(keypoints_2d)
        
        # Initialize parameters
        betas = nn.Parameter(torch.zeros(batch_size, 10, device=self.device))
        body_pose = nn.Parameter(torch.zeros(batch_size, 63, device=self.device))
        global_orient = nn.Parameter(torch.zeros(batch_size, 3, device=self.device))
        transl = nn.Parameter(torch.zeros(batch_size, 3, device=self.device))
        
        # Setup optimizer
        optimizer = optim.Adam([betas, body_pose, global_orient, transl], lr=0.01)
        
        # Convert keypoints to tensor
        target_keypoints = torch.tensor(keypoints_2d, device=self.device)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            body_output = self.body_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl
            )
            
            # Project 3D joints to 2D
            joints_3d = body_output.joints
            joints_2d = self.project_3d_to_2d(joints_3d, camera_params)
            
            # Compute losses
            keypoint_loss = nn.MSELoss()(joints_2d, target_keypoints[..., :2])
            pose_reg = torch.mean(body_pose ** 2)
            shape_reg = torch.mean(betas ** 2)
            
            total_loss = keypoint_loss + 0.1 * pose_reg + 0.01 * shape_reg
            
            total_loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Iteration {i}, Loss: {total_loss.item():.6f}")
        
        return {
            'betas': betas.detach(),
            'body_pose': body_pose.detach(),
            'global_orient': global_orient.detach(),
            'transl': transl.detach()
        }
    
    def project_3d_to_2d(self, points_3d, camera_params):
        """Simple perspective projection"""
        # Simplified projection - assumes camera at origin looking down -Z
        focal_length = camera_params.get('focal_length', 1000)
        
        # Basic perspective projection
        projected = points_3d[:, :, :2] / (points_3d[:, :, 2:3] + 1e-8) * focal_length
        
        return projected
```

### 4. Processing Time Optimization

#### Issue: Slow processing times for longer videos

#### Solutions:
1. **Frame Sampling Strategy**:
```python
class VideoProcessor:
    def __init__(self, fps_target=10):  # Process every Nth frame
        self.fps_target = fps_target
        
    def sample_frames(self, video_path, max_frames=300):
        """Intelligent frame sampling"""
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling rate
        sample_rate = max(1, int(original_fps / self.fps_target))
        selected_frames = list(range(0, min(total_frames, max_frames * sample_rate), sample_rate))
        
        return selected_frames[:max_frames]
        
    def interpolate_missing_frames(self, sparse_results):
        """Interpolate SMPL-X parameters for skipped frames"""
        # Use linear interpolation for smooth results
        from scipy.interpolate import interp1d
        
        frame_indices = [r['frame_idx'] for r in sparse_results]
        params = [r['smplx_params'] for r in sparse_results]
        
        # Create interpolation functions
        all_frames = range(max(frame_indices) + 1)
        interpolated = []
        
        for param_name in params[0].keys():
            param_values = [p[param_name] for p in params]
            param_tensor = torch.stack(param_values)
            
            # Interpolate each parameter dimension
            interpolated_param = []
            for dim in range(param_tensor.shape[-1]):
                f = interp1d(frame_indices, param_tensor[:, dim], 
                           kind='linear', fill_value='extrapolate')
                interpolated_param.append(f(all_frames))
            
            interpolated.append(torch.tensor(interpolated_param).T)
        
        return dict(zip(params[0].keys(), interpolated))
```

2. **Parallel Processing**:
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_frame_batch(args):
    """Process a batch of frames in parallel"""
    frames, processor_config = args
    # Process frames using MediaPipe
    results = []
    for frame in frames:
        result = process_single_frame(frame, processor_config)
        results.append(result)
    return results

def parallel_video_processing(video_path, num_workers=4):
    """Process video using multiple workers"""
    # Split video into chunks
    frame_chunks = split_video_into_chunks(video_path, num_workers)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in frame_chunks:
            future = executor.submit(process_frame_batch, chunk)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in futures:
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    return all_results
```

### 5. Alternative Pipeline Approaches

#### Option A: Lightweight Pipeline (for resource-constrained environments)
```python
class LightweightPipeline:
    """Simplified pipeline using minimal dependencies"""
    
    def __init__(self):
        self.use_open3d_rendering = True
        self.use_basic_smpl = True  # Fall back to SMPL instead of SMPL-X
        self.render_resolution = 512
    
    def process_video(self, video_path):
        # Use MediaPipe + basic SMPL + Open3D rendering
        pass
```

#### Option B: Cloud-Based Processing
```python
class CloudPipeline:
    """Use cloud services for heavy computation"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def upload_and_process(self, video_path):
        # Upload to cloud service (e.g., Google Cloud, AWS)
        # Use pre-configured GPU instances
        # Download results
        pass
```

#### Option C: Real-Time Pipeline
```python
class RealTimePipeline:
    """Optimized for real-time processing"""
    
    def __init__(self):
        self.frame_skip = 2  # Process every 2nd frame
        self.use_model_caching = True
        self.temporal_smoothing_window = 5
        
    def process_stream(self, video_stream):
        # Optimized for webcam/live video processing
        pass
```

## Performance Estimates and Hardware Requirements

### Minimum Hardware Requirements:
- **CPU**: Intel i7-8700K or AMD Ryzen 7 2700X
- **GPU**: RTX 3060 (8GB VRAM) or better
- **RAM**: 16GB system memory
- **Storage**: 50GB free space (for models and temporary files)

### Recommended Hardware:
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **GPU**: RTX 4090 (24GB VRAM) or A100
- **RAM**: 32GB system memory
- **Storage**: 100GB SSD storage

### Processing Time Estimates:

| Video Length | Hardware | Processing Time |
|--------------|----------|-----------------|
| 30 seconds   | RTX 3060 | 15-25 minutes  |
| 30 seconds   | RTX 4090 | 5-10 minutes   |
| 2 minutes    | RTX 3060 | 45-90 minutes  |
| 2 minutes    | RTX 4090 | 15-30 minutes  |

### Memory Usage Guidelines:

| Component | GPU Memory | System Memory |
|-----------|------------|---------------|
| MediaPipe | ~100MB     | ~500MB        |
| SMPL-X Model | ~50MB   | ~200MB        |
| PyTorch3D | 2-8GB      | ~1GB          |
| Video Buffer | ~500MB   | 2-4GB         |

## Fallback Strategies

If the full pipeline fails, try these alternatives in order:

1. **Reduce Quality Settings**:
   - Lower render resolution (1024 → 512 → 256)
   - Use CPU-only processing
   - Skip temporal smoothing

2. **Use Alternative Libraries**:
   - Replace PyTorch3D with Open3D
   - Use basic SMPL instead of SMPL-X
   - Replace EasyMoCap with direct optimization

3. **Process in Chunks**:
   - Split long videos into shorter segments
   - Process each segment separately
   - Merge results with manual alignment

4. **Cloud Processing**:
   - Use Google Colab Pro with GPU
   - Rent cloud GPU instances (RunPod, Lambda Labs)
   - Use pre-built Docker containers
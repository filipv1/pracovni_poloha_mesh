"""
Complete Implementation Architecture: EasyMoCap + PyTorch3D + SMPL-X Pipeline
Data Flow: MediaPipe → preprocessing → SMPL-X fitting → visualization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""
    # Input/Output paths
    input_video_path: str
    output_dir: str
    
    # Model paths
    smplx_model_path: str = "models/smplx"
    easymocap_config_path: str = "configs/smplx_config.yml"
    
    # Processing parameters
    image_size: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    batch_size: int = 16
    
    # Quality settings
    render_resolution: int = 1024
    temporal_smoothing: bool = True
    use_gpu: bool = True
    
    # MediaPipe settings
    mediapipe_confidence: float = 0.7
    mediapipe_tracking_confidence: float = 0.5


class MediaPipeProcessor:
    """Handles MediaPipe pose detection and keypoint extraction"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=config.mediapipe_confidence,
            min_tracking_confidence=config.mediapipe_tracking_confidence
        )
        
        # MediaPipe to COCO keypoint mapping for EasyMoCap
        self.mp_to_coco_mapping = {
            0: 0,   # nose
            11: 5,  # left shoulder
            12: 6,  # right shoulder
            13: 7,  # left elbow
            14: 8,  # right elbow
            15: 9,  # left wrist
            16: 10, # right wrist
            23: 11, # left hip
            24: 12, # right hip
            25: 13, # left knee
            26: 14, # right knee
            27: 15, # left ankle
            28: 16, # right ankle
        }
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process entire video and extract keypoints"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_sequence = []
        
        logger.info(f"Processing {frame_count} frames from {video_path}")
        
        for frame_idx in tqdm(range(frame_count), desc="Extracting keypoints"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
                keypoints_data = {
                    'frame_idx': frame_idx,
                    'keypoints': keypoints,
                    'bbox': self._compute_bbox(keypoints),
                    'confidence': self._compute_confidence(results.pose_landmarks),
                    'image_shape': frame.shape
                }
            else:
                # No detection - create empty keypoints
                keypoints_data = {
                    'frame_idx': frame_idx,
                    'keypoints': np.zeros((17, 3)),
                    'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                    'confidence': 0.0,
                    'image_shape': frame.shape
                }
            
            keypoints_sequence.append(keypoints_data)
        
        cap.release()
        return keypoints_sequence
    
    def _extract_keypoints(self, landmarks, image_shape) -> np.ndarray:
        """Convert MediaPipe landmarks to COCO format keypoints"""
        h, w = image_shape[:2]
        keypoints = np.zeros((17, 3))  # COCO format: 17 keypoints
        
        for mp_idx, coco_idx in self.mp_to_coco_mapping.items():
            if mp_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[mp_idx]
                keypoints[coco_idx] = [
                    landmark.x * w,
                    landmark.y * h,
                    landmark.visibility
                ]
        
        return keypoints
    
    def _compute_bbox(self, keypoints: np.ndarray) -> List[float]:
        """Compute bounding box from keypoints"""
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) == 0:
            return [0, 0, 100, 100]
        
        x_min, y_min = valid_kpts[:, :2].min(axis=0)
        x_max, y_max = valid_kpts[:, :2].max(axis=0)
        
        # Add padding
        padding = 0.3
        w, h = x_max - x_min, y_max - y_min
        x_min = max(0, x_min - w * padding)
        y_min = max(0, y_min - h * padding)
        w *= (1 + 2 * padding)
        h *= (1 + 2 * padding)
        
        return [float(x_min), float(y_min), float(w), float(h)]
    
    def _compute_confidence(self, landmarks) -> float:
        """Compute average confidence of detected landmarks"""
        confidences = [lm.visibility for lm in landmarks.landmark]
        return float(np.mean(confidences))


class EasyMoCapProcessor:
    """Handles EasyMoCap SMPL-X fitting"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Import EasyMoCap modules
        try:
            from easymocap.bodymodels import create_bodymodel
            from easymocap.estimator import MultipleStageOptimizer
            self.create_bodymodel = create_bodymodel
            self.MultipleStageOptimizer = MultipleStageOptimizer
        except ImportError as e:
            logger.error(f"EasyMoCap not found: {e}")
            raise ImportError("Please install EasyMoCap: pip install -e /path/to/EasyMocap")
    
    def setup_body_model(self) -> Any:
        """Initialize SMPL-X body model"""
        model_config = {
            'model_type': 'smplx',
            'model_path': self.config.smplx_model_path,
            'gender': 'neutral',
            'use_face_contour': False,
            'use_hands': False,  # Limited by MediaPipe input
            'device': str(self.device)
        }
        
        body_model = self.create_bodymodel(**model_config)
        return body_model
    
    def create_easymocap_input(self, keypoints_sequence: List[Dict]) -> Dict[str, Any]:
        """Convert MediaPipe keypoints to EasyMoCap input format"""
        easymocap_data = {
            'keypoints2d': [],
            'camera': self._setup_camera_params(),
            'meta': {
                'height': self.config.image_size[1],
                'width': self.config.image_size[0],
                'fps': self.config.fps
            }
        }
        
        for frame_data in keypoints_sequence:
            # Convert to EasyMoCap format
            kpts_2d = {
                'id': 0,  # Person ID
                'keypoints': frame_data['keypoints'].flatten(),  # Flatten to 51-dim
                'bbox': frame_data['bbox'],
                'area': frame_data['bbox'][2] * frame_data['bbox'][3],
                'iscrowd': 0,
                'category_id': 1
            }
            easymocap_data['keypoints2d'].append([kpts_2d])  # List of persons per frame
        
        return easymocap_data
    
    def _setup_camera_params(self) -> Dict[str, Any]:
        """Setup camera parameters for optimization"""
        # Assume reasonable defaults for camera intrinsics
        h, w = self.config.image_size
        focal_length = max(h, w) * 1.2  # Reasonable assumption
        
        camera_params = {
            'K': np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ]),
            'R': np.eye(3),
            't': np.zeros(3),
            'dist': np.zeros(5)
        }
        
        return camera_params
    
    def fit_smplx(self, easymocap_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform SMPL-X fitting using EasyMoCap"""
        # Setup body model
        body_model = self.setup_body_model()
        
        # Configure optimization stages
        optimization_config = {
            'stages': [
                {
                    'iterations': 100,
                    'optimize': ['shapes', 'poses', 'Rh', 'Th'],
                    'weights': {
                        'keypoints2d': 1.0,
                        'pose_reg': 0.1,
                        'shape_reg': 0.01,
                        'smooth_pose': 0.1 if self.config.temporal_smoothing else 0.0,
                        'smooth_shape': 0.1 if self.config.temporal_smoothing else 0.0,
                    }
                }
            ]
        }
        
        # Initialize optimizer
        optimizer = self.MultipleStageOptimizer(body_model, optimization_config)
        
        # Perform optimization
        logger.info("Starting SMPL-X parameter optimization...")
        results = optimizer.optimize(easymocap_data)
        
        # Extract optimized parameters
        smplx_params = {
            'betas': results['shapes'],      # Shape parameters
            'body_pose': results['poses'][:, 3:66],  # Body pose (21*3)
            'global_orient': results['poses'][:, :3],  # Global orientation
            'transl': results['Th'],         # Translation
            'left_hand_pose': torch.zeros(len(results['poses']), 45),  # Default hand pose
            'right_hand_pose': torch.zeros(len(results['poses']), 45), # Default hand pose
        }
        
        return smplx_params


class PyTorch3DRenderer:
    """High-quality mesh rendering using PyTorch3D"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Import PyTorch3D components
        try:
            from pytorch3d.renderer import (
                FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
                MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex
            )
            from pytorch3d.structures import Meshes
            
            self.FoVPerspectiveCameras = FoVPerspectiveCameras
            self.RasterizationSettings = RasterizationSettings
            self.MeshRenderer = MeshRenderer
            self.MeshRasterizer = MeshRasterizer
            self.SoftPhongShader = SoftPhongShader
            self.PointLights = PointLights
            self.TexturesVertex = TexturesVertex
            self.Meshes = Meshes
            
        except ImportError as e:
            logger.error(f"PyTorch3D not found: {e}")
            raise ImportError("Please install PyTorch3D")
        
        self._setup_renderer()
    
    def _setup_renderer(self):
        """Initialize PyTorch3D renderer"""
        # Rasterization settings
        self.raster_settings = self.RasterizationSettings(
            image_size=self.config.render_resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Setup lights
        self.lights = self.PointLights(
            device=self.device,
            location=[[0.0, 0.0, -3.0]],
            ambient_color=[[0.3, 0.3, 0.3]],
            diffuse_color=[[0.7, 0.7, 0.7]]
        )
    
    def render_sequence(self, smplx_output: Dict[str, torch.Tensor], 
                       camera_params: Dict[str, Any]) -> List[torch.Tensor]:
        """Render complete sequence of SMPL-X meshes"""
        from smplx import SMPLX
        
        # Load SMPL-X model
        body_model = SMPLX(
            model_path=self.config.smplx_model_path,
            gender='neutral',
            use_face_contour=False,
            use_hands=False,
            device=self.device
        )
        
        # Generate meshes from SMPL-X parameters
        with torch.no_grad():
            smplx_mesh_output = body_model(**smplx_output)
            vertices = smplx_mesh_output.vertices
            faces = body_model.faces_tensor
        
        # Setup camera
        cameras = self._setup_cameras(len(vertices), camera_params)
        
        # Render frames
        rendered_frames = []
        batch_size = self.config.batch_size
        
        logger.info(f"Rendering {len(vertices)} frames...")
        
        for i in tqdm(range(0, len(vertices), batch_size), desc="Rendering"):
            end_idx = min(i + batch_size, len(vertices))
            batch_vertices = vertices[i:end_idx]
            batch_cameras = cameras[i:end_idx]
            
            # Create textured meshes
            batch_meshes = self._create_textured_meshes(batch_vertices, faces)
            
            # Render batch
            batch_rendered = self._render_batch(batch_meshes, batch_cameras)
            rendered_frames.extend(batch_rendered)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return rendered_frames
    
    def _setup_cameras(self, num_frames: int, camera_params: Dict[str, Any]):
        """Setup cameras for rendering"""
        # Extract camera intrinsics
        K = camera_params.get('K', np.eye(3))
        focal_length = K[0, 0]
        principal_point = [[K[0, 2], K[1, 2]]]
        
        cameras = self.FoVPerspectiveCameras(
            device=self.device,
            focal_length=[[focal_length, focal_length]] * num_frames,
            principal_point=principal_point * num_frames,
        )
        
        return cameras
    
    def _create_textured_meshes(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Create textured meshes for rendering"""
        batch_size = len(vertices)
        
        # Create skin-like vertex colors
        vertex_colors = torch.ones_like(vertices) * torch.tensor(
            [0.8, 0.6, 0.5], device=self.device
        )
        
        # Create textures
        textures = self.TexturesVertex(verts_features=vertex_colors)
        
        # Create meshes
        meshes = self.Meshes(
            verts=vertices,
            faces=faces.unsqueeze(0).repeat(batch_size, 1, 1),
            textures=textures
        )
        
        return meshes
    
    @torch.no_grad()
    def _render_batch(self, meshes, cameras):
        """Render a batch of meshes"""
        # Create renderer
        renderer = self.MeshRenderer(
            rasterizer=self.MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=self.SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )
        )
        
        # Render
        rendered_images = renderer(meshes)
        
        return rendered_images


class CompletePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mediapipe_processor = MediaPipeProcessor(config)
        self.easymocap_processor = EasyMoCapProcessor(config)
        self.pytorch3d_renderer = PyTorch3DRenderer(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process complete video through the pipeline"""
        logger.info("Starting complete video processing pipeline...")
        
        try:
            # Stage 1: MediaPipe keypoint extraction
            logger.info("Stage 1: Extracting keypoints with MediaPipe...")
            keypoints_sequence = self.mediapipe_processor.process_video(video_path)
            
            # Save intermediate results
            keypoints_path = os.path.join(self.config.output_dir, 'keypoints.json')
            with open(keypoints_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                keypoints_json = []
                for kp_data in keypoints_sequence:
                    kp_json = kp_data.copy()
                    kp_json['keypoints'] = kp_data['keypoints'].tolist()
                    keypoints_json.append(kp_json)
                json.dump(keypoints_json, f)
            
            # Stage 2: EasyMoCap SMPL-X fitting
            logger.info("Stage 2: Fitting SMPL-X parameters with EasyMoCap...")
            easymocap_input = self.easymocap_processor.create_easymocap_input(keypoints_sequence)
            smplx_params = self.easymocap_processor.fit_smplx(easymocap_input)
            
            # Stage 3: PyTorch3D rendering
            logger.info("Stage 3: Rendering high-quality meshes with PyTorch3D...")
            camera_params = easymocap_input['camera']
            rendered_frames = self.pytorch3d_renderer.render_sequence(smplx_params, camera_params)
            
            # Stage 4: Export results
            logger.info("Stage 4: Exporting results...")
            results = self._export_results(smplx_params, rendered_frames, keypoints_sequence)
            
            logger.info("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _export_results(self, smplx_params: Dict[str, torch.Tensor], 
                       rendered_frames: List[torch.Tensor],
                       keypoints_sequence: List[Dict]) -> Dict[str, Any]:
        """Export all pipeline results"""
        
        # Save SMPL-X parameters
        params_path = os.path.join(self.config.output_dir, 'smplx_params.pt')
        torch.save(smplx_params, params_path)
        
        # Save rendered video
        video_path = os.path.join(self.config.output_dir, 'rendered_mesh.mp4')
        self._save_video(rendered_frames, video_path)
        
        # Save individual mesh frames (optional)
        meshes_dir = os.path.join(self.config.output_dir, 'meshes')
        os.makedirs(meshes_dir, exist_ok=True)
        
        # Export sample mesh frames as OBJ files
        sample_indices = np.linspace(0, len(rendered_frames)-1, min(10, len(rendered_frames)), dtype=int)
        mesh_files = self._export_sample_meshes(smplx_params, sample_indices, meshes_dir)
        
        results = {
            'keypoints_file': os.path.join(self.config.output_dir, 'keypoints.json'),
            'smplx_params_file': params_path,
            'rendered_video_file': video_path,
            'sample_mesh_files': mesh_files,
            'processing_stats': {
                'total_frames': len(keypoints_sequence),
                'successful_detections': sum(1 for kp in keypoints_sequence if kp['confidence'] > 0.5),
                'average_confidence': np.mean([kp['confidence'] for kp in keypoints_sequence]),
                'output_resolution': self.config.render_resolution,
            }
        }
        
        return results
    
    def _save_video(self, frames: List[torch.Tensor], output_path: str):
        """Save rendered frames as video"""
        if len(frames) == 0:
            logger.warning("No frames to save")
            return
        
        # Convert tensors to numpy arrays
        frame_arrays = []
        for frame in frames:
            # Convert from RGBA to RGB and scale to 0-255
            frame_np = frame.cpu().numpy()
            if frame_np.shape[-1] == 4:  # RGBA
                frame_np = frame_np[..., :3]  # Remove alpha channel
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_arrays.append(frame_np)
        
        # Write video using OpenCV
        height, width = frame_arrays[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.config.fps, (width, height))
        
        for frame in frame_arrays:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        logger.info(f"Saved rendered video to {output_path}")
    
    def _export_sample_meshes(self, smplx_params: Dict[str, torch.Tensor], 
                             sample_indices: np.ndarray, output_dir: str) -> List[str]:
        """Export sample mesh frames as OBJ files"""
        from smplx import SMPLX
        
        # Load SMPL-X model
        body_model = SMPLX(
            model_path=self.config.smplx_model_path,
            gender='neutral',
            use_face_contour=False,
            use_hands=False,
            device=torch.device('cpu')  # Use CPU for export
        )
        
        mesh_files = []
        
        for i, frame_idx in enumerate(sample_indices):
            # Extract parameters for this frame
            frame_params = {k: v[[frame_idx]] for k, v in smplx_params.items()}
            
            # Generate mesh
            with torch.no_grad():
                output = body_model(**frame_params)
                vertices = output.vertices[0].cpu().numpy()
                faces = body_model.faces_tensor.cpu().numpy()
            
            # Save as OBJ file
            obj_path = os.path.join(output_dir, f'mesh_frame_{frame_idx:04d}.obj')
            self._save_obj(vertices, faces, obj_path)
            mesh_files.append(obj_path)
        
        logger.info(f"Saved {len(mesh_files)} sample mesh files to {output_dir}")
        return mesh_files
    
    def _save_obj(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """Save mesh as OBJ file"""
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    """Example usage of the complete pipeline"""
    config = PipelineConfig(
        input_video_path="input_video.mp4",
        output_dir="output_results",
        smplx_model_path="models/smplx",
        render_resolution=1024,
        temporal_smoothing=True,
        use_gpu=True
    )
    
    pipeline = CompletePipeline(config)
    results = pipeline.process_video(config.input_video_path)
    
    print("Pipeline Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
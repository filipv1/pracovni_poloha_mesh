"""
Data Flow Architecture for 3D Human Mesh Processing Pipeline
Detailed specification of data structures and flow between components
"""

from typing import Dict, List, NamedTuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ProcessingStage(Enum):
    """Pipeline processing stages"""
    INPUT_VALIDATION = "input_validation"
    POSE_DETECTION = "pose_detection"
    POSE_ENHANCEMENT = "pose_enhancement"
    MESH_FITTING = "mesh_fitting"
    MESH_ANALYSIS = "mesh_analysis"
    VISUALIZATION = "visualization"
    OUTPUT_GENERATION = "output_generation"

@dataclass
class FrameData:
    """Input frame data structure"""
    frame: np.ndarray  # BGR image (H, W, 3)
    frame_number: int
    timestamp: float
    resolution: tuple  # (width, height)
    fps: float
    
    # Optional metadata
    camera_params: Optional[Dict] = None
    preprocessing_applied: List[str] = None

@dataclass
class PoseDetectionResults:
    """Enhanced pose detection results"""
    # Core MediaPipe results
    pose_landmarks: Optional[object]  # MediaPipe pose landmarks (2D normalized)
    pose_world_landmarks: Optional[List[List[float]]]  # 3D landmarks in meters
    confidence: float
    
    # Enhanced detections
    hand_landmarks: Optional[Dict] = None  # Left/right hand landmarks
    face_landmarks: Optional[List] = None  # Face mesh landmarks
    
    # Quality metrics
    detection_quality: float = 0.0
    landmark_visibility: List[float] = None
    mesh_readiness_score: float = 0.0
    
    # Temporal information
    frame_number: int = 0
    processing_time_ms: float = 0.0

@dataclass
class MeshParameters:
    """Parametric body model parameters"""
    # SMPL parameters
    pose: np.ndarray  # Joint rotations (24, 3) or (55, 3) for SMPL-X
    shape: np.ndarray  # Body shape parameters (10,) or (16,) for SMPL-X
    global_orient: np.ndarray  # Global orientation (3,)
    transl: np.ndarray  # Global translation (3,)
    
    # Optional SMPL-X extensions
    jaw_pose: Optional[np.ndarray] = None  # Jaw rotation (3,)
    leye_pose: Optional[np.ndarray] = None  # Left eye rotation (3,)
    reye_pose: Optional[np.ndarray] = None  # Right eye rotation (3,)
    left_hand_pose: Optional[np.ndarray] = None  # Left hand pose (45,)
    right_hand_pose: Optional[np.ndarray] = None  # Right hand pose (45,)
    
    # Model metadata
    model_type: str = "smpl"  # "smpl", "smpl-x", "mano", "flame"
    gender: str = "neutral"  # "male", "female", "neutral"
    
    # Optimization metadata
    optimization_iterations: int = 0
    convergence_error: float = 0.0
    optimization_time_ms: float = 0.0

@dataclass
class MeshGeometry:
    """3D mesh geometry data"""
    vertices: np.ndarray  # Mesh vertices (N, 3)
    faces: np.ndarray  # Mesh faces (M, 3)
    
    # Optional geometry data
    vertex_normals: Optional[np.ndarray] = None  # Vertex normals (N, 3)
    face_normals: Optional[np.ndarray] = None  # Face normals (M, 3)
    texture_coordinates: Optional[np.ndarray] = None  # UV coordinates (N, 2)
    
    # Mesh statistics
    vertex_count: int = 0
    face_count: int = 0
    surface_area: float = 0.0
    volume: float = 0.0
    
    # Quality metrics
    is_watertight: bool = False
    has_self_intersections: bool = False
    mesh_quality_score: float = 0.0

@dataclass
class MeshAnalysisResults:
    """Results from mesh analysis"""
    frame_number: int
    
    # Geometric analysis
    joint_angles: Dict[str, float]  # Joint name -> angle in degrees
    segment_lengths: Dict[str, float]  # Body segment lengths
    body_proportions: Dict[str, float]  # Anthropometric ratios
    
    # Posture analysis
    trunk_bend_angle: float  # Enhanced from existing analysis
    trunk_bend_direction: str  # "forward", "backward", "left", "right"
    posture_score: float  # 0-1 overall posture quality
    
    # Symmetry analysis
    bilateral_symmetry: Dict[str, float]  # Left-right symmetry scores
    asymmetry_regions: List[str]  # Regions with significant asymmetry
    
    # Movement analysis (temporal)
    velocity_profiles: Optional[Dict[str, np.ndarray]] = None
    acceleration_profiles: Optional[Dict[str, np.ndarray]] = None
    
    # Health and ergonomics
    risk_factors: List[str]  # Identified risk factors
    recommendations: List[str]  # Improvement recommendations
    
    # Confidence and quality
    analysis_confidence: float = 0.0
    data_completeness: float = 0.0

@dataclass
class VisualizationData:
    """Data for 3D visualization"""
    rendered_frame: np.ndarray  # Final rendered frame
    
    # Rendering modes
    skeleton_overlay: bool = True
    mesh_overlay: bool = True
    analysis_overlays: Dict[str, bool] = None
    
    # Visual elements
    color_scheme: str = "default"
    transparency: float = 0.7
    lighting_enabled: bool = True
    
    # 3D view parameters
    camera_position: np.ndarray = None
    camera_target: np.ndarray = None
    view_matrix: np.ndarray = None
    projection_matrix: np.ndarray = None
    
    # Overlay information
    text_overlays: List[Dict] = None
    metric_displays: Dict[str, Union[float, str]] = None
    
    # Performance metrics
    rendering_time_ms: float = 0.0
    frame_rate: float = 0.0

class DataFlowManager:
    """Manages data flow through the pipeline"""
    
    def __init__(self):
        self.pipeline_state = {}
        self.data_cache = {}
        self.performance_metrics = {}
    
    def create_processing_context(self, frame_data: FrameData) -> Dict:
        """Create processing context for a frame"""
        context = {
            'frame_data': frame_data,
            'stage': ProcessingStage.INPUT_VALIDATION,
            'timestamp': frame_data.timestamp,
            'processing_start_time': self._get_current_time(),
            'intermediate_results': {},
            'error_log': [],
            'performance_log': {}
        }
        return context
    
    def validate_stage_input(self, stage: ProcessingStage, data: Dict) -> bool:
        """Validate input data for specific pipeline stage"""
        validation_rules = {
            ProcessingStage.INPUT_VALIDATION: self._validate_input_frame,
            ProcessingStage.POSE_DETECTION: self._validate_pose_input,
            ProcessingStage.POSE_ENHANCEMENT: self._validate_pose_enhancement_input,
            ProcessingStage.MESH_FITTING: self._validate_mesh_fitting_input,
            ProcessingStage.MESH_ANALYSIS: self._validate_mesh_analysis_input,
            ProcessingStage.VISUALIZATION: self._validate_visualization_input,
            ProcessingStage.OUTPUT_GENERATION: self._validate_output_input
        }
        
        validator = validation_rules.get(stage)
        if validator:
            return validator(data)
        return False
    
    def transfer_data(self, from_stage: ProcessingStage, to_stage: ProcessingStage, data: Dict) -> Dict:
        """Handle data transfer between pipeline stages"""
        # Log performance metrics
        self._log_stage_performance(from_stage, data)
        
        # Transform data format if needed
        transformed_data = self._transform_data_format(from_stage, to_stage, data)
        
        # Validate output format
        if not self.validate_stage_input(to_stage, transformed_data):
            raise ValueError(f"Invalid data format for stage {to_stage}")
        
        return transformed_data
    
    def _validate_input_frame(self, data: Dict) -> bool:
        """Validate input frame data"""
        frame_data = data.get('frame_data')
        if not isinstance(frame_data, FrameData):
            return False
        
        frame = frame_data.frame
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        
        return True
    
    def _validate_pose_input(self, data: Dict) -> bool:
        """Validate pose detection input"""
        return 'frame_data' in data and isinstance(data['frame_data'], FrameData)
    
    def _validate_pose_enhancement_input(self, data: Dict) -> bool:
        """Validate pose enhancement input"""
        return ('pose_results' in data and 
                isinstance(data['pose_results'], PoseDetectionResults))
    
    def _validate_mesh_fitting_input(self, data: Dict) -> bool:
        """Validate mesh fitting input"""
        pose_results = data.get('pose_results')
        if not isinstance(pose_results, PoseDetectionResults):
            return False
        
        return (pose_results.pose_world_landmarks is not None and
                pose_results.mesh_readiness_score > 0.5)
    
    def _validate_mesh_analysis_input(self, data: Dict) -> bool:
        """Validate mesh analysis input"""
        return ('mesh_geometry' in data and 
                isinstance(data['mesh_geometry'], MeshGeometry) and
                data['mesh_geometry'].vertices is not None)
    
    def _validate_visualization_input(self, data: Dict) -> bool:
        """Validate visualization input"""
        return 'frame_data' in data and isinstance(data['frame_data'], FrameData)
    
    def _validate_output_input(self, data: Dict) -> bool:
        """Validate output generation input"""
        return 'visualization_data' in data
    
    def _transform_data_format(self, from_stage: ProcessingStage, to_stage: ProcessingStage, data: Dict) -> Dict:
        """Transform data format between stages if needed"""
        # Implementation of stage-specific data transformations
        return data
    
    def _log_stage_performance(self, stage: ProcessingStage, data: Dict):
        """Log performance metrics for a stage"""
        if stage not in self.performance_metrics:
            self.performance_metrics[stage] = []
        
        processing_time = data.get('processing_time_ms', 0.0)
        self.performance_metrics[stage].append(processing_time)
    
    def _get_current_time(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

class PipelineDataFlow:
    """
    Complete data flow specification for the mesh processing pipeline
    """
    
    @staticmethod
    def get_data_flow_specification() -> Dict:
        """Get complete data flow specification"""
        return {
            'input_formats': {
                'video_frame': {
                    'type': 'FrameData',
                    'required_fields': ['frame', 'frame_number', 'timestamp'],
                    'optional_fields': ['camera_params', 'preprocessing_applied'],
                    'constraints': {
                        'frame_shape': '(H, W, 3)',
                        'frame_dtype': 'uint8',
                        'frame_range': '[0, 255]'
                    }
                }
            },
            
            'intermediate_formats': {
                'pose_detection': {
                    'type': 'PoseDetectionResults',
                    'key_fields': ['pose_world_landmarks', 'confidence', 'mesh_readiness_score'],
                    'data_requirements': {
                        'pose_world_landmarks': 'List[List[float]] with shape (33, 3)',
                        'confidence': 'float in [0, 1]',
                        'mesh_readiness_score': 'float in [0, 1]'
                    }
                },
                
                'mesh_parameters': {
                    'type': 'MeshParameters',
                    'key_fields': ['pose', 'shape', 'global_orient', 'transl'],
                    'data_requirements': {
                        'pose': 'np.ndarray shape (24, 3) or (55, 3)',
                        'shape': 'np.ndarray shape (10,) or (16,)',
                        'global_orient': 'np.ndarray shape (3,)',
                        'transl': 'np.ndarray shape (3,)'
                    }
                },
                
                'mesh_geometry': {
                    'type': 'MeshGeometry',
                    'key_fields': ['vertices', 'faces'],
                    'data_requirements': {
                        'vertices': 'np.ndarray shape (N, 3)',
                        'faces': 'np.ndarray shape (M, 3) with dtype int32',
                        'vertex_count': 'int > 0',
                        'face_count': 'int > 0'
                    }
                }
            },
            
            'output_formats': {
                'analysis_results': {
                    'type': 'MeshAnalysisResults',
                    'required_analyses': ['trunk_bend_angle', 'posture_score', 'joint_angles'],
                    'optional_analyses': ['symmetry_analysis', 'movement_analysis']
                },
                
                'visualization': {
                    'type': 'VisualizationData',
                    'required_fields': ['rendered_frame'],
                    'frame_format': 'np.ndarray shape (H, W, 3) dtype uint8'
                }
            },
            
            'pipeline_stages': {
                1: {
                    'name': 'Input Validation',
                    'input': 'FrameData',
                    'output': 'FrameData',
                    'processing_time_target': '<5ms'
                },
                2: {
                    'name': 'Pose Detection',
                    'input': 'FrameData',
                    'output': 'PoseDetectionResults',
                    'processing_time_target': '<50ms'
                },
                3: {
                    'name': 'Mesh Fitting',
                    'input': 'PoseDetectionResults',
                    'output': 'MeshParameters + MeshGeometry',
                    'processing_time_target': '<200ms'
                },
                4: {
                    'name': 'Mesh Analysis',
                    'input': 'MeshGeometry',
                    'output': 'MeshAnalysisResults',
                    'processing_time_target': '<30ms'
                },
                5: {
                    'name': 'Visualization',
                    'input': 'FrameData + MeshGeometry + MeshAnalysisResults',
                    'output': 'VisualizationData',
                    'processing_time_target': '<100ms'
                }
            },
            
            'performance_targets': {
                'total_pipeline_time': '<400ms per frame',
                'real_time_fps': '>2.5 FPS',
                'memory_usage': '<2GB GPU, <4GB RAM',
                'mesh_reconstruction_success_rate': '>80%',
                'pose_detection_success_rate': '>95%'
            }
        }
    
    @staticmethod
    def get_integration_points() -> Dict:
        """Get integration points with existing codebase"""
        return {
            'existing_components': {
                'PoseDetector': {
                    'integration_point': 'Enhanced3DPoseProcessor.__init__',
                    'data_flow': 'FrameData -> PoseDetectionResults',
                    'modifications': 'Extend with hand and face detection'
                },
                
                'TrunkAngleCalculator': {
                    'integration_point': 'MeshAnalyzer._analyze_trunk_bend',
                    'data_flow': 'MeshGeometry -> enhanced trunk analysis',
                    'modifications': 'Add mesh-based angle calculation'
                },
                
                'SkeletonVisualizer': {
                    'integration_point': 'Advanced3DVisualizer._integrate_existing_visualizations',
                    'data_flow': 'VisualizationData -> enhanced rendering',
                    'modifications': 'Add 3D mesh rendering capabilities'
                },
                
                'TrunkAnalysisProcessor': {
                    'integration_point': 'MeshPipeline.__init__',
                    'data_flow': 'Complete pipeline orchestration',
                    'modifications': 'Extend with mesh processing stages'
                }
            },
            
            'backward_compatibility': {
                'maintain_existing_api': True,
                'existing_output_formats': 'Preserved and extended',
                'configuration_compatibility': 'Full backward compatibility',
                'performance_impact': 'Minimal impact on existing functionality'
            }
        }
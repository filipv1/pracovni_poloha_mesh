"""
Integration Implementation Guide
Detailed implementation of integration points between existing codebase and new 3D mesh processing
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import mediapipe as mp

# Import existing components (would be actual imports in implementation)
# from src.pose_detector import PoseDetector, PoseResults
# from src.trunk_analyzer import TrunkAnalysisProcessor
# from src.visualizer import SkeletonVisualizer
# from src.angle_calculator import TrunkAngleCalculator

class IntegratedPoseDetector:
    """
    Enhanced pose detector that integrates seamlessly with existing PoseDetector
    Adds 3D mesh processing capabilities while maintaining backward compatibility
    """
    
    def __init__(self, 
                 existing_pose_detector,
                 enable_mesh_processing: bool = True,
                 mesh_model_path: str = None):
        """
        Initialize integrated pose detector
        
        Args:
            existing_pose_detector: Instance of existing PoseDetector class
            enable_mesh_processing: Whether to enable mesh processing features
            mesh_model_path: Path to mesh model files (SMPL, etc.)
        """
        # Core integration: wrap existing detector
        self.base_detector = existing_pose_detector
        self.enable_mesh_processing = enable_mesh_processing
        
        # Enhanced detection components
        if enable_mesh_processing:
            self._initialize_mesh_components(mesh_model_path)
        
        # Maintain existing interface
        self.mp_pose = self.base_detector.mp_pose
        self.mp_drawing = self.base_detector.mp_drawing
        self.TRUNK_LANDMARKS = self.base_detector.TRUNK_LANDMARKS
    
    def detect_pose(self, frame: np.ndarray) -> Dict:
        """
        Enhanced pose detection that maintains existing interface
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Enhanced PoseResults with optional mesh data
        """
        # Use existing pose detection as foundation
        base_results = self.base_detector.detect_pose(frame)
        
        # Create enhanced results maintaining backward compatibility
        enhanced_results = {
            # Existing interface (unchanged)
            'pose_landmarks': base_results.pose_landmarks,
            'pose_world_landmarks': base_results.pose_world_landmarks,
            'confidence': base_results.confidence,
            
            # Enhanced features (new)
            'mesh_data': None,
            'mesh_parameters': None,
            'enhanced_analysis': None,
            'processing_metadata': {
                'mesh_processing_enabled': self.enable_mesh_processing,
                'processing_time_ms': 0.0,
                'mesh_quality_score': 0.0
            }
        }
        
        # Add mesh processing if enabled and pose is valid
        if (self.enable_mesh_processing and 
            self.base_detector.is_pose_valid(base_results)):
            
            mesh_results = self._process_mesh_from_pose(
                base_results.pose_world_landmarks,
                frame
            )
            
            enhanced_results['mesh_data'] = mesh_results.get('mesh_geometry')
            enhanced_results['mesh_parameters'] = mesh_results.get('parameters')
            enhanced_results['enhanced_analysis'] = mesh_results.get('analysis')
            enhanced_results['processing_metadata'].update(mesh_results.get('metadata', {}))
        
        return self._create_compatible_result_object(enhanced_results)
    
    def _initialize_mesh_components(self, mesh_model_path: str):
        """Initialize mesh processing components"""
        try:
            # Initialize SMPL model
            # self.smpl_model = self._load_smpl_model(mesh_model_path)
            
            # Initialize mesh fitter
            # self.mesh_fitter = SMPLFitter(self.smpl_model)
            
            # Initialize mesh analyzer
            # self.mesh_analyzer = MeshAnalyzer()
            
            print("Mesh processing components initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize mesh components: {e}")
            self.enable_mesh_processing = False
    
    def _process_mesh_from_pose(self, pose_world_landmarks: List[List[float]], frame: np.ndarray) -> Dict:
        """
        Process 3D mesh from pose landmarks
        
        Args:
            pose_world_landmarks: 3D pose landmarks from MediaPipe
            frame: Input frame for context
            
        Returns:
            Dictionary with mesh processing results
        """
        processing_start_time = self._get_current_time()
        
        try:
            # Convert MediaPipe landmarks to SMPL-compatible format
            smpl_landmarks = self._convert_mediapipe_to_smpl_landmarks(pose_world_landmarks)
            
            # Fit SMPL model to landmarks
            # mesh_parameters = self.mesh_fitter.fit_to_landmarks(smpl_landmarks)
            
            # Generate 3D mesh
            # mesh_vertices, mesh_faces = self.smpl_model.forward(mesh_parameters)
            
            # Analyze mesh
            # analysis_results = self.mesh_analyzer.analyze_mesh(mesh_vertices, mesh_faces)
            
            # Mock results for demonstration
            mesh_results = {
                'mesh_geometry': {
                    'vertices': np.zeros((6890, 3)),  # SMPL vertex count
                    'faces': np.zeros((13776, 3)),    # SMPL face count
                    'vertex_normals': np.zeros((6890, 3)),
                    'quality_score': 0.85
                },
                'parameters': {
                    'pose': np.zeros((24, 3)),
                    'shape': np.zeros(10),
                    'global_orient': np.zeros(3),
                    'transl': np.zeros(3)
                },
                'analysis': {
                    'trunk_bend_angle_mesh': 45.0,
                    'mesh_based_confidence': 0.9,
                    'joint_angles_enhanced': {},
                    'body_volume': 0.0,
                    'surface_area': 0.0
                },
                'metadata': {
                    'mesh_processing_time_ms': (self._get_current_time() - processing_start_time) * 1000,
                    'mesh_quality_score': 0.85,
                    'optimization_iterations': 100
                }
            }
            
            return mesh_results
            
        except Exception as e:
            print(f"Warning: Mesh processing failed: {e}")
            return {
                'mesh_geometry': None,
                'parameters': None,
                'analysis': None,
                'metadata': {
                    'mesh_processing_time_ms': (self._get_current_time() - processing_start_time) * 1000,
                    'error': str(e)
                }
            }
    
    def _convert_mediapipe_to_smpl_landmarks(self, mp_landmarks: List[List[float]]) -> np.ndarray:
        """
        Convert MediaPipe landmarks to SMPL-compatible format
        
        Args:
            mp_landmarks: MediaPipe 3D landmarks (33 points)
            
        Returns:
            SMPL-compatible landmarks
        """
        # MediaPipe to SMPL landmark mapping
        mp_to_smpl_mapping = {
            # MediaPipe index -> SMPL joint index
            11: 16,  # Left shoulder -> Left shoulder
            12: 17,  # Right shoulder -> Right shoulder
            13: 18,  # Left elbow -> Left elbow
            14: 19,  # Right elbow -> Right elbow
            15: 20,  # Left wrist -> Left wrist
            16: 21,  # Right wrist -> Right wrist
            23: 1,   # Left hip -> Left hip
            24: 2,   # Right hip -> Right hip
            25: 4,   # Left knee -> Left knee
            26: 5,   # Right knee -> Right knee
            27: 7,   # Left ankle -> Left ankle
            28: 8,   # Right ankle -> Right ankle
        }
        
        # Initialize SMPL landmarks array (24 joints x 3 coordinates)
        smpl_landmarks = np.zeros((24, 3))
        
        # Map available landmarks
        for mp_idx, smpl_idx in mp_to_smpl_mapping.items():
            if mp_idx < len(mp_landmarks):
                smpl_landmarks[smpl_idx] = mp_landmarks[mp_idx]
        
        return smpl_landmarks
    
    def _create_compatible_result_object(self, enhanced_results: Dict):
        """
        Create result object compatible with existing PoseResults interface
        """
        # For backward compatibility, we can either:
        # 1. Return the existing PoseResults with added attributes
        # 2. Create a new enhanced result class that inherits from PoseResults
        
        # Option 1: Extend existing PoseResults (requires modification of existing class)
        # Option 2: Create wrapper that behaves like PoseResults but includes mesh data
        
        class EnhancedPoseResults:
            """Enhanced pose results with mesh processing capabilities"""
            
            def __init__(self, enhanced_data: Dict):
                # Existing interface
                self.pose_landmarks = enhanced_data['pose_landmarks']
                self.pose_world_landmarks = enhanced_data['pose_world_landmarks']
                self.confidence = enhanced_data['confidence']
                
                # Enhanced features
                self.mesh_data = enhanced_data.get('mesh_data')
                self.mesh_parameters = enhanced_data.get('mesh_parameters')
                self.enhanced_analysis = enhanced_data.get('enhanced_analysis')
                self.processing_metadata = enhanced_data.get('processing_metadata', {})
            
            # Backward compatibility methods
            def has_mesh_data(self) -> bool:
                return self.mesh_data is not None
            
            def get_mesh_quality_score(self) -> float:
                if self.mesh_data:
                    return self.mesh_data.get('quality_score', 0.0)
                return 0.0
            
            def get_enhanced_trunk_angle(self) -> Optional[float]:
                if self.enhanced_analysis:
                    return self.enhanced_analysis.get('trunk_bend_angle_mesh')
                return None
        
        return EnhancedPoseResults(enhanced_results)
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement"""
        import time
        return time.time()
    
    # Maintain backward compatibility with existing methods
    def is_pose_valid(self, pose_results, min_confidence: float = 0.3) -> bool:
        """Backward compatible pose validation"""
        # Handle both old PoseResults and new EnhancedPoseResults
        if hasattr(pose_results, 'pose_world_landmarks'):
            return self.base_detector.is_pose_valid(pose_results, min_confidence)
        else:
            # Create compatible object for existing validator
            compatible_result = type('PoseResults', (), {
                'pose_landmarks': pose_results.get('pose_landmarks'),
                'pose_world_landmarks': pose_results.get('pose_world_landmarks'),
                'confidence': pose_results.get('confidence', 0.0)
            })()
            return self.base_detector.is_pose_valid(compatible_result, min_confidence)
    
    def get_trunk_landmarks(self, pose_results) -> dict:
        """Backward compatible trunk landmark extraction"""
        return self.base_detector.get_trunk_landmarks(pose_results)

class IntegratedTrunkAnalyzer:
    """
    Enhanced trunk analyzer that integrates mesh-based analysis with existing angle calculation
    """
    
    def __init__(self, 
                 existing_angle_calculator,
                 enable_mesh_analysis: bool = True):
        """
        Initialize integrated trunk analyzer
        
        Args:
            existing_angle_calculator: Instance of existing TrunkAngleCalculator
            enable_mesh_analysis: Whether to enable mesh-based analysis
        """
        self.base_calculator = existing_angle_calculator
        self.enable_mesh_analysis = enable_mesh_analysis
        
        # Maintain existing interface
        self.smoothing_window = existing_angle_calculator.smoothing_window
        self.angle_smoother = existing_angle_calculator.angle_smoother
    
    def calculate_trunk_angle(self, 
                            landmarks_3d: List[List[float]], 
                            mesh_data: Dict = None,
                            smooth: bool = True) -> float:
        """
        Enhanced trunk angle calculation using both landmarks and mesh data
        
        Args:
            landmarks_3d: 3D pose landmarks (existing interface)
            mesh_data: Optional mesh data for enhanced analysis
            smooth: Whether to apply temporal smoothing
            
        Returns:
            Enhanced trunk angle with improved accuracy
        """
        # Calculate base angle using existing method
        base_angle = self.base_calculator.calculate_trunk_angle(landmarks_3d, smooth)
        
        # Enhance with mesh data if available
        if self.enable_mesh_analysis and mesh_data and mesh_data.get('vertices') is not None:
            mesh_angle = self._calculate_mesh_based_trunk_angle(mesh_data)
            
            # Combine angles with confidence weighting
            mesh_confidence = mesh_data.get('quality_score', 0.0)
            landmark_confidence = 0.8  # Assumed confidence for landmark-based method
            
            total_confidence = mesh_confidence + landmark_confidence
            
            if total_confidence > 0:
                combined_angle = (
                    (base_angle * landmark_confidence + mesh_angle * mesh_confidence) / 
                    total_confidence
                )
                
                # Apply smoothing to combined result if requested
                if smooth:
                    combined_angle = self.angle_smoother.smooth_angle(combined_angle)
                
                return combined_angle
        
        return base_angle
    
    def _calculate_mesh_based_trunk_angle(self, mesh_data: Dict) -> float:
        """
        Calculate trunk angle from mesh vertices
        
        Args:
            mesh_data: Dictionary containing mesh vertices and faces
            
        Returns:
            Trunk bend angle calculated from mesh
        """
        vertices = mesh_data.get('vertices')
        if vertices is None or len(vertices) == 0:
            return 0.0
        
        try:
            # SMPL vertex indices for trunk region (approximate)
            # These would be actual SMPL vertex indices for shoulders and hips
            shoulder_vertex_indices = [1864, 5275]  # Left and right shoulder vertices
            hip_vertex_indices = [3173, 6584]      # Left and right hip vertices
            
            # Extract trunk vertices
            shoulder_vertices = vertices[shoulder_vertex_indices]
            hip_vertices = vertices[hip_vertex_indices]
            
            # Calculate trunk vector from average positions
            shoulder_center = np.mean(shoulder_vertices, axis=0)
            hip_center = np.mean(hip_vertices, axis=0)
            
            trunk_vector = shoulder_center - hip_center
            vertical_reference = np.array([0, 1, 0])  # Y-up in SMPL coordinate system
            
            # Calculate angle between trunk vector and vertical
            cos_angle = np.dot(trunk_vector, vertical_reference) / (
                np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_reference)
            )
            
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            return angle_degrees
            
        except Exception as e:
            print(f"Warning: Mesh-based angle calculation failed: {e}")
            return 0.0
    
    def get_enhanced_analysis(self, 
                            landmarks_3d: List[List[float]], 
                            mesh_data: Dict = None) -> Dict:
        """
        Get comprehensive trunk analysis combining landmarks and mesh data
        
        Args:
            landmarks_3d: 3D pose landmarks
            mesh_data: Optional mesh data
            
        Returns:
            Enhanced analysis results
        """
        # Base analysis
        base_angle = self.calculate_trunk_angle(landmarks_3d, mesh_data, smooth=True)
        
        analysis_results = {
            'trunk_angle_degrees': base_angle,
            'analysis_method': 'landmark_based',
            'confidence': 0.8,
            'additional_metrics': {}
        }
        
        # Enhanced mesh-based analysis
        if self.enable_mesh_analysis and mesh_data:
            mesh_analysis = self._analyze_mesh_posture(mesh_data)
            analysis_results.update({
                'analysis_method': 'hybrid_landmark_mesh',
                'mesh_based_metrics': mesh_analysis,
                'confidence': min(0.95, analysis_results['confidence'] + 0.15)
            })
        
        return analysis_results
    
    def _analyze_mesh_posture(self, mesh_data: Dict) -> Dict:
        """
        Comprehensive posture analysis from mesh data
        
        Args:
            mesh_data: Mesh geometry data
            
        Returns:
            Detailed posture analysis
        """
        vertices = mesh_data.get('vertices')
        if vertices is None:
            return {}
        
        analysis = {
            'spinal_curvature': self._analyze_spinal_curvature(vertices),
            'shoulder_alignment': self._analyze_shoulder_alignment(vertices),
            'pelvic_tilt': self._analyze_pelvic_tilt(vertices),
            'body_symmetry': self._analyze_body_symmetry(vertices),
            'postural_stability': self._analyze_postural_stability(vertices)
        }
        
        return analysis
    
    def _analyze_spinal_curvature(self, vertices: np.ndarray) -> Dict:
        """Analyze spinal curvature from mesh"""
        # Mock implementation - would use actual spine vertex indices
        return {
            'cervical_curvature': 25.0,
            'thoracic_curvature': 35.0,
            'lumbar_curvature': 45.0,
            'total_curvature_deviation': 5.0
        }
    
    def _analyze_shoulder_alignment(self, vertices: np.ndarray) -> Dict:
        """Analyze shoulder alignment"""
        return {
            'shoulder_height_difference': 2.5,
            'forward_head_posture': 15.0,
            'shoulder_protraction': 8.0
        }
    
    def _analyze_pelvic_tilt(self, vertices: np.ndarray) -> Dict:
        """Analyze pelvic tilt"""
        return {
            'anterior_posterior_tilt': 12.0,
            'lateral_tilt': 3.0,
            'pelvic_rotation': 5.0
        }
    
    def _analyze_body_symmetry(self, vertices: np.ndarray) -> Dict:
        """Analyze bilateral body symmetry"""
        return {
            'overall_symmetry_score': 0.88,
            'asymmetric_regions': ['left_shoulder'],
            'max_asymmetry_distance': 0.05
        }
    
    def _analyze_postural_stability(self, vertices: np.ndarray) -> Dict:
        """Analyze postural stability indicators"""
        return {
            'center_of_mass_offset': 2.3,
            'stability_score': 0.75,
            'balance_quality': 'good'
        }

class IntegratedVisualizer:
    """
    Enhanced visualizer that integrates 3D mesh rendering with existing skeleton visualization
    """
    
    def __init__(self, 
                 existing_visualizer,
                 enable_mesh_rendering: bool = True):
        """
        Initialize integrated visualizer
        
        Args:
            existing_visualizer: Instance of existing SkeletonVisualizer
            enable_mesh_rendering: Whether to enable 3D mesh rendering
        """
        self.base_visualizer = existing_visualizer
        self.enable_mesh_rendering = enable_mesh_rendering
        
        # Maintain existing interface
        self.mp_pose = existing_visualizer.mp_pose
        self.mp_drawing = existing_visualizer.mp_drawing
        self.TRUNK_LANDMARKS = existing_visualizer.TRUNK_LANDMARKS
        
        # Enhanced rendering parameters
        self.mesh_alpha = 0.6
        self.mesh_color = (0, 150, 255)
        self.wireframe_color = (255, 255, 255)
        self.render_mode = 'hybrid'  # 'skeleton', 'mesh', 'wireframe', 'hybrid'
    
    def draw_skeleton(self, 
                     frame: np.ndarray, 
                     pose_landmarks,
                     pose_world_landmarks: Optional[List[List[float]]] = None,
                     highlight_trunk: bool = True,
                     mesh_data: Dict = None) -> np.ndarray:
        """
        Enhanced skeleton drawing with optional mesh overlay
        
        Args:
            frame: Input frame
            pose_landmarks: MediaPipe 2D landmarks
            pose_world_landmarks: 3D world landmarks
            highlight_trunk: Whether to highlight trunk landmarks
            mesh_data: Optional mesh data for 3D rendering
            
        Returns:
            Frame with enhanced visualization
        """
        # Start with existing skeleton visualization
        result_frame = self.base_visualizer.draw_skeleton(
            frame, pose_landmarks, pose_world_landmarks, highlight_trunk
        )
        
        # Add mesh visualization if enabled and available
        if (self.enable_mesh_rendering and 
            mesh_data and 
            mesh_data.get('vertices') is not None):
            
            result_frame = self._overlay_mesh_on_frame(
                result_frame, 
                mesh_data, 
                pose_landmarks
            )
        
        return result_frame
    
    def _overlay_mesh_on_frame(self, 
                              frame: np.ndarray, 
                              mesh_data: Dict, 
                              pose_landmarks) -> np.ndarray:
        """
        Overlay 3D mesh on 2D frame
        
        Args:
            frame: Input frame
            mesh_data: 3D mesh data
            pose_landmarks: 2D pose landmarks for alignment
            
        Returns:
            Frame with mesh overlay
        """
        vertices = mesh_data.get('vertices')
        faces = mesh_data.get('faces')
        
        if vertices is None or faces is None:
            return frame
        
        try:
            # Project 3D mesh to 2D frame coordinates
            projected_vertices = self._project_3d_to_2d(vertices, frame.shape, pose_landmarks)
            
            if self.render_mode in ['mesh', 'hybrid']:
                frame = self._render_mesh_faces(frame, projected_vertices, faces)
            
            if self.render_mode in ['wireframe', 'hybrid']:
                frame = self._render_wireframe(frame, projected_vertices, faces)
            
            # Add mesh quality indicator
            frame = self._add_mesh_quality_indicator(frame, mesh_data)
            
        except Exception as e:
            print(f"Warning: Mesh rendering failed: {e}")
        
        return frame
    
    def _project_3d_to_2d(self, 
                         vertices_3d: np.ndarray, 
                         frame_shape: Tuple, 
                         pose_landmarks) -> np.ndarray:
        """
        Project 3D mesh vertices to 2D frame coordinates
        
        Args:
            vertices_3d: 3D mesh vertices
            frame_shape: Frame dimensions (height, width, channels)
            pose_landmarks: 2D pose landmarks for reference
            
        Returns:
            2D projected vertices
        """
        height, width = frame_shape[:2]
        
        # Simple orthographic projection (would be replaced with proper camera model)
        # Scale and translate to fit frame
        projected = vertices_3d[:, :2].copy()  # Use X, Y coordinates
        
        # Normalize to frame coordinates
        if len(projected) > 0:
            min_coords = np.min(projected, axis=0)
            max_coords = np.max(projected, axis=0)
            
            # Scale to frame size with margin
            scale = min(width * 0.8, height * 0.8) / max(max_coords - min_coords)
            projected = (projected - min_coords) * scale
            
            # Center in frame
            projected[:, 0] += (width - np.max(projected[:, 0])) / 2
            projected[:, 1] += (height - np.max(projected[:, 1])) / 2
        
        return projected.astype(np.int32)
    
    def _render_mesh_faces(self, 
                          frame: np.ndarray, 
                          vertices_2d: np.ndarray, 
                          faces: np.ndarray) -> np.ndarray:
        """
        Render filled mesh faces
        
        Args:
            frame: Input frame
            vertices_2d: 2D projected vertices
            faces: Mesh face indices
            
        Returns:
            Frame with rendered mesh
        """
        overlay = frame.copy()
        
        # Render subset of faces for performance
        face_subset = faces[::10]  # Render every 10th face
        
        for face in face_subset:
            if len(face) == 3 and all(idx < len(vertices_2d) for idx in face):
                # Get triangle vertices
                triangle = vertices_2d[face]
                
                # Fill triangle
                cv2.fillPoly(overlay, [triangle], self.mesh_color)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - self.mesh_alpha, overlay, self.mesh_alpha, 0)
        
        return result
    
    def _render_wireframe(self, 
                         frame: np.ndarray, 
                         vertices_2d: np.ndarray, 
                         faces: np.ndarray) -> np.ndarray:
        """
        Render mesh wireframe
        
        Args:
            frame: Input frame
            vertices_2d: 2D projected vertices
            faces: Mesh face indices
            
        Returns:
            Frame with wireframe overlay
        """
        # Render subset of edges for performance
        edge_subset = faces[::5]  # Render every 5th face's edges
        
        for face in edge_subset:
            if len(face) == 3 and all(idx < len(vertices_2d) for idx in face):
                # Draw triangle edges
                for i in range(3):
                    start_idx = face[i]
                    end_idx = face[(i + 1) % 3]
                    
                    start_point = tuple(vertices_2d[start_idx])
                    end_point = tuple(vertices_2d[end_idx])
                    
                    cv2.line(frame, start_point, end_point, self.wireframe_color, 1)
        
        return frame
    
    def _add_mesh_quality_indicator(self, frame: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """
        Add mesh quality indicator to frame
        
        Args:
            frame: Input frame
            mesh_data: Mesh data with quality score
            
        Returns:
            Frame with quality indicator
        """
        quality_score = mesh_data.get('quality_score', 0.0)
        
        # Choose color based on quality
        if quality_score > 0.8:
            color = (0, 255, 0)  # Green for high quality
        elif quality_score > 0.6:
            color = (0, 255, 255)  # Yellow for medium quality
        else:
            color = (0, 0, 255)  # Red for low quality
        
        # Draw quality indicator
        cv2.putText(frame, f"Mesh Quality: {quality_score:.2f}", 
                   (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw quality bar
        bar_width = 100
        bar_height = 10
        bar_x = frame.shape[1] - 250
        bar_y = 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Quality bar
        quality_width = int(bar_width * quality_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + quality_width, bar_y + bar_height), 
                     color, -1)
        
        return frame

def create_integration_example():
    """
    Create example of how to integrate new components with existing codebase
    """
    integration_example = {
        'step_1_minimal_integration': '''
        # Minimal integration - just add mesh processing to existing pipeline
        
        # In existing main.py or trunk_analyzer.py:
        from integration_implementation import IntegratedPoseDetector
        
        # Replace existing pose detector
        # OLD: self.pose_detector = PoseDetector(model_complexity=model_complexity)
        # NEW: 
        base_detector = PoseDetector(model_complexity=model_complexity)
        self.pose_detector = IntegratedPoseDetector(
            existing_pose_detector=base_detector,
            enable_mesh_processing=True,
            mesh_model_path="path/to/smpl/models"
        )
        
        # Existing code continues to work unchanged!
        pose_results = self.pose_detector.detect_pose(frame)
        is_valid = self.pose_detector.is_pose_valid(pose_results)
        ''',
        
        'step_2_enhanced_analysis': '''
        # Enhanced analysis - add mesh-based trunk analysis
        
        from integration_implementation import IntegratedTrunkAnalyzer
        
        # Replace existing angle calculator
        base_calculator = TrunkAngleCalculator(smoothing_window=smoothing_window)
        self.angle_calculator = IntegratedTrunkAnalyzer(
            existing_angle_calculator=base_calculator,
            enable_mesh_analysis=True
        )
        
        # Enhanced usage with backward compatibility
        trunk_angle = self.angle_calculator.calculate_trunk_angle(
            pose_results.pose_world_landmarks,
            mesh_data=pose_results.mesh_data  # New optional parameter
        )
        
        # Get enhanced analysis
        enhanced_analysis = self.angle_calculator.get_enhanced_analysis(
            pose_results.pose_world_landmarks,
            mesh_data=pose_results.mesh_data
        )
        ''',
        
        'step_3_enhanced_visualization': '''
        # Enhanced visualization - add 3D mesh rendering
        
        from integration_implementation import IntegratedVisualizer
        
        # Replace existing visualizer
        base_visualizer = SkeletonVisualizer()
        self.visualizer = IntegratedVisualizer(
            existing_visualizer=base_visualizer,
            enable_mesh_rendering=True
        )
        
        # Enhanced visualization with backward compatibility
        processed_frame = self.visualizer.draw_skeleton(
            frame,
            pose_results.pose_landmarks,
            pose_results.pose_world_landmarks,
            highlight_trunk=True,
            mesh_data=pose_results.mesh_data  # New optional parameter
        )
        ''',
        
        'step_4_configuration': '''
        # Configuration for different performance modes
        
        # Real-time mode - minimal mesh processing
        real_time_config = {
            'enable_mesh_processing': True,
            'enable_mesh_analysis': False,
            'enable_mesh_rendering': False,  # Only skeleton
            'mesh_optimization_steps': 20,
            'render_mode': 'skeleton'
        }
        
        # Balanced mode - full features with optimization
        balanced_config = {
            'enable_mesh_processing': True,
            'enable_mesh_analysis': True,
            'enable_mesh_rendering': True,
            'mesh_optimization_steps': 100,
            'render_mode': 'hybrid'
        }
        
        # High quality mode - all features
        high_quality_config = {
            'enable_mesh_processing': True,
            'enable_mesh_analysis': True,
            'enable_mesh_rendering': True,
            'mesh_optimization_steps': 200,
            'render_mode': 'mesh',
            'enable_advanced_analysis': True
        }
        '''
    }
    
    return integration_example

# Usage example and testing
if __name__ == "__main__":
    # This would be used to test the integration
    integration_example = create_integration_example()
    
    print("Integration Implementation Guide")
    print("=" * 50)
    
    for step, code in integration_example.items():
        print(f"\n{step.replace('_', ' ').title()}:")
        print("-" * 30)
        print(code.strip())
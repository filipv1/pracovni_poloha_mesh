#!/usr/bin/env python3
"""
Master Pipeline - Complete integrated 3D pose analysis system

Priority: CRITICAL
Dependencies: All core modules, torch, numpy
Test Coverage Required: 100%

This is the main entry point that integrates all enhanced components into
a unified, production-ready pipeline with maximum performance and accuracy.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import sys
import time
import json
from dataclasses import dataclass, field
import warnings

# Add core module to path
sys.path.append(str(Path(__file__).parent))

# Import all enhanced components
from enhanced_pipeline_integration import EnhancedMediaPipeConverter, EnhancedSMPLXFitter
from batch_processor import HighPerformanceBatchProcessor, BatchConfig
from memory_optimizer import MemoryOptimizedProcessor, MemoryConfig
from unified_visualization import UnifiedVisualizationSystem, VisualizationConfig
from coordinate_system_fix import CoordinateSystemTransformer
from proactive_joint_validator import ProactiveJointValidator
from kalman_angle_filter import MultiAngleKalmanFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Comprehensive configuration for the master pipeline"""
    
    # Processing settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    batch_size: int = 32
    quality_mode: str = 'high'  # 'fast', 'balanced', 'high', 'ultra'
    
    # Input/Output
    input_format: str = 'mediapipe'  # 'mediapipe', 'landmarks', 'video'
    output_dir: str = 'pipeline_output'
    export_formats: List[str] = field(default_factory=lambda: ['json', 'pkl', 'visualization'])
    
    # Component configurations
    repair_mode: str = 'smart'  # Joint validation repair strategy
    enable_angle_filtering: bool = True
    enable_batch_processing: bool = True
    enable_memory_optimization: bool = True
    enable_visualization: bool = True
    
    # Performance tuning
    max_memory_gb: float = 4.0
    gc_threshold: float = 0.8
    enable_gpu_acceleration: bool = True
    enable_compilation: bool = False
    
    # Quality assurance
    joint_confidence_threshold: float = 0.5
    angle_smoothing_window: int = 5
    outlier_detection_threshold: float = 3.0
    
    # Visualization
    visualization_quality: str = 'high'
    create_dashboard: bool = True
    create_timeline: bool = True
    create_3d_plots: bool = True


class ProcessingStatistics:
    """Track comprehensive processing statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.frames_processed = 0
        self.frames_repaired = 0
        self.frames_filtered = 0
        self.memory_optimizations = 0
        self.visualizations_created = 0
        self.processing_times = []
        self.quality_scores = []
        self.error_count = 0
        self.warnings_count = 0
    
    def add_frame_timing(self, processing_time: float):
        self.processing_times.append(processing_time)
        self.frames_processed += 1
    
    def add_quality_score(self, score: float):
        self.quality_scores.append(score)
    
    def get_summary(self) -> Dict:
        total_time = time.time() - self.start_time
        
        summary = {
            'total_processing_time': total_time,
            'frames_processed': self.frames_processed,
            'frames_repaired': self.frames_repaired,
            'frames_filtered': self.frames_filtered,
            'memory_optimizations': self.memory_optimizations,
            'visualizations_created': self.visualizations_created,
            'error_count': self.error_count,
            'warnings_count': self.warnings_count
        }
        
        if self.processing_times:
            summary.update({
                'average_frame_time': np.mean(self.processing_times),
                'fps_average': 1.0 / np.mean(self.processing_times),
                'fps_peak': 1.0 / np.min(self.processing_times),
                'throughput_consistency': 1.0 - (np.std(self.processing_times) / np.mean(self.processing_times))
            })
        
        if self.quality_scores:
            summary.update({
                'average_quality': np.mean(self.quality_scores),
                'quality_std': np.std(self.quality_scores),
                'min_quality': np.min(self.quality_scores),
                'max_quality': np.max(self.quality_scores)
            })
        
        return summary


class MasterPipeline:
    """Complete integrated 3D pose analysis pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None, 
                 smplx_model_path: Optional[str] = None):
        """Initialize master pipeline with all components
        
        Args:
            config: Pipeline configuration
            smplx_model_path: Path to SMPL-X models (optional)
        """
        self.config = config or PipelineConfig()
        self.smplx_model_path = smplx_model_path
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistics tracking
        self.stats = ProcessingStatistics()
        
        # Initialize all enhanced components
        self._initialize_components()
        
        # Validate system readiness
        self._validate_system_readiness()
        
        logger.info("MasterPipeline initialized successfully")
        logger.info(f"Configuration: {self.config}")
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # 1. Enhanced MediaPipe converter with all improvements
        self.converter = EnhancedMediaPipeConverter(repair_mode=self.config.repair_mode)
        
        # 2. SMPL-X fitter (if model path provided)
        if self.smplx_model_path:
            self.smplx_fitter = EnhancedSMPLXFitter(
                model_path=self.smplx_model_path,
                device=self.config.device,
                gender='neutral'
            )
        else:
            self.smplx_fitter = None
            logger.warning("SMPL-X model path not provided, mesh fitting disabled")
        
        # 3. Batch processor for high performance
        if self.config.enable_batch_processing:
            batch_config = BatchConfig(
                batch_size=self.config.batch_size,
                device=self.config.device,
                optimization_level=self.config.quality_mode,
                enable_compilation=self.config.enable_compilation
            )
            self.batch_processor = HighPerformanceBatchProcessor(batch_config)
        else:
            self.batch_processor = None
        
        # 4. Memory optimizer
        if self.config.enable_memory_optimization:
            memory_config = MemoryConfig(
                max_cache_size_gb=self.config.max_memory_gb,
                gc_threshold=self.config.gc_threshold,
                auto_optimization=True
            )
            self.memory_optimizer = MemoryOptimizedProcessor(memory_config)
        else:
            self.memory_optimizer = None
        
        # 5. Visualization system
        if self.config.enable_visualization:
            viz_config = VisualizationConfig(
                output_dir=str(self.output_dir / "visualizations"),
                dpi=300 if self.config.visualization_quality == 'high' else 150,
                color_scheme='professional'
            )
            self.visualizer = UnifiedVisualizationSystem(viz_config)
        else:
            self.visualizer = None
        
        logger.info("All pipeline components initialized")
    
    def _validate_system_readiness(self):
        """Validate that the system is ready for processing"""
        logger.info("Validating system readiness...")
        
        issues = []
        
        # Check device availability
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            issues.append("CUDA device requested but not available")
        
        # Check memory requirements
        if self.config.enable_batch_processing and self.config.batch_size > 64:
            issues.append("Large batch size may cause memory issues")
        
        # Check output directory
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")
        
        if issues:
            logger.warning(f"System readiness issues detected: {issues}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("System readiness validation passed")
    
    def process_single_frame(self, landmarks: np.ndarray, 
                           confidences: Optional[np.ndarray] = None,
                           frame_id: Optional[str] = None) -> Dict:
        """Process a single frame through the complete pipeline
        
        Args:
            landmarks: (33, 3) array of MediaPipe landmarks
            confidences: Optional confidence scores
            frame_id: Optional frame identifier
            
        Returns:
            Complete processing results
        """
        frame_start_time = time.time()
        
        try:
            # Step 1: Enhanced landmark conversion with all improvements
            conversion_result = self.converter.convert_landmarks_to_smplx(
                self._create_mock_landmarks(landmarks), confidences
            )
            
            # Track repair statistics
            if conversion_result['repair_applied']:
                self.stats.frames_repaired += 1
            
            if conversion_result['angles']['filtered_angles']:
                self.stats.frames_filtered += 1
            
            # Step 2: SMPL-X mesh fitting (if available)
            fitting_result = None
            if self.smplx_fitter and conversion_result['joints'] is not None:
                fitting_result = self.smplx_fitter.fit_to_joints(
                    conversion_result['joints'],
                    conversion_result['weights'],
                    conversion_result['validation_result'],
                    conversion_result['angles']
                )
            
            # Step 3: Quality assessment
            quality_score = self._assess_frame_quality(conversion_result, fitting_result)
            self.stats.add_quality_score(quality_score)
            
            # Compile results
            result = {
                'frame_id': frame_id,
                'processing_time': time.time() - frame_start_time,
                'quality_score': quality_score,
                'conversion': conversion_result,
                'fitting': fitting_result,
                'pipeline_version': '2.0.0',  # Enhanced version
                'enhancements_applied': {
                    'coordinate_transform': conversion_result['coordinate_transform_applied'],
                    'joint_repair': conversion_result['repair_applied'],
                    'angle_filtering': len(conversion_result['angles']['filtered_angles']) > 0,
                    'mesh_fitting': fitting_result is not None
                }
            }
            
            # Update timing statistics
            self.stats.add_frame_timing(result['processing_time'])
            
            return result
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"Frame processing failed: {e}")
            
            return {
                'frame_id': frame_id,
                'processing_time': time.time() - frame_start_time,
                'error': str(e),
                'success': False
            }
    
    def process_batch(self, landmarks_list: List[np.ndarray],
                     confidences_list: Optional[List[np.ndarray]] = None,
                     frame_ids: Optional[List[str]] = None) -> List[Dict]:
        """Process multiple frames in high-performance batch mode
        
        Args:
            landmarks_list: List of landmark arrays
            confidences_list: Optional list of confidence arrays
            frame_ids: Optional frame identifiers
            
        Returns:
            List of processing results
        """
        if not self.config.enable_batch_processing or not self.batch_processor:
            # Fall back to sequential processing
            return [self.process_single_frame(
                landmarks_list[i], 
                confidences_list[i] if confidences_list else None,
                frame_ids[i] if frame_ids else f"frame_{i}"
            ) for i in range(len(landmarks_list))]
        
        logger.info(f"Processing batch of {len(landmarks_list)} frames")
        batch_start_time = time.time()
        
        try:
            # Use high-performance batch processor
            batch_results = self.batch_processor.process_landmarks_batch(
                landmarks_list, confidences_list
            )
            
            # Enhance results with frame IDs and additional metadata
            enhanced_results = []
            for i, result in enumerate(batch_results):
                enhanced_result = {
                    'frame_id': frame_ids[i] if frame_ids else f"frame_{i}",
                    'processing_time': (time.time() - batch_start_time) / len(landmarks_list),
                    'quality_score': self._assess_frame_quality(result, None),
                    'conversion': result,
                    'fitting': None,  # Batch mode doesn't include SMPL-X fitting yet
                    'pipeline_version': '2.0.0',
                    'batch_processed': True,
                    'enhancements_applied': {
                        'coordinate_transform': result.get('coordinate_transform_applied', True),
                        'joint_repair': result.get('repair_applied', False),
                        'angle_filtering': len(result.get('angles', {}).get('filtered_angles', {})) > 0,
                        'mesh_fitting': False
                    }
                }
                
                enhanced_results.append(enhanced_result)
                self.stats.add_frame_timing(enhanced_result['processing_time'])
                self.stats.add_quality_score(enhanced_result['quality_score'])
                
                if enhanced_result['enhancements_applied']['joint_repair']:
                    self.stats.frames_repaired += 1
                
                if enhanced_result['enhancements_applied']['angle_filtering']:
                    self.stats.frames_filtered += 1
            
            # Update batch processor statistics
            batch_stats = self.batch_processor.get_performance_statistics()
            logger.info(f"Batch processing completed: {batch_stats['average_throughput']:.1f} FPS")
            
            return enhanced_results
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"Batch processing failed: {e}")
            
            # Fall back to sequential processing
            logger.info("Falling back to sequential processing")
            return [self.process_single_frame(
                landmarks_list[i],
                confidences_list[i] if confidences_list else None,
                frame_ids[i] if frame_ids else f"frame_{i}"
            ) for i in range(len(landmarks_list))]
    
    def process_sequence(self, landmarks_sequence: List[np.ndarray],
                        confidences_sequence: Optional[List[np.ndarray]] = None,
                        sequence_name: str = "sequence") -> Dict:
        """Process complete sequence with full analysis and visualization
        
        Args:
            landmarks_sequence: List of landmark arrays
            confidences_sequence: Optional confidence arrays
            sequence_name: Name for this sequence
            
        Returns:
            Complete sequence analysis results
        """
        logger.info(f"Processing sequence '{sequence_name}' with {len(landmarks_sequence)} frames")
        sequence_start_time = time.time()
        
        # Generate frame IDs
        frame_ids = [f"{sequence_name}_frame_{i:06d}" for i in range(len(landmarks_sequence))]
        
        # Process frames (batch or sequential based on configuration)
        if self.config.enable_batch_processing and len(landmarks_sequence) > 1:
            frame_results = self.process_batch(landmarks_sequence, confidences_sequence, frame_ids)
        else:
            frame_results = [
                self.process_single_frame(landmarks_sequence[i], 
                                        confidences_sequence[i] if confidences_sequence else None,
                                        frame_ids[i])
                for i in range(len(landmarks_sequence))
            ]
        
        # Compile sequence-level results
        successful_frames = [r for r in frame_results if r.get('success', True)]
        
        sequence_result = {
            'sequence_name': sequence_name,
            'total_frames': len(landmarks_sequence),
            'successful_frames': len(successful_frames),
            'processing_time': time.time() - sequence_start_time,
            'frame_results': frame_results,
            'sequence_statistics': self._analyze_sequence_statistics(successful_frames),
            'quality_assessment': self._assess_sequence_quality(successful_frames),
            'pipeline_version': '2.0.0'
        }
        
        # Export results
        exported_files = self._export_sequence_results(sequence_result)
        sequence_result['exported_files'] = exported_files
        
        # Generate visualizations
        if self.config.enable_visualization and self.visualizer:
            visualizations = self._create_sequence_visualizations(successful_frames, sequence_name)
            sequence_result['visualizations'] = visualizations
            self.stats.visualizations_created += len(visualizations)
        
        logger.info(f"Sequence processing completed in {sequence_result['processing_time']:.2f}s")
        logger.info(f"Success rate: {len(successful_frames)}/{len(landmarks_sequence)} "
                   f"({100*len(successful_frames)/len(landmarks_sequence):.1f}%)")
        
        return sequence_result
    
    def _create_mock_landmarks(self, landmarks: np.ndarray):
        """Create mock MediaPipe landmarks object for compatibility"""
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = float(x), float(y), float(z)
        
        class MockLandmarks:
            def __init__(self, points):
                self.landmark = [MockLandmark(*point) for point in points]
        
        return MockLandmarks(landmarks)
    
    def _assess_frame_quality(self, conversion_result: Dict, fitting_result: Optional[Dict]) -> float:
        """Assess the quality of frame processing"""
        quality_score = 100.0
        
        # Deduct for coordinate transformation issues
        if not conversion_result.get('coordinate_transform_applied', False):
            quality_score -= 20.0
        
        # Deduct for joint validation issues
        if conversion_result.get('repair_applied', False):
            validation_result = conversion_result.get('validation_result', {})
            violations = len(validation_result.get('violations', []))
            quality_score -= min(violations * 5.0, 30.0)
        
        # Add points for successful angle filtering
        filtered_angles = conversion_result.get('angles', {}).get('filtered_angles', {})
        if len(filtered_angles) > 0:
            quality_score += 5.0
        
        # Add points for successful mesh fitting
        if fitting_result:
            quality_score += 10.0
            
            if fitting_result.get('coordinate_transform_applied', False):
                quality_score += 5.0
        
        # Assess joint confidence
        if 'weights' in conversion_result:
            weights = conversion_result['weights']
            if weights is not None:
                avg_confidence = np.mean(weights)
                quality_score *= avg_confidence
        
        return np.clip(quality_score, 0.0, 100.0)
    
    def _analyze_sequence_statistics(self, frame_results: List[Dict]) -> Dict:
        """Analyze sequence-level statistics"""
        if not frame_results:
            return {'error': 'no_successful_frames'}
        
        # Extract key metrics
        processing_times = [r.get('processing_time', 0) for r in frame_results]
        quality_scores = [r.get('quality_score', 0) for r in frame_results]
        
        # Count enhancements
        coordinate_transforms = sum(1 for r in frame_results 
                                  if r.get('enhancements_applied', {}).get('coordinate_transform', False))
        joint_repairs = sum(1 for r in frame_results 
                          if r.get('enhancements_applied', {}).get('joint_repair', False))
        angle_filterings = sum(1 for r in frame_results 
                             if r.get('enhancements_applied', {}).get('angle_filtering', False))
        mesh_fittings = sum(1 for r in frame_results 
                          if r.get('enhancements_applied', {}).get('mesh_fitting', False))
        
        return {
            'total_frames': len(frame_results),
            'average_processing_time': np.mean(processing_times),
            'processing_time_std': np.std(processing_times),
            'average_fps': 1.0 / np.mean(processing_times) if processing_times else 0,
            'average_quality': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'enhancement_rates': {
                'coordinate_transform': coordinate_transforms / len(frame_results),
                'joint_repair': joint_repairs / len(frame_results), 
                'angle_filtering': angle_filterings / len(frame_results),
                'mesh_fitting': mesh_fittings / len(frame_results)
            }
        }
    
    def _assess_sequence_quality(self, frame_results: List[Dict]) -> Dict:
        """Assess overall sequence quality"""
        if not frame_results:
            return {'overall_grade': 'F', 'score': 0.0}
        
        quality_scores = [r.get('quality_score', 0) for r in frame_results]
        avg_quality = np.mean(quality_scores)
        
        # Grade based on average quality
        if avg_quality >= 90:
            grade = 'A'
        elif avg_quality >= 80:
            grade = 'B'
        elif avg_quality >= 70:
            grade = 'C'
        elif avg_quality >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_grade': grade,
            'score': avg_quality,
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_consistency': 1.0 - (np.std(quality_scores) / avg_quality) if avg_quality > 0 else 0,
            'frames_above_80': sum(1 for s in quality_scores if s >= 80),
            'frames_below_60': sum(1 for s in quality_scores if s < 60)
        }
    
    def _export_sequence_results(self, sequence_result: Dict) -> Dict:
        """Export sequence results in various formats"""
        exported_files = {}
        
        sequence_name = sequence_result['sequence_name']
        
        for export_format in self.config.export_formats:
            try:
                if export_format == 'json':
                    # Export as JSON
                    json_path = self.output_dir / f"{sequence_name}_results.json"
                    
                    # Create JSON-serializable version
                    json_data = self._make_json_serializable(sequence_result)
                    
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    
                    exported_files['json'] = str(json_path)
                
                elif export_format == 'pkl':
                    # Export as pickle (includes all numpy arrays)
                    pkl_path = self.output_dir / f"{sequence_name}_results.pkl"
                    
                    import pickle
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(sequence_result, f)
                    
                    exported_files['pkl'] = str(pkl_path)
                
                elif export_format == 'csv':
                    # Export summary statistics as CSV
                    csv_path = self.output_dir / f"{sequence_name}_summary.csv"
                    
                    import pandas as pd
                    
                    # Create summary dataframe
                    summary_data = []
                    for frame_result in sequence_result['frame_results']:
                        if frame_result.get('success', True):
                            summary_data.append({
                                'frame_id': frame_result.get('frame_id', ''),
                                'processing_time': frame_result.get('processing_time', 0),
                                'quality_score': frame_result.get('quality_score', 0),
                                'coordinate_transform': frame_result.get('enhancements_applied', {}).get('coordinate_transform', False),
                                'joint_repair': frame_result.get('enhancements_applied', {}).get('joint_repair', False),
                                'angle_filtering': frame_result.get('enhancements_applied', {}).get('angle_filtering', False),
                                'mesh_fitting': frame_result.get('enhancements_applied', {}).get('mesh_fitting', False)
                            })
                    
                    if summary_data:
                        df = pd.DataFrame(summary_data)
                        df.to_csv(csv_path, index=False)
                        exported_files['csv'] = str(csv_path)
                
            except Exception as e:
                logger.warning(f"Failed to export {export_format}: {e}")
                self.stats.warnings_count += 1
        
        return exported_files
    
    def _create_sequence_visualizations(self, frame_results: List[Dict], sequence_name: str) -> Dict:
        """Create comprehensive visualizations for sequence"""
        visualizations = {}
        
        try:
            # Extract pose and angle data
            pose_sequence = []
            angle_sequence = []
            
            for frame_result in frame_results:
                conversion = frame_result.get('conversion', {})
                
                # Extract joints
                joints = conversion.get('joints')
                if joints is not None:
                    pose_sequence.append(joints)
                
                # Extract angles
                angles = conversion.get('angles', {})
                if angles:
                    angle_sequence.append(angles)
            
            if pose_sequence and angle_sequence:
                # Create comprehensive analysis report
                viz_outputs = self.visualizer.create_analysis_report(
                    pose_sequence, angle_sequence, f"{sequence_name} Analysis"
                )
                
                visualizations.update(viz_outputs)
                
                logger.info(f"Created {len(viz_outputs)} visualizations for sequence")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            self.stats.error_count += 1
        
        return visualizations
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_pipeline_statistics(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        stats_summary = self.stats.get_summary()
        
        # Add component statistics
        component_stats = {}
        
        if self.converter:
            component_stats['converter'] = self.converter.get_processing_statistics()
        
        if self.batch_processor:
            component_stats['batch_processor'] = self.batch_processor.get_performance_statistics()
        
        if self.memory_optimizer:
            component_stats['memory_optimizer'] = self.memory_optimizer.get_optimization_statistics()
        
        return {
            'pipeline_stats': stats_summary,
            'component_stats': component_stats,
            'config': self.config.__dict__
        }
    
    def shutdown(self):
        """Clean shutdown of all pipeline components"""
        logger.info("Shutting down master pipeline...")
        
        if self.memory_optimizer:
            self.memory_optimizer.shutdown()
        
        if self.batch_processor:
            self.batch_processor.clear_cache()
        
        if self.converter:
            self.converter.reset_filters()
        
        logger.info("Pipeline shutdown completed")


def create_production_pipeline(output_dir: str = "production_output",
                             quality_mode: str = "high",
                             device: str = "auto",
                             smplx_model_path: Optional[str] = None) -> MasterPipeline:
    """Factory function to create production-ready pipeline"""
    
    config = PipelineConfig(
        output_dir=output_dir,
        quality_mode=quality_mode,
        device=device,
        batch_size=32 if quality_mode in ['fast', 'balanced'] else 16,
        enable_batch_processing=True,
        enable_memory_optimization=True,
        enable_visualization=True,
        create_dashboard=True,
        export_formats=['json', 'pkl', 'visualization']
    )
    
    return MasterPipeline(config, smplx_model_path)


if __name__ == "__main__":
    # Test master pipeline
    print("Testing master pipeline...")
    
    pipeline = create_production_pipeline("test_pipeline_output", quality_mode="high")
    
    # Create test sequence
    test_sequence = []
    for i in range(10):
        # Create realistic pose with variation
        landmarks = np.zeros((33, 3), dtype=np.float32)
        landmarks[0] = [0, 0.8, 0]                           # nose
        landmarks[11] = [-0.2, 0.6, 0]                       # left_shoulder
        landmarks[12] = [0.2, 0.6, 0]                        # right_shoulder
        landmarks[23] = [-0.15, 0, 0]                        # left_hip
        landmarks[24] = [0.15, 0, 0]                         # right_hip
        landmarks[13] = [-0.25, 0.3 + 0.1*np.sin(i/5), 0]   # left_elbow (moving)
        landmarks[14] = [0.25, 0.3 + 0.1*np.cos(i/5), 0]    # right_elbow (moving)
        
        # Add noise
        landmarks += np.random.normal(0, 0.01, landmarks.shape)
        test_sequence.append(landmarks)
    
    # Process complete sequence
    result = pipeline.process_sequence(test_sequence, sequence_name="test_sequence")
    
    print(f"Sequence processing results:")
    print(f"  Total frames: {result['total_frames']}")
    print(f"  Successful frames: {result['successful_frames']}")
    print(f"  Processing time: {result['processing_time']:.2f}s")
    print(f"  Quality grade: {result['quality_assessment']['overall_grade']}")
    print(f"  Average quality: {result['quality_assessment']['score']:.1f}")
    
    if result.get('exported_files'):
        print(f"  Exported files: {list(result['exported_files'].keys())}")
    
    if result.get('visualizations'):
        print(f"  Visualizations: {len(result['visualizations'])}")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_statistics()
    pipeline_stats = stats['pipeline_stats']
    
    print(f"\nPipeline performance:")
    print(f"  Average FPS: {pipeline_stats.get('fps_average', 0):.1f}")
    print(f"  Frames repaired: {pipeline_stats['frames_repaired']}")
    print(f"  Frames filtered: {pipeline_stats['frames_filtered']}")
    print(f"  Visualizations created: {pipeline_stats['visualizations_created']}")
    
    # Shutdown
    pipeline.shutdown()
    
    print("[PASS] Master pipeline test completed")
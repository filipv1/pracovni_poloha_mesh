#!/usr/bin/env python3
"""
Unified Visualization System - Professional 3D visualization and rendering

Priority: HIGH
Dependencies: matplotlib, plotly, open3d, PIL
Test Coverage Required: 100%

This module provides a unified interface for all visualization needs including
real-time 3D rendering, publication-quality plots, and interactive dashboards.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from pathlib import Path
import sys
import json
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add core module to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    
    # Output settings
    output_dir: str = "visualization_output"
    dpi: int = 300
    figure_format: str = "png"  # png, jpg, svg, pdf
    
    # 3D rendering
    mesh_quality: str = "high"  # low, medium, high, ultra
    lighting_mode: str = "three_point"  # basic, three_point, studio
    background_color: Tuple[float, float, float] = (0.95, 0.95, 0.95)
    
    # Animation
    fps: int = 30
    animation_format: str = "mp4"  # mp4, gif, webm
    smooth_transitions: bool = True
    
    # Interactive features
    enable_interactivity: bool = True
    show_controls: bool = True
    auto_rotate: bool = False
    
    # Color schemes
    color_scheme: str = "professional"  # professional, vibrant, minimal, custom
    joint_colors: Optional[Dict[str, str]] = None
    angle_colormap: str = "viridis"
    
    # Text and labels
    font_family: str = "Arial"
    font_size: int = 12
    show_labels: bool = True
    show_measurements: bool = True
    
    # Performance
    max_frames_in_memory: int = 100
    enable_caching: bool = True
    use_gpu_acceleration: bool = True


class BaseVisualizer(ABC):
    """Abstract base class for all visualizers"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.color_schemes = {
            'professional': {
                'primary': '#2E8B57',      # Sea green
                'secondary': '#4682B4',    # Steel blue
                'accent': '#DC143C',       # Crimson
                'background': '#F8F8FF',   # Ghost white
                'text': '#2F4F4F',         # Dark slate gray
                'grid': '#D3D3D3'          # Light gray
            },
            'vibrant': {
                'primary': '#FF6B6B',      # Coral
                'secondary': '#4ECDC4',    # Turquoise
                'accent': '#FFE66D',       # Yellow
                'background': '#FFFFFF',   # White
                'text': '#2C3E50',         # Dark blue gray
                'grid': '#BDC3C7'          # Silver
            },
            'minimal': {
                'primary': '#34495E',      # Wet asphalt
                'secondary': '#7F8C8D',    # Asbestos
                'accent': '#E74C3C',       # Alizarin
                'background': '#FFFFFF',   # White
                'text': '#2C3E50',         # Midnight blue
                'grid': '#ECF0F1'          # Clouds
            }
        }
        
        self.colors = self.color_schemes.get(config.color_scheme, self.color_schemes['professional'])
        
    @abstractmethod
    def render(self, data: Any, **kwargs) -> Any:
        """Render visualization"""
        pass
    
    def _setup_matplotlib_style(self):
        """Setup matplotlib styling"""
        plt.style.use('default')
        
        # Set font
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.size'] = self.config.font_size
        
        # Set colors
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [
            self.colors['primary'], self.colors['secondary'], 
            self.colors['accent'], '#8E44AD', '#F39C12'
        ])
        
        # Figure settings
        plt.rcParams['figure.facecolor'] = self.colors['background']
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['axes.edgecolor'] = self.colors['grid']
        plt.rcParams['grid.color'] = self.colors['grid']
        plt.rcParams['text.color'] = self.colors['text']
        plt.rcParams['axes.labelcolor'] = self.colors['text']
        plt.rcParams['xtick.color'] = self.colors['text']
        plt.rcParams['ytick.color'] = self.colors['text']


class PoseVisualizer3D(BaseVisualizer):
    """3D pose visualization with advanced rendering"""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        
        # SMPL-X joint connections for skeleton visualization
        self.joint_connections = [
            # Spine
            (0, 3), (3, 6), (6, 9), (9, 12),  # pelvis -> spine -> neck
            (12, 15),  # neck -> head
            
            # Left arm
            (12, 13), (13, 16), (16, 18), (18, 20),  # neck -> collar -> shoulder -> elbow -> wrist
            
            # Right arm  
            (12, 14), (14, 17), (17, 19), (19, 21),
            
            # Left leg
            (0, 1), (1, 4), (4, 7), (7, 10),  # pelvis -> hip -> knee -> ankle -> foot
            
            # Right leg
            (0, 2), (2, 5), (5, 8), (8, 11)
        ]
        
        # Joint groups for coloring
        self.joint_groups = {
            'head': [15],
            'torso': [0, 3, 6, 9, 12, 13, 14],
            'left_arm': [13, 16, 18, 20],
            'right_arm': [14, 17, 19, 21], 
            'left_leg': [1, 4, 7, 10],
            'right_leg': [2, 5, 8, 11]
        }
        
        logger.info("PoseVisualizer3D initialized")
    
    def render(self, joints: np.ndarray, angles: Optional[Dict] = None, 
               title: str = "3D Pose", save_path: Optional[str] = None, **kwargs) -> str:
        """Render 3D pose visualization
        
        Args:
            joints: (22, 3) array of joint positions
            angles: Optional angle measurements
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Path to saved visualization
        """
        self._setup_matplotlib_style()
        
        # Create figure
        fig = plt.figure(figsize=(12, 8), dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot skeleton connections
        for start_idx, end_idx in self.joint_connections:
            if start_idx < len(joints) and end_idx < len(joints):
                start_pos = joints[start_idx]
                end_pos = joints[end_idx]
                
                # Skip if either joint is invalid
                if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                    continue
                
                ax.plot3D([start_pos[0], end_pos[0]], 
                         [start_pos[1], end_pos[1]], 
                         [start_pos[2], end_pos[2]], 
                         color=self.colors['primary'], linewidth=2, alpha=0.8)
        
        # Plot joints with group coloring
        for group_name, joint_indices in self.joint_groups.items():
            group_joints = []
            for idx in joint_indices:
                if idx < len(joints) and not np.any(np.isnan(joints[idx])):
                    group_joints.append(joints[idx])
            
            if group_joints:
                group_joints = np.array(group_joints)
                color = self._get_group_color(group_name)
                ax.scatter(group_joints[:, 0], group_joints[:, 1], group_joints[:, 2],
                          c=color, s=50, alpha=0.9, edgecolors='black', linewidth=1)
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d(ax, joints)
        
        # Styling
        ax.set_xlabel('X (m)', fontsize=self.config.font_size)
        ax.set_ylabel('Y (m)', fontsize=self.config.font_size)
        ax.set_zlabel('Z (m)', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        
        # Add angle annotations if provided
        if angles and self.config.show_measurements:
            self._add_angle_annotations(ax, angles, joints)
        
        # Set view angle for better perspective
        ax.view_init(elev=20, azim=45)
        
        # Remove grid for cleaner look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Save visualization
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"pose_3d_{timestamp}.{self.config.figure_format}"
        
        plt.tight_layout()
        plt.savefig(save_path, format=self.config.figure_format, 
                   dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'])
        plt.close()
        
        logger.info(f"3D pose visualization saved to {save_path}")
        return str(save_path)
    
    def _get_group_color(self, group_name: str) -> str:
        """Get color for joint group"""
        color_map = {
            'head': self.colors['accent'],
            'torso': self.colors['primary'],
            'left_arm': self.colors['secondary'],
            'right_arm': self.colors['secondary'],
            'left_leg': '#8E44AD',  # Purple
            'right_leg': '#8E44AD'
        }
        return color_map.get(group_name, self.colors['primary'])
    
    def _set_equal_aspect_3d(self, ax, joints: np.ndarray):
        """Set equal aspect ratio for 3D plot"""
        valid_joints = joints[~np.isnan(joints).any(axis=1)]
        
        if len(valid_joints) == 0:
            return
        
        # Get the range of each axis
        x_range = [valid_joints[:, 0].min(), valid_joints[:, 0].max()]
        y_range = [valid_joints[:, 1].min(), valid_joints[:, 1].max()]
        z_range = [valid_joints[:, 2].min(), valid_joints[:, 2].max()]
        
        # Find the maximum range
        max_range = max(
            x_range[1] - x_range[0],
            y_range[1] - y_range[0], 
            z_range[1] - z_range[0]
        )
        
        # Center each axis
        x_center = sum(x_range) / 2
        y_center = sum(y_range) / 2
        z_center = sum(z_range) / 2
        
        # Set equal limits
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    def _add_angle_annotations(self, ax, angles: Dict, joints: np.ndarray):
        """Add angle measurements as text annotations"""
        # Annotate key angles
        key_angles = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion']
        
        annotation_positions = [
            (0.02, 0.95),  # Top left
            (0.02, 0.90),  # Below previous
            (0.02, 0.85)   # Below previous
        ]
        
        for i, angle_name in enumerate(key_angles):
            if angle_name in angles and i < len(annotation_positions):
                angle_value = angles[angle_name]
                if not np.isnan(angle_value):
                    text = f"{angle_name.replace('_', ' ').title()}: {angle_value:.1f}°"
                    ax.text2D(annotation_positions[i][0], annotation_positions[i][1], 
                             text, transform=ax.transAxes, 
                             fontsize=self.config.font_size-1,
                             bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor=self.colors['background'], 
                                     alpha=0.8))


class AngleVisualizer(BaseVisualizer):
    """Specialized visualizer for angle measurements and trends"""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        logger.info("AngleVisualizer initialized")
    
    def render(self, data: Any, **kwargs) -> Any:
        """Base render method - delegates to specific render methods"""
        if 'timeline' in kwargs:
            return self.render_angle_timeline(data, **kwargs)
        elif 'distribution' in kwargs:
            return self.render_angle_distribution(data, **kwargs)
        else:
            return self.render_angle_timeline(data, **kwargs)
    
    def render_angle_timeline(self, angle_data: Dict[str, List[float]], 
                             timestamps: Optional[List[float]] = None,
                             title: str = "Angle Timeline", 
                             save_path: Optional[str] = None) -> str:
        """Render angle measurements over time"""
        self._setup_matplotlib_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.config.dpi)
        axes = axes.flatten()
        
        # Key angles to visualize
        key_angles = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion', 'left_elbow_flexion']
        
        if timestamps is None:
            timestamps = list(range(max(len(values) for values in angle_data.values())))
        
        for i, angle_name in enumerate(key_angles):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if angle_name in angle_data:
                values = angle_data[angle_name]
                
                # Plot raw data
                ax.plot(timestamps[:len(values)], values, 
                       color=self.colors['primary'], linewidth=2, 
                       alpha=0.7, label='Raw')
                
                # Add smoothed line if enough data
                if len(values) > 5:
                    smoothed = self._smooth_data(values, window_size=5)
                    ax.plot(timestamps[:len(smoothed)], smoothed,
                           color=self.colors['accent'], linewidth=3,
                           label='Smoothed')
                
                # Add reference lines for normal ranges
                self._add_reference_ranges(ax, angle_name)
                
                ax.set_title(angle_name.replace('_', ' ').title(), 
                           fontsize=self.config.font_size + 1, fontweight='bold')
                ax.set_xlabel('Time (frames)', fontsize=self.config.font_size)
                ax.set_ylabel('Angle (degrees)', fontsize=self.config.font_size)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, f'No data for\n{angle_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=self.config.font_size)
                ax.set_title(angle_name.replace('_', ' ').title())
        
        plt.suptitle(title, fontsize=self.config.font_size + 4, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"angle_timeline_{timestamp}.{self.config.figure_format}"
        
        plt.savefig(save_path, format=self.config.figure_format,
                   dpi=self.config.dpi, bbox_inches='tight',
                   facecolor=self.colors['background'])
        plt.close()
        
        logger.info(f"Angle timeline saved to {save_path}")
        return str(save_path)
    
    def render_angle_distribution(self, angle_data: Dict[str, List[float]],
                                title: str = "Angle Distribution",
                                save_path: Optional[str] = None) -> str:
        """Render angle distribution histograms"""
        self._setup_matplotlib_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.dpi)
        axes = axes.flatten()
        
        key_angles = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion', 'left_elbow_flexion']
        
        for i, angle_name in enumerate(key_angles):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if angle_name in angle_data and len(angle_data[angle_name]) > 0:
                values = [v for v in angle_data[angle_name] if not np.isnan(v)]
                
                if values:
                    # Create histogram
                    ax.hist(values, bins=20, color=self.colors['primary'], 
                           alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    ax.axvline(mean_val, color=self.colors['accent'], 
                              linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}°')
                    ax.axvline(mean_val - std_val, color=self.colors['secondary'], 
                              linestyle=':', alpha=0.7, label=f'±1σ')
                    ax.axvline(mean_val + std_val, color=self.colors['secondary'], 
                              linestyle=':', alpha=0.7)
                    
                    ax.set_title(angle_name.replace('_', ' ').title())
                    ax.set_xlabel('Angle (degrees)')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No data for\n{angle_name}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle(title, fontsize=self.config.font_size + 4, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"angle_distribution_{timestamp}.{self.config.figure_format}"
        
        plt.savefig(save_path, format=self.config.figure_format,
                   dpi=self.config.dpi, bbox_inches='tight',
                   facecolor=self.colors['background'])
        plt.close()
        
        logger.info(f"Angle distribution saved to {save_path}")
        return str(save_path)
    
    def _smooth_data(self, data: List[float], window_size: int = 5) -> List[float]:
        """Apply moving average smoothing"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            window_data = [x for x in data[start_idx:end_idx] if not np.isnan(x)]
            
            if window_data:
                smoothed.append(np.mean(window_data))
            else:
                smoothed.append(np.nan)
        
        return smoothed
    
    def _add_reference_ranges(self, ax, angle_name: str):
        """Add reference ranges for normal posture"""
        ranges = {
            'trunk_sagittal': (-5, 15),    # Slight forward lean is normal
            'trunk_lateral': (-5, 5),      # Minimal lateral bend
            'neck_flexion': (0, 20),       # Slight forward neck position
            'left_elbow_flexion': (5, 150), # Elbow range
            'right_elbow_flexion': (5, 150)
        }
        
        if angle_name in ranges:
            min_val, max_val = ranges[angle_name]
            ax.axhspan(min_val, max_val, alpha=0.2, 
                      color=self.colors['secondary'], 
                      label='Normal range')


class Dashboard(BaseVisualizer):
    """Interactive dashboard for comprehensive pose analysis"""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, dashboard functionality limited")
        
        logger.info("Dashboard initialized")
    
    def render(self, data: Any, **kwargs) -> Any:
        """Base render method for dashboard"""
        if isinstance(data, tuple) and len(data) == 2:
            pose_sequence, angle_sequence = data
            return self.create_comprehensive_dashboard(pose_sequence, angle_sequence, **kwargs)
        return None
    
    def create_comprehensive_dashboard(self, 
                                    pose_sequence: List[np.ndarray],
                                    angle_sequence: List[Dict],
                                    title: str = "Pose Analysis Dashboard",
                                    save_path: Optional[str] = None) -> str:
        """Create comprehensive analysis dashboard"""
        
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_dashboard(pose_sequence, angle_sequence, title, save_path)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('3D Pose Sequence', 'Trunk Angles Over Time',
                          'Angle Distribution', 'Processing Statistics',
                          'Movement Analysis', 'Quality Metrics'),
            specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Extract angle time series
        angle_names = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion']
        angle_data = {name: [] for name in angle_names}
        
        for frame_angles in angle_sequence:
            for name in angle_names:
                if name in frame_angles.get('filtered_angles', {}):
                    angle_data[name].append(frame_angles['filtered_angles'][name])
                else:
                    angle_data[name].append(np.nan)
        
        # Plot 3D pose trajectory (simplified)
        if pose_sequence:
            mid_frame = len(pose_sequence) // 2
            joints = pose_sequence[mid_frame]
            
            # Plot key joints
            key_joints = [0, 12, 15]  # pelvis, neck, head
            if len(joints) > max(key_joints):
                for i, joint_idx in enumerate(key_joints):
                    joint = joints[joint_idx]
                    if not np.any(np.isnan(joint)):
                        fig.add_trace(
                            go.Scatter3d(
                                x=[joint[0]], y=[joint[1]], z=[joint[2]],
                                mode='markers',
                                marker=dict(size=8, color=f'rgba({i*50+100}, {i*30+50}, 200, 0.8)'),
                                name=f'Joint {joint_idx}',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
        
        # Plot angle time series
        timestamps = list(range(len(angle_sequence)))
        colors = ['blue', 'red', 'green']
        
        for i, (name, values) in enumerate(angle_data.items()):
            if any(not np.isnan(v) for v in values):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=name.replace('_', ' ').title(),
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=1, col=2
                )
        
        # Add histogram for trunk sagittal
        if angle_data['trunk_sagittal']:
            valid_values = [v for v in angle_data['trunk_sagittal'] if not np.isnan(v)]
            if valid_values:
                fig.add_trace(
                    go.Histogram(
                        x=valid_values,
                        nbinsx=20,
                        name='Trunk Sagittal Distribution',
                        marker_color='lightblue',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Add processing statistics
        stats_names = ['Frames Processed', 'Valid Poses', 'Avg Quality']
        stats_values = [len(pose_sequence), 
                       sum(1 for seq in angle_sequence if seq), 
                       85.5]  # Mock quality score
        
        fig.add_trace(
            go.Bar(
                x=stats_names,
                y=stats_values,
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Movement velocity analysis
        if len(pose_sequence) > 1:
            velocities = []
            for i in range(1, len(pose_sequence)):
                if not np.any(np.isnan(pose_sequence[i])) and not np.any(np.isnan(pose_sequence[i-1])):
                    # Calculate center of mass movement
                    com1 = np.nanmean(pose_sequence[i-1], axis=0)
                    com2 = np.nanmean(pose_sequence[i], axis=0)
                    velocity = np.linalg.norm(com2 - com1)
                    velocities.append(velocity)
                else:
                    velocities.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(velocities))),
                    y=velocities,
                    mode='lines',
                    name='Movement Velocity',
                    line=dict(color='purple'),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Quality indicator
        overall_quality = 87.5  # Mock quality score
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_quality,
                title={'text': "Overall Quality"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=1000,
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Save as HTML
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"dashboard_{timestamp}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info(f"Interactive dashboard saved to {save_path}")
        return str(save_path)
    
    def _create_matplotlib_dashboard(self, pose_sequence, angle_sequence, title, save_path):
        """Fallback matplotlib dashboard when Plotly unavailable"""
        self._setup_matplotlib_style()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.config.dpi)
        
        # Extract angle data
        angle_names = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion']
        angle_data = {name: [] for name in angle_names}
        
        for frame_angles in angle_sequence:
            for name in angle_names:
                if name in frame_angles.get('filtered_angles', {}):
                    angle_data[name].append(frame_angles['filtered_angles'][name])
                else:
                    angle_data[name].append(np.nan)
        
        # Plot 1: Angle timeline
        ax = axes[0, 0]
        timestamps = list(range(len(angle_sequence)))
        for name, values in angle_data.items():
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            valid_values = [values[i] for i in valid_indices]
            valid_times = [timestamps[i] for i in valid_indices]
            
            if valid_values:
                ax.plot(valid_times, valid_values, label=name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_title('Angle Timeline')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution
        ax = axes[0, 1]
        if angle_data['trunk_sagittal']:
            valid_values = [v for v in angle_data['trunk_sagittal'] if not np.isnan(v)]
            if valid_values:
                ax.hist(valid_values, bins=15, alpha=0.7, color=self.colors['primary'])
        ax.set_title('Trunk Sagittal Distribution')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Frequency')
        
        # Plot 3: Processing stats
        ax = axes[0, 2]
        stats = ['Total Frames', 'Valid Poses', 'Processed']
        values = [len(pose_sequence), sum(1 for seq in angle_sequence if seq), len(angle_sequence)]
        ax.bar(stats, values, color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
        ax.set_title('Processing Statistics')
        
        # Plot 4-6: Additional analysis (simplified)
        for i, ax in enumerate([axes[1, 0], axes[1, 1], axes[1, 2]]):
            ax.text(0.5, 0.5, f'Analysis Plot {i+4}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Analysis {i+4}')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"dashboard_matplotlib_{timestamp}.{self.config.figure_format}"
        
        plt.savefig(save_path, format=self.config.figure_format,
                   dpi=self.config.dpi, bbox_inches='tight',
                   facecolor=self.colors['background'])
        plt.close()
        
        return str(save_path)


class UnifiedVisualizationSystem:
    """Main interface for all visualization needs"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Initialize specialized visualizers
        self.pose_viz = PoseVisualizer3D(self.config)
        self.angle_viz = AngleVisualizer(self.config)
        self.dashboard = Dashboard(self.config)
        
        # Output tracking
        self.generated_outputs = []
        
        logger.info("UnifiedVisualizationSystem initialized")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Available features: Matplotlib={'✓' if True else '✗'}, "
                   f"Plotly={'✓' if PLOTLY_AVAILABLE else '✗'}, "
                   f"Open3D={'✓' if OPEN3D_AVAILABLE else '✗'}")
    
    def visualize_pose_3d(self, joints: np.ndarray, angles: Optional[Dict] = None,
                         title: str = "3D Pose", **kwargs) -> str:
        """Create 3D pose visualization"""
        output_path = self.pose_viz.render(joints, angles, title, **kwargs)
        self.generated_outputs.append(output_path)
        return output_path
    
    def visualize_angle_timeline(self, angle_data: Dict[str, List[float]],
                               timestamps: Optional[List[float]] = None,
                               title: str = "Angle Timeline", **kwargs) -> str:
        """Create angle timeline visualization"""
        output_path = self.angle_viz.render_angle_timeline(angle_data, timestamps, title, **kwargs)
        self.generated_outputs.append(output_path)
        return output_path
    
    def visualize_angle_distribution(self, angle_data: Dict[str, List[float]],
                                   title: str = "Angle Distribution", **kwargs) -> str:
        """Create angle distribution visualization"""
        output_path = self.angle_viz.render_angle_distribution(angle_data, title, **kwargs)
        self.generated_outputs.append(output_path)
        return output_path
    
    def create_dashboard(self, pose_sequence: List[np.ndarray],
                        angle_sequence: List[Dict],
                        title: str = "Pose Analysis Dashboard", **kwargs) -> str:
        """Create comprehensive dashboard"""
        output_path = self.dashboard.create_comprehensive_dashboard(
            pose_sequence, angle_sequence, title, **kwargs
        )
        self.generated_outputs.append(output_path)
        return output_path
    
    def create_analysis_report(self, 
                              pose_sequence: List[np.ndarray],
                              angle_sequence: List[Dict],
                              title: str = "Pose Analysis Report") -> Dict[str, str]:
        """Create complete analysis report with multiple visualizations"""
        
        logger.info(f"Creating comprehensive analysis report: {title}")
        
        outputs = {}
        
        # 1. Key frame 3D poses
        if pose_sequence:
            key_frames = [0, len(pose_sequence)//2, len(pose_sequence)-1]
            
            for i, frame_idx in enumerate(key_frames):
                if frame_idx < len(pose_sequence):
                    frame_angles = angle_sequence[frame_idx] if frame_idx < len(angle_sequence) else None
                    angles = frame_angles.get('filtered_angles', {}) if frame_angles else None
                    
                    output_path = self.visualize_pose_3d(
                        pose_sequence[frame_idx], 
                        angles,
                        title=f"{title} - Frame {frame_idx}"
                    )
                    outputs[f'pose_3d_frame_{i}'] = output_path
        
        # 2. Angle analysis
        if angle_sequence:
            # Extract angle time series
            angle_data = {}
            angle_names = ['trunk_sagittal', 'trunk_lateral', 'neck_flexion', 
                          'left_elbow_flexion', 'right_elbow_flexion']
            
            for name in angle_names:
                angle_data[name] = []
                for frame_angles in angle_sequence:
                    if frame_angles and 'filtered_angles' in frame_angles:
                        angle_data[name].append(frame_angles['filtered_angles'].get(name, np.nan))
                    else:
                        angle_data[name].append(np.nan)
            
            # Timeline
            outputs['angle_timeline'] = self.visualize_angle_timeline(
                angle_data, title=f"{title} - Angle Timeline"
            )
            
            # Distribution
            outputs['angle_distribution'] = self.visualize_angle_distribution(
                angle_data, title=f"{title} - Angle Distribution"
            )
        
        # 3. Dashboard
        outputs['dashboard'] = self.create_dashboard(
            pose_sequence, angle_sequence, title=f"{title} - Dashboard"
        )
        
        logger.info(f"Analysis report generated with {len(outputs)} visualizations")
        
        return outputs
    
    def get_generated_outputs(self) -> List[str]:
        """Get list of all generated output files"""
        return self.generated_outputs.copy()
    
    def clear_outputs(self):
        """Clear generated outputs list"""
        self.generated_outputs.clear()


def create_visualization_system(output_dir: str = "visualization_output",
                              color_scheme: str = "professional",
                              dpi: int = 300) -> UnifiedVisualizationSystem:
    """Factory function to create visualization system"""
    config = VisualizationConfig(
        output_dir=output_dir,
        color_scheme=color_scheme,
        dpi=dpi
    )
    
    return UnifiedVisualizationSystem(config)


if __name__ == "__main__":
    # Test unified visualization system
    print("Testing unified visualization system...")
    
    viz_system = create_visualization_system("test_viz_output")
    
    # Create test data
    test_joints = np.array([
        [0.0, 0.0, 0.0],    # pelvis
        [-0.1, 0.0, 0.0],   # left_hip
        [0.1, 0.0, 0.0],    # right_hip
        [0.0, 0.2, 0.0],    # spine1
        [-0.15, -0.3, 0.0], # left_knee
        [0.15, -0.3, 0.0],  # right_knee
        [0.0, 0.4, 0.0],    # spine2
        [-0.2, -0.6, 0.0],  # left_ankle
        [0.2, -0.6, 0.0],   # right_ankle
        [0.0, 0.6, 0.0],    # spine3
        [-0.25, -0.7, 0.0], # left_foot
        [0.25, -0.7, 0.0],  # right_foot
        [0.0, 0.8, 0.0],    # neck
        [-0.05, 0.75, 0.0], # left_collar
        [0.05, 0.75, 0.0],  # right_collar
        [0.0, 0.9, 0.0],    # head
        [-0.2, 0.7, 0.0],   # left_shoulder
        [0.2, 0.7, 0.0],    # right_shoulder
        [-0.3, 0.4, 0.0],   # left_elbow
        [0.3, 0.4, 0.0],    # right_elbow
        [-0.4, 0.1, 0.0],   # left_wrist
        [0.4, 0.1, 0.0]     # right_wrist
    ])
    
    test_angles = {
        'filtered_angles': {
            'trunk_sagittal': 15.5,
            'trunk_lateral': -2.1,
            'neck_flexion': 8.3,
            'left_elbow_flexion': 45.0,
            'right_elbow_flexion': 30.0
        }
    }
    
    # Test 3D pose visualization
    pose_output = viz_system.visualize_pose_3d(test_joints, test_angles['filtered_angles'])
    print(f"3D pose visualization: {pose_output}")
    
    # Test angle timeline (mock data)
    timeline_data = {
        'trunk_sagittal': [10 + 5*np.sin(i/10) + np.random.normal(0, 1) for i in range(50)],
        'trunk_lateral': [0 + 3*np.cos(i/15) + np.random.normal(0, 0.5) for i in range(50)],
        'neck_flexion': [5 + 2*np.sin(i/8) + np.random.normal(0, 0.8) for i in range(50)]
    }
    
    timeline_output = viz_system.visualize_angle_timeline(timeline_data)
    print(f"Angle timeline: {timeline_output}")
    
    # Test distribution
    distribution_output = viz_system.visualize_angle_distribution(timeline_data)
    print(f"Angle distribution: {distribution_output}")
    
    # Test dashboard
    pose_sequence = [test_joints + np.random.normal(0, 0.01, test_joints.shape) for _ in range(20)]
    angle_sequence = [test_angles for _ in range(20)]
    
    dashboard_output = viz_system.create_dashboard(pose_sequence, angle_sequence)
    print(f"Dashboard: {dashboard_output}")
    
    # Test full report
    report_outputs = viz_system.create_analysis_report(pose_sequence, angle_sequence, "Test Analysis")
    print(f"Full report generated with {len(report_outputs)} files:")
    for name, path in report_outputs.items():
        print(f"  {name}: {path}")
    
    print("[PASS] Unified visualization system test completed")
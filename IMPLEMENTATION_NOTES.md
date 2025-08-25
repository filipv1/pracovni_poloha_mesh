# Implementation Notes for Ergonomics Measurement System

## Core Architecture
- **MediaPipe** → 33 3D body landmarks (pose_world_landmarks in meters)
- **SMPL-X fitting** → 10,475 vertices mesh
- **Angle calculation** → Ergonomic metrics (e.g., trunk angle > 60° for X seconds)

## Critical Issues to Solve

### 1. Missing Pose Detection Handling
**Problem**: When MediaPipe fails to detect pose, the system must not generate invalid mesh/angles.

**Solution Options**:
```python
if not results.pose_landmarks:
    # Option 1: Skip frame completely
    mesh_data = None
    angles = None
    
    # Option 2: Mark as "NO_POSE_DETECTED" 
    mesh_data = {
        'status': 'NO_POSE',
        'frame': frame_idx,
        'vertices': None,
        'angles': None
    }
    
    # Option 3: Use last valid pose (temporal interpolation)
    if last_valid_mesh:
        mesh_data = last_valid_mesh.copy()
        mesh_data['interpolated'] = True
```

**Ergonomic Analysis Impact**:
- Must exclude NO_POSE frames from angle calculations
- Report detection rate: `(detected_frames / total_frames) * 100`
- Flag segments with low detection rate as unreliable

### 2. Savitzky-Golay Filter for Angle Smoothing
**What it is**: Polynomial smoothing filter that preserves shape while removing noise.

**Why it's better than simple moving average**:
- Preserves peaks (important for detecting maximum angles)
- Less lag than moving average
- Can compute derivatives (angular velocity/acceleration)

**Implementation**:
```python
from scipy.signal import savgol_filter

# Window size should be odd, typically 5-15 frames
# Polynomial order typically 2-3
smoothed_angles = savgol_filter(
    raw_angles,
    window_length=9,  # at 25fps = 0.36 seconds
    polyorder=2
)
```

### 3. Occlusion Handling

**Detection**:
- MediaPipe provides `visibility` score (0-1) for each landmark
- Threshold: visibility < 0.5 = occluded

**Strategies**:
1. **Temporal prediction**: Use Kalman filter or LSTM to predict occluded joints
2. **Symmetry assumption**: Mirror visible side for occluded side
3. **Interpolation**: Linear/spline interpolation between last/next visible frames
4. **Confidence weighting**: Reduce influence of low-visibility joints in SMPL-X fitting

### 4. Multi-Person Tracking

**Pipeline**:
```
Video → YOLO (person detection) → ByteTrack (tracking) → Individual MediaPipe → SMPL-X
```

**Implementation with YOLOv11**:
```python
from ultralytics import YOLO

detector = YOLO('yolo11n.pt')
results = detector.track(frame, persist=True)

for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
    person_roi = frame[y1:y2, x1:x2]
    # Process each person separately
```

**Track management**:
- Maintain separate angle history per track_id
- Handle track loss/reacquisition
- Primary worker selection (largest bbox, most central, longest track)

### 5. 3D Scene Context

**Depth Estimation** (Depth Anything V2):
- Provides relative depth map
- Helps resolve depth ambiguity in poses
- Can detect furniture/workspace layout

**Benefits**:
- Better pose estimation when person interacts with objects
- Automatic workspace classification (standing desk, sitting desk, floor work)
- Distance-to-object measurements for ergonomics

**Limitations**:
- Relative depth only (not metric) without calibration
- Computational cost (~100ms per frame on GPU)

## Angle Calculation Pipeline

```python
class ErgonomicAnalyzer:
    def __init__(self):
        self.angle_history = []
        self.detection_history = []
        
    def process_frame(self, mesh_data):
        if mesh_data is None or mesh_data.get('status') == 'NO_POSE':
            # Mark frame as invalid
            self.detection_history.append(False)
            return None
            
        # Calculate angles from mesh joints
        angles = self.calculate_angles(mesh_data['joints'])
        
        # Apply Savitzky-Golay smoothing
        if len(self.angle_history) >= 9:
            smoothed = savgol_filter(
                self.angle_history[-9:] + [angles],
                window_length=9,
                polyorder=2
            )
            angles = smoothed[-1]
            
        self.angle_history.append(angles)
        self.detection_history.append(True)
        
        return angles
        
    def get_statistics(self):
        valid_frames = sum(self.detection_history)
        total_frames = len(self.detection_history)
        
        return {
            'detection_rate': valid_frames / total_frames,
            'trunk_>60_seconds': self.calculate_time_above_threshold(60),
            'continuous_segments': self.find_continuous_work_segments()
        }
```

## Production Checklist

- [ ] Implement pose detection validation
- [ ] Add frame interpolation for missing poses
- [ ] Integrate Savitzky-Golay filter
- [ ] Add confidence thresholds
- [ ] Implement multi-person tracking
- [ ] Add depth estimation (optional)
- [ ] Create ergonomic report generator
- [ ] Add visualization overlays for angles
- [ ] Implement CSV/JSON export for angle data
- [ ] Add calibration phase (T-pose or known reference)

## Testing Scenarios

1. **Occlusion test**: Person walks behind object
2. **Multi-person**: Two workers in frame
3. **Poor lighting**: Low contrast conditions
4. **Rapid movement**: Fast pose changes
5. **Long duration**: 8-hour workday video
6. **Various workspaces**: Standing, sitting, floor work

## Performance Targets

- Detection rate: > 90% for good lighting
- Processing speed: > 10 FPS on RTX 4090
- Angle accuracy: ± 5 degrees vs ground truth
- Temporal consistency: < 2° jitter after smoothing
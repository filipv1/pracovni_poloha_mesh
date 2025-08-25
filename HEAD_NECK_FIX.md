# Head and Neck Orientation Fix

## Problem
Current implementation uses **nose (landmark 0)** as head position, causing:
- Head appears too far forward
- Neck looks overly bent during forward lean
- Incorrect ergonomic angle measurements

## Solution: Use Ears for Better Head Estimation

### MediaPipe Facial Landmarks Available:
- **0**: Nose tip
- **7**: Left ear
- **8**: Right ear
- **9-10**: Mouth corners
- **1-6**: Eyes

### Anthropometric Facts:
- Ear canal to skull top: ~13cm
- Nose to ear center: ~10cm
- Head center of mass: Between ears and slightly forward

## Implementation

### Quick Integration (Monkey-patch):
```python
# In run_production_simple.py, add after imports:
from improved_head_estimation import integrate_with_pipeline

# In CompletePipeline.__init__, after creating converter:
integrate_with_pipeline(PreciseMediaPipeConverter)
```

### Direct Integration:
```python
# Modify PreciseMediaPipeConverter.convert_landmarks_to_smplx():

def convert_landmarks_to_smplx(self, mp_landmarks):
    # ... existing code ...
    
    # After basic joint mapping, improve head/neck:
    if mp_landmarks and len(mp_points) >= 13:
        # Get ear and nose positions
        nose = mp_points[0]
        left_ear = mp_points[7] if 7 < len(mp_points) else None
        right_ear = mp_points[8] if 8 < len(mp_points) else None
        
        if left_ear is not None and right_ear is not None:
            # Calculate ear center (skull base approximation)
            ear_center = (left_ear + right_ear) / 2
            
            # Vector from ears to nose (forward direction)
            forward = nose - ear_center
            forward_dist = np.linalg.norm(forward)
            
            # Calculate up vector
            ear_line = right_ear - left_ear
            up = np.cross(ear_line, forward)
            up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 1, 0])
            
            # Estimate skull top (1.3x forward distance above ears)
            skull_top = ear_center + up * (forward_dist * 1.3)
            
            # SMPL-X head joint (slightly forward of skull top)
            smplx_joints[15] = skull_top + forward * 0.2
            
            # Better neck position
            shoulder_center = (mp_points[11] + mp_points[12]) / 2
            smplx_joints[12] = shoulder_center + (ear_center - shoulder_center) * 0.3
            
            # Update confidence
            joint_weights[15] = 0.85  # Higher confidence with ears
            joint_weights[12] = 0.85
```

## Benefits for Ergonomics

### Before (Nose-based):
```
Neck angle = angle(shoulders → nose)
Problem: Overestimates forward head posture
```

### After (Ear-based):
```
Neck angle = angle(shoulders → skull_center)
Correct: Matches actual cervical spine angle
```

## Expected Improvements:
1. **More accurate neck flexion angles** (5-10° difference)
2. **Better head tilt detection**
3. **Correct forward head posture measurement**
4. **Reduced jitter in head position**

## Testing the Fix:
```python
# Test with person looking down
# Before: Neck appears bent 60°
# After: Neck shows correct 40° (actual angle)

# Test with person looking straight
# Before: Slight forward lean detected (false positive)
# After: Neutral position correctly identified
```

## Fallback Strategy:
If ears not visible (profile view):
1. Use visible ear + nose to estimate other ear
2. If no ears visible: Fall back to nose with reduced confidence
3. Mark frame as "LIMITED_HEAD_TRACKING" in ergonomic report

## Integration Checklist:
- [ ] Add improved_head_estimation.py to project
- [ ] Import in run_production_simple.py
- [ ] Call integrate_with_pipeline() after converter creation
- [ ] Test with video showing various head positions
- [ ] Verify ergonomic angles are more accurate
- [ ] Document confidence thresholds for head tracking
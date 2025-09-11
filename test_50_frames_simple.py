#!/usr/bin/env python3
"""
Test simple animation with 50 frames
"""

import subprocess
from pathlib import Path

print("\nTEST: 50 FRAMES WITH SIMPLE ANIMATION")
print("="*60)

# Test with simple animation renderer
blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

cmd = [
    blender_exe,
    "--background",
    "--python", "blender_simple_animation.py",
    "--",
    "--obj_dir", "blender_export_skin_5614",
    "--output", "test_50_frames_animated.mp4",
    "--fps", "25",
    "--resolution_x", "640",
    "--resolution_y", "480",
    "--samples", "16",
    "--max_frames", "50"  # Limit to 50 frames for testing
]

print("Running Blender with simple animation...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.stdout:
    print("\n[OUTPUT]:")
    # Show last part of output
    lines = result.stdout.split('\n')
    for line in lines[-30:]:
        if line.strip():
            print(line)

if result.stderr:
    print("\n[ERRORS]:")
    print(result.stderr[:1000])

# Check for output
test_file = Path("test_50_frames_animated.mp4")
if test_file.exists():
    size_mb = test_file.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Video created: {test_file.name} ({size_mb:.2f} MB)")
    if size_mb < 0.5:
        print("[WARNING] File size is suspiciously small!")
else:
    print("\n[ERROR] Video file not found")

print("="*60)
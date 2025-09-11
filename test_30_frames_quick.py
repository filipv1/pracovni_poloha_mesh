#!/usr/bin/env python3
"""
Quick test with 30 frames only
"""

import subprocess
from pathlib import Path
import shutil

print("\nQUICK TEST: 30 FRAMES")
print("="*60)

# Create temp directory with 30 frames
test_dir = Path("test_30_frames_dir")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir()

export_dir = Path("blender_export_skin_5614")

# Copy first 30 frames
for i in range(30):
    # Copy mesh
    src = export_dir / f"frame_{i:04d}.obj"
    if src.exists():
        dst = test_dir / f"frame_{i:04d}.obj"
        shutil.copy2(src, dst)
    
    # Copy trunk skin
    src = export_dir / f"trunk_skin_{i:04d}.obj"
    if src.exists():
        dst = test_dir / f"trunk_skin_{i:04d}.obj"
        shutil.copy2(src, dst)

print(f"Copied 30 frames to {test_dir}")

# Run with no vectors renderer
blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

cmd = [
    blender_exe,
    "--background",
    "--python", "blender_simple_visibility.py",
    "--",
    "--obj_dir", str(test_dir),
    "--output", "test_30_frames.mp4",
    "--fps", "25",
    "--resolution_x", "1280",
    "--resolution_y", "720",
    "--samples", "16"  # Lower samples for speed
]

print("Running Blender (mesh only, no vectors)...")
result = subprocess.run(cmd, capture_output=True, text=True)

# Show last part of output
if result.stdout:
    lines = result.stdout.split('\n')
    for line in lines[-20:]:
        if line.strip() and any(x in line for x in ["frames", "Animation", "Rendering", "SUCCESS", "ERROR"]):
            print(line)

if result.stderr:
    print("\n[STDERR]:")
    print(result.stderr[:500])

# Check output
test_file = Path("test_30_frames.mp4")
if test_file.exists():
    size_mb = test_file.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Video created: {test_file.name} ({size_mb:.2f} MB)")
else:
    print("\n[ERROR] No video created")

# Clean up
if test_dir.exists():
    shutil.rmtree(test_dir)

print("="*60)
#!/usr/bin/env python3
"""
FINAL TEST with 30 frames only
"""

import subprocess
from pathlib import Path
import shutil

print("\nFINAL TEST: 30 FRAMES")
print("="*60)

# Create temp directory with 30 frames
test_dir = Path("final_test_30_dir")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir()

export_dir = Path("blender_export_skin_5614")

# Copy first 30 frames (mesh + trunk)
for i in range(30):
    # Copy mesh
    src = export_dir / f"frame_{i:04d}.obj"
    if src.exists():
        shutil.copy2(src, test_dir / f"frame_{i:04d}.obj")
    
    # Copy trunk skin
    src = export_dir / f"trunk_skin_{i:04d}.obj"
    if src.exists():
        shutil.copy2(src, test_dir / f"trunk_skin_{i:04d}.obj")

print(f"Copied 30 frames to {test_dir}")

# Run with final solution
blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

cmd = [
    blender_exe,
    "--background",
    "--python", "blender_mesh_only_fixed.py",
    "--",
    "--obj_dir", str(test_dir),
    "--output", "FINAL_30_FRAMES.mp4",
    "--fps", "25",
    "--resolution_x", "1280",
    "--resolution_y", "720",
    "--samples", "32"  # Medium quality for speed
]

print("Running FINAL SOLUTION renderer...")
print("Resolution: 1280x720")
print("FPS: 25")
print("Frames: 30")

result = subprocess.run(cmd, capture_output=True, text=True)

# Show important output
if result.stdout:
    lines = result.stdout.split('\n')
    for line in lines[-30:]:
        if line.strip() and any(x in line for x in ["Imported", "Animation", "Frame 1 check", "SUCCESS", "ERROR"]):
            print(line)

if result.stderr:
    print("\n[STDERR]:")
    print(result.stderr[:500])

# Check output
test_file = Path("FINAL_30_FRAMES.mp4")
if test_file.exists():
    size_mb = test_file.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Video created: {test_file.name} ({size_mb:.2f} MB)")
else:
    print("\n[ERROR] No video created")

# Clean up
if test_dir.exists():
    shutil.rmtree(test_dir)

print("="*60)
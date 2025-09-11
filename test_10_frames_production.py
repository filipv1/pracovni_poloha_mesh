#!/usr/bin/env python3
"""
Test production renderer with just 10 frames to debug issues
"""

import subprocess
from pathlib import Path
import shutil

print("\nTEST: 10 FRAMES PRODUCTION RENDERER")
print("="*60)

# Check if export directory exists
export_dir = Path("blender_export_skin_5614")
if not export_dir.exists():
    print("[ERROR] Export directory not found")
    exit(1)

# Create test directory with 10 frames
test_dir = Path("test_10_frames_dir")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir()

# Copy first 10 frames
frame_count = 0
for i in range(10):
    # Copy frame mesh
    src = export_dir / f"frame_{i:04d}.obj"
    if src.exists():
        dst = test_dir / f"frame_{i:04d}.obj"
        shutil.copy2(src, dst)
        frame_count += 1
    
    # Copy trunk
    for trunk_name in [f"trunk_{i:04d}.obj", f"trunk_skin_{i:04d}.obj"]:
        src = export_dir / trunk_name
        if src.exists():
            dst = test_dir / trunk_name
            shutil.copy2(src, dst)

print(f"Copied {frame_count} frames to test directory")

# Test with production renderer
print("\nTesting production renderer...")
blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

# Use absolute paths
test_output = Path("test_10_frames.mp4").resolve()

cmd = [
    blender_exe,
    "--background",
    "--python", "blender_headless_render_production.py",
    "--",
    "--obj_dir", str(test_dir.resolve()),
    "--output", str(test_output),
    "--fps", "25",
    "--resolution_x", "640",
    "--resolution_y", "480",
    "--samples", "16"
]

print("Running Blender...")
result = subprocess.run(cmd, capture_output=True, text=True)

print("\n[BLENDER OUTPUT]:")
print(result.stdout)

if result.stderr:
    print("\n[BLENDER ERRORS]:")
    print(result.stderr)

print("\n[RETURN CODE]:", result.returncode)

# Check for output
possible_files = list(Path(".").glob("test_10_frames*.mp4"))
if possible_files:
    for f in possible_files:
        size_kb = f.stat().st_size / 1024
        print(f"[OK] Found: {f.name} ({size_kb:.1f} KB)")
else:
    print("[ERROR] No output file found")

# Clean up test directory
if test_dir.exists():
    shutil.rmtree(test_dir)

print("="*60)
#!/usr/bin/env python3
"""
Test 20 frames with no blinking fix
"""

import subprocess
from pathlib import Path

print("\nTEST: 20 FRAMES NO BLINKING")
print("="*60)

blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

# Test with 20 frames at medium quality
cmd = [
    blender_exe,
    "--background",
    "--python", "blender_final_fixed.py",
    "--",
    "--obj_dir", "blender_export_skin_5614",
    "--output", "test_20_noblink.mp4",
    "--fps", "25",
    "--resolution_x", "1280",
    "--resolution_y", "720",
    "--samples", "32"
]

print("Running Blender with fixed animation...")
print("Resolution: 1280x720")
print("FPS: 25")
print("Limiting to first 20 frames for quick test")

# Create temporary script to limit frames
temp_script = Path("blender_20_frames_test.py")
with open("blender_final_fixed.py", "r") as f:
    content = f.read()
    # Limit to 20 frames
    content = content.replace("for frame_idx in frame_numbers:", "for frame_idx in frame_numbers[:20]:")
    content = content.replace("scene.frame_end = len(frame_numbers)", "scene.frame_end = min(20, len(frame_numbers))")
    
temp_script.write_text(content)

# Update command to use temp script
cmd[4] = str(temp_script)

result = subprocess.run(cmd, capture_output=True, text=True)

if result.stdout:
    lines = result.stdout.split('\n')
    # Show important lines
    for line in lines:
        if any(x in line for x in ["Frame range", "Import complete", "Animation ready", "SUCCESS", "ERROR"]):
            print(line)

if result.stderr:
    print("\n[ERRORS]:")
    print(result.stderr[:500])

# Check output
test_file = Path("test_20_noblink.mp4")
if test_file.exists():
    size_mb = test_file.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Video created: {test_file.name} ({size_mb:.2f} MB)")
else:
    print("\n[ERROR] Video file not found")

# Clean up temp script
if temp_script.exists():
    temp_script.unlink()

print("="*60)
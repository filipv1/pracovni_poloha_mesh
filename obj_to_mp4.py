#!/usr/bin/env python3
"""
Simple OBJ to MP4 converter
Renders existing OBJ sequences to video using Blender
"""

import subprocess
import argparse
from pathlib import Path

def render_obj_to_mp4(obj_dir="blender_export_skin_5614", 
                      output="output.mp4",
                      fps=25,
                      resolution=(1280, 720),
                      quality="medium"):
    """
    Render OBJ sequence to MP4 using Blender
    """
    
    # Quality presets
    quality_settings = {
        "low": {"samples": 16, "resolution": (640, 480)},
        "medium": {"samples": 32, "resolution": (1280, 720)},
        "high": {"samples": 64, "resolution": (1920, 1080)}
    }
    
    if quality in quality_settings:
        settings = quality_settings[quality]
        resolution = settings["resolution"]
        samples = settings["samples"]
    else:
        samples = 32
    
    # Check OBJ directory
    obj_path = Path(obj_dir)
    if not obj_path.exists():
        print(f"[ERROR] OBJ directory not found: {obj_dir}")
        return False
    
    obj_count = len(list(obj_path.glob("*.obj")))
    print(f"Found {obj_count} OBJ files in {obj_dir}")
    
    # Blender command - use FINAL FIXED script
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
    
    # Use MESH ONLY script - NO VECTORS - NO BLINKING
    render_script = "blender_mesh_only_fixed.py"
    print(f"[INFO] Using MESH-ONLY renderer (NO VECTORS) for {obj_count} frames")
    
    cmd = [
        blender_exe,
        "--background",
        "--python", render_script,
        "--",
        "--obj_dir", obj_dir,
        "--output", output,
        "--fps", str(fps),
        "--resolution_x", str(resolution[0]),
        "--resolution_y", str(resolution[1]),
        "--samples", str(samples)
    ]
    
    print(f"\nRendering with Blender...")
    print(f"  Output: {output}")
    print(f"  Quality: {quality}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  FPS: {fps}")
    print(f"  Samples: {samples}")
    
    # Run Blender
    print("\nRunning Blender (this may take a few minutes)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Always show Blender output for debugging
    print("\n[BLENDER OUTPUT]:")
    if result.stdout:
        print(result.stdout[-3000:])  # Last 3000 chars
    if result.stderr:
        print("\n[BLENDER ERRORS]:")
        print(result.stderr[-2000:])
    
    if result.returncode == 0:
        # Find actual output (with frame numbers)
        output_base = Path(output).stem
        possible_files = list(Path(".").glob(f"{output_base}*.mp4"))
        
        if possible_files:
            actual_file = possible_files[0]
            size_mb = actual_file.stat().st_size / (1024 * 1024)
            print(f"\n[SUCCESS] Video created: {actual_file.name} ({size_mb:.1f} MB)")
            
            # Rename if needed
            if actual_file.name != output:
                actual_file.rename(output)
                print(f"[INFO] Renamed to: {output}")
            
            return True
        else:
            print(f"[ERROR] Output file not found")
            return False
    else:
        print(f"[ERROR] Render failed")
        if result.stderr:
            print(result.stderr[:500])
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Render OBJ sequence to MP4 video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Default: blender_export_skin_5614 -> output.mp4
  %(prog)s -o my_video.mp4          # Custom output name
  %(prog)s -q high                  # High quality (1920x1080)
  %(prog)s -d custom_obj_dir        # Custom OBJ directory
  %(prog)s --fps 30 -q low          # 30 FPS, low quality
        """
    )
    
    parser.add_argument("-d", "--dir", default="blender_export_skin_5614",
                       help="Directory with OBJ files")
    parser.add_argument("-o", "--output", default="output.mp4",
                       help="Output MP4 filename")
    parser.add_argument("-q", "--quality", choices=["low", "medium", "high"],
                       default="medium", help="Quality preset")
    parser.add_argument("--fps", type=int, default=25,
                       help="Frames per second")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("OBJ TO MP4 CONVERTER")
    print("="*60)
    
    success = render_obj_to_mp4(
        obj_dir=args.dir,
        output=args.output,
        fps=args.fps,
        quality=args.quality
    )
    
    if success:
        print("\nDONE! Video is ready.")
    else:
        print("\nFAILED! Check errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
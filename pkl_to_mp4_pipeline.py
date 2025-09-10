#!/usr/bin/env python3
"""
PKL to MP4 Pipeline
Converts PKL mesh data to MP4 video using Blender headless rendering

Pipeline:
1. Load PKL file
2. Export OBJ sequences using export_all_vectors_skin_to_blender.py
3. Render animation in Blender headless
4. Output MP4 video
"""

import subprocess
import sys
import time
import shutil
import argparse
import pickle
from pathlib import Path

def load_pkl_metadata(pkl_path):
    """Load PKL file and extract metadata"""
    print(f"[1/4] Loading PKL file: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # Handle both old and new PKL format
    if isinstance(pkl_data, dict) and 'mesh_sequence' in pkl_data:
        meshes = pkl_data['mesh_sequence']
        metadata = pkl_data.get('metadata', {})
        fps = metadata.get('fps', 30.0)
        print(f"  [OK] New PKL format with metadata")
        print(f"  [OK] FPS: {fps}")
        print(f"  [OK] Frames: {len(meshes)}")
    else:
        meshes = pkl_data
        fps = 30.0
        print(f"  [OK] Old PKL format")
        print(f"  [OK] Using default FPS: {fps}")
        print(f"  [OK] Frames: {len(meshes)}")
    
    return len(meshes), fps

def export_obj_sequences(pkl_path, output_dir="blender_export_skin_5614"):
    """Export OBJ sequences from PKL using existing script"""
    print(f"[2/4] Exporting OBJ sequences...")
    print(f"  • This may take 1-2 minutes for 500 frames...")
    print(f"  • Exporting to: {output_dir}/")
    
    # Call the export function directly instead of subprocess
    try:
        # Import the export function
        from export_all_vectors_skin_to_blender import export_all_vectors_skin_to_blender
        
        # Call it directly with parameters
        result_dir = export_all_vectors_skin_to_blender(
            pkl_file=str(pkl_path),
            output_dir=output_dir,
            lumbar_vertex=5614  # Default vertex
        )
        
        if not result_dir:
            print(f"  [ERROR] Export failed")
            return False
            
        print(f"  [OK] OBJ files exported to {output_dir}/")
        
        # Verify export
        export_path = Path(output_dir)
        if not export_path.exists():
            print(f"  [ERROR] Export directory not found: {export_path}")
            return False
            
        obj_files = list(export_path.glob("*.obj"))
        print(f"  [OK] Found {len(obj_files)} OBJ files")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Export error: {e}")
        return False

def render_in_blender(obj_dir, output_mp4, fps, quality="medium"):
    """Render animation in Blender headless"""
    print(f"[3/4] Rendering animation in Blender...")
    print(f"  • Output: {output_mp4}")
    print(f"  • Quality: {quality}")
    print(f"  • FPS: {fps}")
    
    # Quality presets
    quality_settings = {
        "low": {"resolution": (1280, 720), "samples": 32},
        "medium": {"resolution": (1920, 1080), "samples": 64},
        "high": {"resolution": (2560, 1440), "samples": 128}
    }
    
    settings = quality_settings.get(quality, quality_settings["medium"])
    
    # Check if Blender is available
    blender_exe = shutil.which("blender")
    if not blender_exe:
        # Try common Windows paths (including user installations)
        common_paths = [
            # Program Files locations
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.4\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
            # User installations
            Path.home() / "Downloads" / "blender-4.2.0-windows-x64" / "blender.exe",
            Path.home() / "Downloads" / "blender-4.1.0-windows-x64" / "blender.exe",
            Path.home() / "Downloads" / "blender-4.0.0-windows-x64" / "blender.exe",
            Path.home() / "Desktop" / "Blender" / "blender.exe",
            # Portable installations
            r"D:\Blender\blender.exe",
            r"E:\Blender\blender.exe",
            r"C:\Blender\blender.exe",
            # Steam installation
            r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
        ]
        for path in common_paths:
            if isinstance(path, str):
                path = Path(path)
            if path.exists():
                blender_exe = str(path)
                break
    
    if not blender_exe:
        print(f"  [ERROR] Blender not found in common locations")
        print(f"  • Download Blender from: https://www.blender.org/download/")
        print(f"  • Or set BLENDER_PATH environment variable")
        print(f"  • Or create blender_path.txt with path to blender.exe")
        
        # Check for custom path file
        custom_path_file = Path("blender_path.txt")
        if custom_path_file.exists():
            custom_path = custom_path_file.read_text().strip()
            if Path(custom_path).exists():
                blender_exe = custom_path
                print(f"  [OK] Found Blender from blender_path.txt: {blender_exe}")
            else:
                print(f"  [ERROR] Invalid path in blender_path.txt: {custom_path}")
                return False
        else:
            return False
    
    print(f"  • Using Blender: {blender_exe}")
    
    # Build Blender command
    cmd = [
        blender_exe,
        "--background",  # Run without UI
        "--python", "blender_headless_render.py",
        "--",  # Separator for script arguments
        "--obj_dir", obj_dir,
        "--output", output_mp4,
        "--fps", str(fps),
        "--resolution_x", str(settings["resolution"][0]),
        "--resolution_y", str(settings["resolution"][1]),
        "--samples", str(settings["samples"])
    ]
    
    try:
        # Run Blender
        print(f"  • Starting render process...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  [ERROR] Render failed after {elapsed:.1f}s")
            if result.stderr:
                print(f"  Stderr: {result.stderr[:1000]}")  # First 1000 chars
            if result.stdout:
                print(f"  Stdout: {result.stdout[-1000:]}")  # Last 1000 chars
            return False
        
        print(f"  [OK] Render completed in {elapsed:.1f}s")
        
        # Verify output - Blender adds frame numbers to filename
        # Try to find the actual output file
        output_base = Path(output_mp4).stem
        output_dir = Path(output_mp4).parent
        
        # Look for files with frame numbers
        possible_outputs = [
            output_mp4,  # Original name
            output_dir / f"{output_base}0001-9999.mp4",  # With frame range
            output_dir / f"{output_base}001-999.mp4",
            output_dir / f"{output_base}0001-0499.mp4",  # For 499 frames
        ]
        
        # Also search with glob pattern
        glob_matches = list(output_dir.glob(f"{output_base}*.mp4"))
        
        actual_output = None
        for possible in possible_outputs:
            if possible.exists():
                actual_output = possible
                break
        
        if not actual_output and glob_matches:
            actual_output = glob_matches[0]
        
        if not actual_output:
            print(f"  [ERROR] Output file not created: {output_mp4}")
            print(f"  Searched for: {output_base}*.mp4")
            return False
        
        # Rename to expected name if different
        if actual_output != Path(output_mp4):
            print(f"  • Found output: {actual_output.name}")
            print(f"  • Renaming to: {Path(output_mp4).name}")
            actual_output.rename(output_mp4)
            
        file_size = Path(output_mp4).stat().st_size / (1024 * 1024)  # MB
        print(f"  [OK] Output file size: {file_size:.1f} MB")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Render timeout after 30 minutes")
        return False
    except Exception as e:
        print(f"  [ERROR] Render error: {e}")
        return False

def cleanup_obj_files(obj_dir, do_cleanup=False):
    """Optionally cleanup temporary OBJ files"""
    if not do_cleanup:
        return
        
    print(f"[4/4] Cleaning up temporary files...")
    
    try:
        obj_path = Path(obj_dir)
        if obj_path.exists():
            # Count files before deletion
            obj_files = list(obj_path.glob("*.obj"))
            num_files = len(obj_files)
            
            # Delete directory
            shutil.rmtree(obj_path)
            print(f"  [OK] Deleted {num_files} OBJ files")
    except Exception as e:
        print(f"  [ERROR] Cleanup failed: {e}")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description="Convert PKL mesh data to MP4 video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s arm_meshes.pkl output.mp4
  %(prog)s data.pkl video.mp4 --quality high
  %(prog)s data.pkl video.mp4 --cleanup
        """
    )
    
    parser.add_argument("pkl_file", help="Input PKL file with mesh data")
    parser.add_argument("output_mp4", help="Output MP4 video file")
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Render quality preset (default: medium)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete temporary OBJ files after rendering"
    )
    parser.add_argument(
        "--obj_dir",
        default="blender_export_skin_5614",
        help="Directory for OBJ export (default: blender_export_skin_5614)"
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*60)
    print("PKL TO MP4 PIPELINE")
    print("="*60)
    
    # Validate input
    pkl_path = Path(args.pkl_file)
    if not pkl_path.exists():
        print(f"ERROR: PKL file not found: {pkl_path}")
        return 1
    
    # Load metadata
    try:
        num_frames, fps = load_pkl_metadata(pkl_path)
    except Exception as e:
        print(f"ERROR: Failed to load PKL: {e}")
        return 1
    
    # Export OBJs
    if not export_obj_sequences(pkl_path, args.obj_dir):
        print("\nERROR: OBJ export failed")
        return 1
    
    # Render in Blender
    if not render_in_blender(args.obj_dir, args.output_mp4, fps, args.quality):
        print("\nERROR: Blender render failed")
        return 1
    
    # Optional cleanup
    cleanup_obj_files(args.obj_dir, args.cleanup)
    
    # Success
    print("\n" + "="*60)
    print("SUCCESS!")
    print(f"Video saved to: {args.output_mp4}")
    print(f"Duration: {num_frames/fps:.1f} seconds at {fps} FPS")
    print("="*60 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
Compress video before uploading to RunPod
Reduces file size while maintaining quality
"""

import subprocess
import tempfile
import base64
from pathlib import Path

def compress_video(video_path, target_size_mb=5):
    """
    Compress video to target size using ffmpeg
    """
    input_path = Path(video_path)
    output_path = input_path.parent / f"{input_path.stem}_compressed.mp4"

    # Calculate bitrate for target size
    # Get video duration
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]

    try:
        duration = float(subprocess.check_output(cmd).decode().strip())
    except:
        duration = 30  # Default 30 seconds if can't determine

    # Calculate target bitrate (in kbps)
    target_bitrate = int((target_size_mb * 8192) / duration)

    print(f"Compressing video to ~{target_size_mb}MB")
    print(f"Duration: {duration:.1f}s, Target bitrate: {target_bitrate}kbps")

    # Compress video
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-b:v', f'{target_bitrate}k',
        '-maxrate', f'{target_bitrate * 1.5}k',
        '-bufsize', f'{target_bitrate * 2}k',
        '-c:a', 'aac',
        '-b:a', '64k',
        '-movflags', '+faststart',
        '-y',  # Overwrite output
        str(output_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # Check final size
    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Compressed: {input_path.stat().st_size/(1024*1024):.1f}MB -> {final_size_mb:.1f}MB")

    return str(output_path)

def video_to_base64(video_path):
    """Convert video file to base64"""
    with open(video_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compress_video.py <video_file>")
        sys.exit(1)

    input_video = sys.argv[1]

    # Compress video
    compressed = compress_video(input_video, target_size_mb=5)

    # Convert to base64
    base64_data = video_to_base64(compressed)

    print(f"\nBase64 size: {len(base64_data) / (1024*1024):.2f}MB")
    print("Ready for upload to RunPod!")

    # Save base64 to file
    output_file = Path(input_video).stem + "_base64.txt"
    with open(output_file, 'w') as f:
        f.write(base64_data)
    print(f"Base64 saved to: {output_file}")
# PKL to MP4 Pipeline

Automated pipeline for converting PKL mesh data to MP4 video using Blender.

## Requirements

- Python 3.9+ (with conda environment `trunk_analysis` activated)
- Blender 3.0+ (must be installed and accessible)
- ~2GB free disk space for temporary OBJ files

## Installation

1. Make sure Blender is installed:
   - Download from https://www.blender.org/download/
   - Default installation path is fine
   - The script will auto-detect Blender location

2. Activate conda environment:
   ```bash
   conda activate trunk_analysis
   ```

## Usage

### Basic Usage
```bash
python pkl_to_mp4_pipeline.py input.pkl output.mp4
```

### With Options
```bash
# High quality render
python pkl_to_mp4_pipeline.py input.pkl output.mp4 --quality high

# Clean up temporary files after rendering
python pkl_to_mp4_pipeline.py input.pkl output.mp4 --cleanup

# Custom OBJ export directory
python pkl_to_mp4_pipeline.py input.pkl output.mp4 --obj_dir custom_export_dir
```

### Quick Test
```bash
# Windows batch script
run_pkl_to_mp4.bat

# Or manually with test file
python pkl_to_mp4_pipeline.py fpsmeshes.pkl test.mp4 --quality low
```

## Quality Presets

| Preset | Resolution | FPS | Render Time (500 frames) |
|--------|-----------|-----|-------------------------|
| low    | 1280x720  | 15  | ~5 minutes              |
| medium | 1920x1080 | 30  | ~10 minutes             |
| high   | 2560x1440 | 30  | ~20 minutes             |

## Pipeline Stages

1. **Load PKL** - Reads mesh data and metadata (FPS)
2. **Export OBJs** - Creates OBJ files for each frame (~1GB for 500 frames)
3. **Blender Render** - Imports OBJs and renders animation
4. **Output MP4** - Final video with all vectors visualized

## Troubleshooting

### "Blender not found"
- Install Blender from https://www.blender.org/
- Or add Blender to system PATH
- Or edit `pkl_to_mp4_pipeline.py` to add your Blender path

### "Export failed"
- Check that `export_all_vectors_skin_to_blender.py` exists
- Ensure PKL file is valid and contains mesh data
- Check disk space (need ~2GB free)

### "Render failed"
- Check Blender console output for errors
- Try lower quality setting
- Ensure enough RAM (8GB+ recommended)

### Long render times
- Use `--quality low` for testing
- Close other applications to free RAM
- Consider using fewer frames for preview

## Output

The MP4 video will contain:
- 3D mesh (semi-transparent gray)
- Trunk vector (red)
- Neck vector (blue)
- Left arm vector (green)
- Right arm vector (yellow)

## Advanced Usage

### Batch Processing
```bash
for %%f in (*.pkl) do (
    python pkl_to_mp4_pipeline.py "%%f" "%%~nf.mp4" --quality medium
)
```

### Custom Blender Settings
Edit `blender_headless_render.py` to customize:
- Camera angle
- Lighting
- Materials
- Background

## Files Created

- `pkl_to_mp4_pipeline.py` - Main orchestrator script
- `blender_headless_render.py` - Blender rendering script
- `blender_export_skin_5614/` - Temporary OBJ files (can be deleted with --cleanup)
- `output.mp4` - Final rendered video
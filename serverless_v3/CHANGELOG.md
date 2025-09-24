# Changelog

## Version 4.0.0 (2025-01-23) - EXTENDED PIPELINE

### Major Changes
- Added complete 4-step processing pipeline
- Multiple output files support
- Extended frontend for all file types

### New Features
1. **CSV Generation**: Angles data from skin-based calculations
2. **Excel Analysis**: Ergonomic time analysis with Czech labels
3. **4 Videos Generation**: Original, MediaPipe, 3D mesh, and overlay
4. **Multiple Downloads**: Dynamic buttons for each file type

### Backend Changes
- Modified `handler_v3.py` to run extended pipeline:
  - Step 1: Generate PKL with SMPL-X mesh
  - Step 2: Calculate angles and create CSV
  - Step 3: Generate ergonomic Excel analysis
  - Step 4: Create 4 visualization videos
- Updated `s3_utils.py` to support multiple results
- All outputs uploaded to R2/S3 storage

### Frontend Changes
- Replaced single download button with dynamic buttons
- Added icons for different file types
- Support for PKL, CSV, Excel, and video downloads

### Bug Fixes
- Removed Unicode characters for Windows compatibility
- Fixed encoding issues in print statements
- Corrected file path handling in pipeline

### Breaking Changes
- Job status now returns `results` object instead of single `download_url`
- Frontend requires update to handle new response format

## Version 3.0.0 (Previous)
- Initial V3 architecture with async processing
- Single PKL file output only

---

## How to Force RunPod Update

Since RunPod caches Docker images, use the new tag to force update:

```bash
# Old image (V3)
your_username/ergonomic-analysis:v3

# New image (V4) - use this to force update
your_username/ergonomic-analysis:v4-extended-pipeline
```

Or use specific version:
```bash
your_username/ergonomic-analysis:4.0.0
```
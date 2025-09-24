# Testing Instructions for Extended Pipeline

## Setup

1. **Start the proxy server:**
   ```bash
   cd serverless_v3
   python full_proxy.py
   ```
   The proxy should run on port 5001.

2. **Open the frontend:**
   Open `serverless_v3/frontend/index-with-proxy.html` in a web browser.

## Testing Steps

1. **Upload a test video:**
   - Use a short MP4 video file for testing (ideally under 30 seconds)
   - Drag and drop or click to select the file

2. **Monitor processing:**
   The pipeline will execute these steps:
   - Step 1: Generate 3D mesh using SMPL-X (creates PKL file)
   - Step 2: Calculate angles from mesh (creates CSV file)
   - Step 3: Generate ergonomic analysis (creates Excel file)
   - Step 4: Generate 4 visualization videos (creates 4 MP4 files)

3. **Download results:**
   Once complete, you should see download buttons for:
   - 3D Mesh Data (PKL)
   - Angles Data (CSV)
   - Ergonomic Analysis (Excel)
   - Video 1-4 (MP4 files)

## What Changed

### Backend (handler_v3.py):
- Extended `process_video_async` to run the complete pipeline
- Added calls to:
  - `create_combined_angles_csv_skin.py`
  - `ergonomic_time_analysis.py`
  - `generate_4videos_from_pkl.py`
- Uploads all results to R2/S3 storage
- Returns URLs for all generated files

### Frontend (index-with-proxy.html):
- Updated result section to show multiple download buttons
- Each file type gets its own download button with icon
- Handles both new multi-file format and old PKL-only format

### Storage (s3_utils.py):
- Added 'results' field support in JobManager
- Allows storing multiple file URLs in job status

## Troubleshooting

If something fails:
1. Check the console in `full_proxy.py` for proxy errors
2. Check RunPod logs for processing errors
3. Verify all required Python packages are installed
4. Ensure SMPL-X models are in place for the 3D pipeline

## Notes

- Processing time depends on video length and RunPod GPU
- All files are stored temporarily on R2/S3 with presigned URLs
- URLs expire after the configured time (see config.py)
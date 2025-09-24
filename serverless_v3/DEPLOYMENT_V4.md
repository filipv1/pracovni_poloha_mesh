# RunPod Deployment Instructions - Version 4.0.0

## Quick Update Instructions

### 1. Build and Push New Docker Image

```bash
# Build with new tag to force update
docker build -t your_username/ergonomic-analysis:v4 .

# Also tag with version number
docker tag your_username/ergonomic-analysis:v4 your_username/ergonomic-analysis:4.0.0

# Push to Docker Hub
docker push your_username/ergonomic-analysis:v4
docker push your_username/ergonomic-analysis:4.0.0
```

### 2. Update RunPod Endpoint

1. Go to RunPod Console
2. Navigate to your Serverless endpoint
3. **IMPORTANT**: Change Docker image from:
   - OLD: `your_username/ergonomic-analysis:latest` or `:v3`
   - NEW: `your_username/ergonomic-analysis:v4`

4. Save and restart the endpoint

### 3. Verify Update

Check RunPod logs for these messages:
```
=== HANDLER V4 STARTED (Extended Pipeline) ===
Starting RunPod serverless worker V4 (Extended Pipeline)
Version: 4.0.0 - Outputs: PKL, CSV, Excel, 4 Videos
```

## What Changed in V4

### New Outputs
1. **PKL file** - 3D mesh data (same as before)
2. **CSV file** - Angle measurements (NEW)
3. **Excel file** - Ergonomic analysis (NEW)
4. **4 MP4 videos** - Visualizations (NEW)

### Processing Steps
1. Generate 3D mesh with SMPL-X
2. Calculate angles from skin vertices
3. Create ergonomic time analysis
4. Generate 4 visualization videos

### Response Format
Old V3 response:
```json
{
  "download_url": "https://..."
}
```

New V4 response:
```json
{
  "results": {
    "pkl_url": "https://...",
    "csv_url": "https://...",
    "excel_url": "https://...",
    "videos": [
      {"name": "original", "url": "https://..."},
      {"name": "mediapipe", "url": "https://..."},
      {"name": "mesh3d", "url": "https://..."},
      {"name": "overlay", "url": "https://..."}
    ]
  }
}
```

## Troubleshooting

### If you still get only PKL file:
1. RunPod is using cached old image
2. Make sure to use NEW tag (`:v4` not `:latest`)
3. You may need to:
   - Delete the old endpoint
   - Create new endpoint with v4 image
   - Or contact RunPod support to clear cache

### Check Version in Logs
Look for:
- "V3" = old version (PKL only)
- "V4" = new version (all outputs)

### Frontend Compatibility
Make sure you're using the updated frontend:
- `serverless_v3/frontend/index-with-proxy.html`
- Should show multiple download buttons when complete

## Files Modified for V4

- `runpod/handler_v3.py` - Extended pipeline
- `runpod/s3_utils.py` - Multiple results support
- `frontend/index-with-proxy.html` - Multiple downloads
- `Dockerfile` - Version 4.0.0
- `VERSION` - 4.0.0

## Contact

If the update doesn't work, check:
1. Docker Hub shows new v4 tag
2. RunPod endpoint uses v4 tag
3. RunPod logs show V4 messages
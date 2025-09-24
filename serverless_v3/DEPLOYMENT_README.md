# Serverless V3 Deployment Guide

## Architecture Overview

V3 implements an asynchronous processing pipeline with CloudFlare R2/S3 storage to handle videos of any size without timeouts.

```
Frontend → R2/S3 (presigned URL) → RunPod (async) → R2/S3 (results) → Frontend (polling)
```

## Prerequisites

1. **CloudFlare R2 Account** (recommended) or AWS S3
2. **RunPod Account** with API key
3. **Docker Hub Account** for image hosting
4. **Docker** installed locally for building images

## Setup Steps

### 1. Configure CloudFlare R2

1. Create R2 bucket:
   - Go to CloudFlare dashboard → R2
   - Create bucket named `ergonomic-analysis`
   - Note your Account ID

2. Create R2 API Token:
   - Go to R2 → Manage R2 API Tokens
   - Create token with Object Read & Write permissions
   - Save Access Key ID and Secret Access Key

### 2. Configure RunPod

1. Create new Serverless Endpoint:
   - Go to RunPod → Serverless → + New Endpoint
   - Select GPU: RTX 4090 (24GB VRAM)
   - Container image: `vaclavik/ergonomic-analysis-v3:latest`
   - Container disk: 20 GB
   - Max workers: 3
   - Idle timeout: 5 seconds
   - Execution timeout: 3600 seconds (1 hour)

2. Set Environment Variables in RunPod:
   ```
   STORAGE_PROVIDER=r2
   R2_ACCOUNT_ID=your_account_id
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   R2_BUCKET_NAME=ergonomic-analysis
   ```

3. Save endpoint ID and API key

### 3. Build and Deploy Docker Image

Windows:
```bash
cd serverless_v3
build_and_push.bat
```

Linux/Mac:
```bash
cd serverless_v3
chmod +x build_and_push.sh
./build_and_push.sh
```

### 4. Configure Frontend

Edit `frontend/index.html`:
```javascript
const RUNPOD_ENDPOINT = 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync';
const RUNPOD_API_KEY = 'YOUR_API_KEY';
```

### 5. Deploy Frontend

The frontend is a static HTML file that can be hosted on:
- **GitHub Pages** (free)
- **Netlify** (free tier)
- **Vercel** (free tier)
- **CloudFlare Pages** (free)
- **Any web server**

## Testing

### Test with small file first:
1. Open frontend in browser
2. Upload a small test video (< 100MB)
3. Monitor progress through polling
4. Download results when complete

### Test with large file:
1. Upload video > 1GB
2. Verify no timeout issues
3. Check RunPod logs for processing

## API Endpoints

The V3 handler supports these actions:

### 1. Generate Upload URL
```json
{
  "action": "generate_upload_url",
  "filename": "video.mp4"
}
```
Returns:
```json
{
  "status": "success",
  "upload_url": "https://...",
  "video_key": "uploads/video.mp4"
}
```

### 2. Start Processing
```json
{
  "action": "start_processing",
  "video_key": "uploads/video.mp4",
  "quality": "medium"
}
```
Returns:
```json
{
  "status": "success",
  "job_id": "uuid",
  "message": "Processing started"
}
```

### 3. Get Status
```json
{
  "action": "get_status",
  "job_id": "uuid"
}
```
Returns:
```json
{
  "status": "success",
  "job_status": {
    "job_id": "uuid",
    "status": "processing|completed|failed",
    "progress": 0-100,
    "download_url": "https://..." (when completed)
  }
}
```

## File Structure in R2/S3

```
bucket/
├── uploads/          # Input videos
│   └── video_xxx.mp4
├── results/          # Output PKL files
│   └── job_xxx.pkl
└── status/           # Job status JSONs
    └── job_xxx.json
```

## Monitoring

1. **RunPod Dashboard**: View active workers, GPU usage, logs
2. **CloudFlare R2**: Monitor storage usage, bandwidth
3. **Frontend Console**: Debug network requests, polling

## Cost Estimation

- **RunPod**: $0.00013/sec * processing_time
  - 10 min video ≈ 600 sec processing ≈ $0.078
- **CloudFlare R2**:
  - Storage: $0.015/GB/month
  - Operations: $0.36 per million requests
  - Egress: FREE (major advantage over S3)
- **AWS S3** (if used instead):
  - Storage: $0.023/GB/month
  - Egress: $0.09/GB (expensive for large PKL files!)

## Troubleshooting

### "Failed to return job results"
- Check R2/S3 credentials in RunPod env vars
- Verify bucket exists and has proper permissions

### Timeout errors
- Increase RunPod execution timeout (max 24 hours)
- Check if worker has enough memory/disk

### Upload fails
- Verify presigned URL is valid (1 hour expiry by default)
- Check CORS settings if frontend on different domain

### Polling never completes
- Check RunPod logs for processing errors
- Verify status JSON is being updated in R2/S3

## Security Notes

1. Never expose R2/S3 credentials in frontend
2. Use presigned URLs with short expiry times
3. Implement rate limiting in production
4. Consider adding authentication layer

## Production Checklist

- [ ] Set up production R2 bucket with lifecycle rules
- [ ] Configure RunPod autoscaling
- [ ] Add error alerting (email/Slack)
- [ ] Implement job cleanup (delete old files)
- [ ] Add user authentication
- [ ] Set up CDN for frontend
- [ ] Monitor costs and usage
- [ ] Create backup strategy

## Support

For issues, check:
1. RunPod logs: Dashboard → Endpoint → Logs
2. Browser console for frontend errors
3. R2 bucket for file presence
4. Network tab for API responses
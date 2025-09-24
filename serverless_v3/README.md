# Serverless V3 - Asynchronous Processing Architecture

## 🚀 Overview

V3 is a production-ready serverless architecture designed to handle video files of **any size** without timeout limitations. It uses CloudFlare R2 (or S3) for storage and RunPod for GPU processing.

### Key Features

- ✅ **No size limits** - Handle GB+ video files
- ✅ **No timeout issues** - Asynchronous processing with polling
- ✅ **Cost-effective** - CloudFlare R2 with free egress
- ✅ **Scalable** - RunPod auto-scaling with GPU workers
- ✅ **Simple frontend** - Static HTML with JavaScript polling
- ✅ **Production-ready** - Robust error handling and retry logic

## 🏗️ Architecture

```
┌──────────┐     ┌────────────┐     ┌─────────┐     ┌───────────┐
│          │────▶│            │────▶│         │────▶│           │
│ Frontend │     │ R2/S3      │     │ RunPod  │     │ R2/S3     │
│  (HTML)  │◀────│ (Storage)  │◀────│ (GPU)   │     │ (Results) │
│          │     │            │     │         │     │           │
└──────────┘     └────────────┘     └─────────┘     └───────────┘
     ▲                                                      │
     │                   Polling for status                 │
     └──────────────────────────────────────────────────────┘
```

### Workflow

1. **Upload**: Frontend gets presigned URL and uploads video directly to R2/S3
2. **Process**: RunPod worker downloads video, processes with SMPL-X, uploads results
3. **Poll**: Frontend polls for job status until completion
4. **Download**: Frontend gets presigned URL to download results

## 📁 Project Structure

```
serverless_v3/
├── config/
│   └── config.py           # Central configuration
├── runpod/
│   ├── handler_v3.py       # RunPod async handler
│   └── s3_utils.py         # R2/S3 storage utilities
├── frontend/
│   └── index.html          # Web interface with polling
├── Dockerfile              # Docker image for RunPod
├── requirements_v3.txt     # Python dependencies
├── build_and_push.bat      # Windows deployment script
├── build_and_push.sh       # Linux/Mac deployment script
├── test_local.py           # Local testing utilities
└── DEPLOYMENT_README.md    # Detailed deployment guide
```

## ⚡ Quick Start

### 1. Prerequisites

- CloudFlare account with R2 enabled
- RunPod account with API key
- Docker installed locally
- Docker Hub account

### 2. Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in credentials
3. Build and push Docker image:
   ```bash
   # Windows
   build_and_push.bat

   # Linux/Mac
   ./build_and_push.sh
   ```

### 3. Deploy to RunPod

1. Create serverless endpoint in RunPod dashboard
2. Set Docker image: `vaclavik/ergonomic-analysis-v3:latest`
3. Configure environment variables (R2 credentials)
4. Note endpoint ID and API key

### 4. Configure Frontend

Edit `frontend/index.html`:
```javascript
const RUNPOD_ENDPOINT = 'your_endpoint_url';
const RUNPOD_API_KEY = 'your_api_key';
```

### 5. Test

1. Open `frontend/index.html` in browser
2. Upload test video
3. Watch progress through polling
4. Download results when complete

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available options:

- `STORAGE_PROVIDER`: Choose between 'r2' or 's3'
- `R2_*` or `S3_*`: Storage credentials
- `RUNPOD_*`: RunPod endpoint configuration
- Processing parameters (quality, timeouts, etc.)

### Quality Settings

- `low`: Fast processing, lower accuracy
- `medium`: Balanced (default)
- `high`: Better accuracy, slower
- `ultra`: Maximum accuracy, longest processing

## 💰 Cost Analysis

### RunPod GPU
- RTX 4090: ~$0.00013/second
- 10-minute video ≈ 600s processing ≈ $0.078

### CloudFlare R2
- Storage: $0.015/GB/month
- Operations: $0.36 per million
- **Egress: FREE** (major advantage!)

### AWS S3 (alternative)
- Storage: $0.023/GB/month
- **Egress: $0.09/GB** (expensive for large files!)

## 🔍 API Reference

### Actions

#### 1. Generate Upload URL
```json
// Request
{
  "action": "generate_upload_url",
  "filename": "video.mp4"
}

// Response
{
  "status": "success",
  "upload_url": "https://...",
  "video_key": "uploads/video.mp4"
}
```

#### 2. Start Processing
```json
// Request
{
  "action": "start_processing",
  "video_key": "uploads/video.mp4",
  "quality": "medium"
}

// Response
{
  "status": "success",
  "job_id": "uuid-xxx",
  "message": "Processing started"
}
```

#### 3. Get Status
```json
// Request
{
  "action": "get_status",
  "job_id": "uuid-xxx"
}

// Response
{
  "status": "success",
  "job_status": {
    "status": "processing",
    "progress": 45,
    "download_url": null
  }
}
```

## 🐛 Troubleshooting

### Common Issues

**Timeout Errors**
- Check RunPod execution timeout settings (max 24h)
- Verify worker has enough memory

**Upload Failures**
- Check presigned URL expiry (default 1 hour)
- Verify CORS settings for cross-origin requests

**Processing Stuck**
- Check RunPod logs for errors
- Verify R2/S3 credentials are correct
- Ensure bucket exists and has proper permissions

## 📊 Monitoring

- **RunPod Dashboard**: View worker status, GPU usage, logs
- **CloudFlare R2**: Monitor storage, bandwidth, costs
- **Browser Console**: Debug frontend, network requests

## 🔒 Security

- Never expose storage credentials in frontend
- Use short-lived presigned URLs
- Implement rate limiting for production
- Add authentication layer for users

## 🎯 Production Checklist

- [ ] Production R2 bucket with lifecycle policies
- [ ] RunPod autoscaling configuration
- [ ] Error alerting (email/Slack)
- [ ] Automated job cleanup
- [ ] User authentication system
- [ ] CDN for frontend hosting
- [ ] Cost monitoring alerts
- [ ] Backup and recovery plan

## 📚 Additional Resources

- [CloudFlare R2 Documentation](https://developers.cloudflare.com/r2/)
- [RunPod API Reference](https://docs.runpod.io/docs/serverless/endpoints/job-operations)
- [SMPL-X Model Documentation](https://smpl-x.is.tue.mpg.de/)

## 🤝 Support

For issues or questions:
1. Check RunPod logs for processing errors
2. Verify R2/S3 bucket permissions
3. Review browser console for frontend issues
4. See DEPLOYMENT_README.md for detailed setup

---

**V3 Architecture** - Built for production, designed for scale 🚀
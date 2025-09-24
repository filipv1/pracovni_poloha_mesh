# V3 Serverless Architecture - Complete Documentation

## 📅 Implementation Date: September 23, 2025

## 🎯 Overview

V3 is a complete rewrite of the serverless architecture designed to handle videos of **any size** without timeouts. It implements an asynchronous processing pipeline with CloudFlare R2 storage and RunPod GPU workers.

## 🏗️ Architecture Components

### 1. Frontend (Browser)
- **Location**: `serverless_v3/frontend/index-configured.html` or `index-with-proxy.html`
- **Server**: Python HTTP server on port 8000
- **Features**:
  - Drag & drop video upload
  - Progress tracking with polling
  - Automatic status updates every 5 seconds

### 2. Proxy Server (CORS Solution)
- **File**: `serverless_v3/full_proxy.py`
- **Port**: 5001
- **Purpose**: Solves CORS issues for:
  - RunPod API calls
  - CloudFlare R2 uploads
- **Endpoints**:
  - `/runpod` - Proxies to RunPod API
  - `/upload` - Proxies to R2 presigned URLs
  - `/health` - Health check

### 3. RunPod Serverless Worker
- **Handler**: `serverless_v3/runpod/handler_v3.py`
- **Endpoint ID**: `d1mtcfjymab45g`
- **Docker Image**: `vaclavikmasa/ergonomic-analysis-v3:v3-with-models`
- **Actions**:
  - `generate_upload_url` - Creates R2 presigned URL
  - `start_processing` - Begins async video processing
  - `get_status` - Returns job status
  - `generate_download_url` - Creates download URL

### 4. CloudFlare R2 Storage
- **Bucket**: `ergonomic-analysis`
- **Structure**:
  ```
  uploads/     # Input videos
  results/     # Output PKL files
  status/      # Job status JSONs
  ```

## 🔄 Processing Flow

1. **Upload Request**:
   ```
   Browser → Proxy → RunPod → R2 Presigned URL
   ```

2. **Video Upload**:
   ```
   Browser → Proxy → R2 Storage
   ```

3. **Processing**:
   ```
   RunPod downloads from R2 → Processes → Uploads PKL to R2
   ```

4. **Status Polling**:
   ```
   Browser → Proxy → RunPod → R2 Status JSON
   ```

## 🐛 Known Issues & Solutions

### 1. CORS Errors
**Problem**: Direct browser → API calls blocked by CORS
**Solution**: Proxy server on port 5001 handles all API calls

### 2. Failed to fetch
**Problem**: Frontend opened as `file://` can't make HTTP requests
**Solution**: Serve frontend via HTTP server (`python -m http.server 8000`)

### 3. SMPL-X models missing
**Problem**: Docker image didn't include SMPLX_NEUTRAL.npz
**Solution**: Added to Dockerfile, rebuild with `rebuild_with_models.bat`

### 4. Port conflicts
**Problem**: Port 5000 was already in use
**Solution**: Changed proxy to port 5001

### 5. UUID not subscriptable
**Problem**: `uuid.uuid4()[:8]` error in handler
**Solution**: Changed to `str(uuid.uuid4())[:8]`

## 🐳 Docker Setup

### Dockerfile Key Points
```dockerfile
# Base image with PyTorch and CUDA
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Critical: SMPL-X model MUST be included
COPY models/smplx/SMPLX_NEUTRAL.npz /app/models/smplx/

# V3 specific handler
CMD ["python", "-u", "serverless_v3/runpod/handler_v3.py"]
```

### Build Commands
```bash
# Build from serverless_v3 directory
docker build -f Dockerfile -t vaclavikmasa/ergonomic-analysis-v3:TAG ..

# Tags used:
- v3latestv2 (bug fixes)
- v3-with-models (includes SMPL-X model)
```

### Image Size
- Base image: ~6GB
- With SMPL-X model: +104MB
- Total: ~6.1GB

## 🚀 Launch Procedure

### Automatic (Recommended)
```bash
cd serverless_v3
START_V3.bat
```
This will:
1. Install Flask if missing
2. Start proxy server (port 5001)
3. Start frontend server (port 8000)
4. Open browser automatically

### Manual
```bash
# Terminal 1 - Proxy
cd serverless_v3
pip install flask flask-cors
python full_proxy.py

# Terminal 2 - Frontend
cd serverless_v3/frontend
python -m http.server 8000

# Browser
http://localhost:8000/index-with-proxy.html
```

## 🔑 Credentials & Configuration

### CloudFlare R2
- **Account ID**: `605252007a9788aa8b697311c0bcfec6`
- **Bucket**: `ergonomic-analysis`
- **Note**: Credentials stored in `.env` file

### RunPod
- **Endpoint ID**: `d1mtcfjymab45g`
- **API Key**: Stored in `.env`
- **Container Image**: `vaclavikmasa/ergonomic-analysis-v3:v3-with-models`

### Environment Variables (RunPod)
```
STORAGE_PROVIDER=r2
R2_ACCOUNT_ID=xxx
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
R2_BUCKET_NAME=ergonomic-analysis
```

## 📊 Performance & Costs

### Processing Time
- 10 second video: ~20-30 seconds
- 1 minute video: ~2-3 minutes
- 10 minute video: ~20-30 minutes

### Costs
- **RunPod**: $0.00013/sec (~$0.08 per 10min video)
- **CloudFlare R2**:
  - Storage: $0.015/GB/month
  - Egress: FREE (major advantage!)
- **Total per video**: ~$0.10-0.20

## 🎭 Frontend Bugs

### Minor Issues
1. **Progress bar**: Sometimes jumps or doesn't update smoothly
2. **Error messages**: Not always clear when R2 upload fails
3. **Download button**: Might not appear immediately after completion
4. **Polling**: Continues even after completion (minor performance issue)

### Workarounds
- Refresh page if stuck
- Check RunPod logs for real status
- Download URL expires after 24 hours

## 📝 File Structure

```
serverless_v3/
├── frontend/
│   ├── index.html                  # Original (needs CORS fix)
│   ├── index-configured.html       # With credentials
│   └── index-with-proxy.html       # Uses proxy (WORKING)
├── runpod/
│   ├── handler_v3.py               # Async RunPod handler
│   └── s3_utils.py                 # R2/S3 storage utilities
├── config/
│   └── config.py                   # Central configuration
├── Dockerfile                      # Docker image definition
├── full_proxy.py                   # CORS proxy server
├── START_V3.bat                    # One-click launcher
├── rebuild_with_models.bat         # Docker rebuild script
└── requirements_v3.txt             # Python dependencies
```

## ✅ What Works

1. **Video upload** - Any size via presigned URLs
2. **Async processing** - No timeout issues
3. **Progress tracking** - Real-time status updates
4. **PKL download** - Successful result retrieval
5. **CORS handling** - Proxy solves all issues
6. **R2 storage** - Free egress is huge benefit

## ❌ What Could Be Better

1. **Frontend polish** - Some UI bugs
2. **Error handling** - Better user messages
3. **Docker size** - 6GB+ is quite large
4. **Setup complexity** - Requires multiple services
5. **Documentation** - Was created on the fly

## 🔮 Future Improvements

1. **Production deployment**:
   - Use proper WSGI server instead of Flask dev server
   - Add nginx for frontend
   - Implement proper logging

2. **Frontend improvements**:
   - Fix progress bar smoothness
   - Add better error messages
   - Show processing time estimates

3. **Optimization**:
   - Smaller Docker base image
   - Cache SMPL-X models in RunPod network volume
   - Implement job cleanup after X days

4. **Monitoring**:
   - Add health checks
   - Implement alerting for failures
   - Track processing metrics

## 📚 Lessons Learned

1. **CORS is a pain** - Always use proxy for browser → API
2. **RunPod has limits** - 10MB response, 300s HTTP timeout
3. **Docker context matters** - Build from parent directory
4. **Port conflicts happen** - Always check what's running
5. **Documentation is crucial** - Should have written this earlier

## 🙏 Acknowledgments

This was a challenging implementation with many iterations. The final solution works but could be more elegant. The key insight was that synchronous processing doesn't scale - async with polling is the way.

---

**Status**: FUNCTIONAL WITH MINOR BUGS
**Production Ready**: 80% (needs polish)
**User Satisfaction**: "funguje relativně ok" ✅

---

*Created under pressure but works!* 💪
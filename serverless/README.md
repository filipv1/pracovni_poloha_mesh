# Ergonomic Analyzer - Serverless Architecture

Professional workplace posture analysis using 3D pose estimation, deployed on serverless infrastructure.

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  Static Web     │ → GitHub Pages (FREE)
│  HTML/JS/CSS    │
└────────┬────────┘
         │
    ┌────▼────┐
    │  AUTH   │ → Simple password protection
    └────┬────┘
         │
┌────────▼────────┐
│ Cloudflare      │ → Workers + KV + R2 (FREE tier)
│ Orchestrator    │   100k requests/day
└────────┬────────┘
         │
┌────────▼────────┐
│ RunPod          │ → GPU Processing ($0.0001/sec)
│ Serverless      │   RTX 4090/A10G
└─────────────────┘
```

## ✨ Features

- **3D Human Mesh Reconstruction** - MediaPipe → SMPL-X fitting
- **Ergonomic Analysis** - Trunk, neck, and arm angle calculations
- **Skin-based Measurements** - Surface vertex analysis for visual accuracy
- **Professional Reports** - XLSX with statistics and PKL with full 3D data
- **Email Notifications** - Completion/failure alerts
- **48-hour Storage** - Automatic cleanup
- **Simple Auth** - Password protection
- **Usage Logging** - Track who uses the service

## 🚀 Quick Start

### Prerequisites

1. **Accounts needed (all have free tiers):**
   - [Docker Hub](https://hub.docker.com/) - For container registry
   - [RunPod](https://runpod.io/) - For GPU processing
   - [Cloudflare](https://cloudflare.com/) - For Worker + Storage
   - [Resend](https://resend.com/) - For emails
   - [GitHub](https://github.com/) - For frontend hosting

2. **Local tools:**
   - Docker Desktop
   - Node.js 18+
   - Git

### Setup Instructions

#### Step 1: Configure Environment

```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your values
```

#### Step 2: Local Testing (Optional)

```bash
# Test the processing pipeline locally
cd serverless
docker-compose up

# Frontend will be at: http://localhost:8080
# Test processing at: http://localhost:8000
```

#### Step 3: Deploy RunPod Container

```bash
# Windows
deploy.bat
# Choose option 2 (RunPod only)

# Linux/Mac
./deploy.sh
# Choose option 2 (RunPod only)
```

After Docker push completes:
1. Go to [RunPod Console](https://runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - Name: `ergonomic-analyzer`
   - Container Image: `your-dockerhub-username/ergonomic-analyzer:latest`
   - GPU: `RTX 4090` or `A10G`
   - Container Disk: `20 GB`
   - Max Workers: `1` (increase based on usage)
4. Copy the Endpoint ID
5. Add to `.env`: `RUNPOD_ENDPOINT_ID=your_endpoint_id`

#### Step 4: Deploy Cloudflare Worker

```bash
# Install Wrangler CLI if needed
npm install -g wrangler

# Deploy Worker
# Windows: deploy.bat → option 3
# Linux/Mac: ./deploy.sh → option 3
```

The script will:
- Create KV namespaces for jobs and logs
- Create R2 bucket for file storage
- Set all secrets
- Deploy the worker

#### Step 5: Deploy Frontend

```bash
# Windows: deploy.bat → option 4
# Linux/Mac: ./deploy.sh → option 4
```

Then:
1. Go to your repo Settings → Pages
2. Source: Deploy from branch
3. Branch: `gh-pages` / `root`
4. Save

Your app will be live at: `https://your-github-username.github.io/ergonomic-analyzer`

## 📁 Project Structure

```
serverless/
├── runpod/
│   ├── Dockerfile              # GPU container definition
│   ├── handler.py              # RunPod serverless handler
│   ├── handler_local_test.py   # Local testing wrapper
│   └── requirements-frozen.txt # Python dependencies
├── cloudflare/
│   ├── worker.js               # Main orchestrator
│   └── wrangler.toml          # Cloudflare config
├── frontend/
│   └── index.html             # Complete web app
├── docker-compose.yml         # Local testing setup
├── deploy.sh                  # Linux/Mac deployment
├── deploy.bat                 # Windows deployment
└── .env.example              # Configuration template
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DOCKER_USERNAME` | Docker Hub username | `johndoe` |
| `RUNPOD_API_KEY` | RunPod API key | `RP_xxxxx` |
| `RUNPOD_ENDPOINT_ID` | RunPod endpoint (set after deploy) | `abc123` |
| `CF_SUBDOMAIN` | Cloudflare subdomain | `ergonomic` |
| `RESEND_API_KEY` | Resend.com API key | `re_xxxxx` |
| `AUTH_PASSWORD_HASH` | Password for access | `mypassword` |
| `JWT_SECRET` | Random secret for JWT | `random32chars` |
| `GITHUB_USERNAME` | GitHub username | `johndoe` |

### Cost Breakdown

| Service | Free Tier | Paid Usage | Your Cost |
|---------|-----------|------------|-----------|
| GitHub Pages | Unlimited | - | $0 |
| Cloudflare Workers | 100k req/day | $0.50/million | $0 |
| Cloudflare KV | 100k reads/day | $0.50/million | $0 |
| Cloudflare R2 | 10GB storage | $0.015/GB | $0 |
| RunPod Serverless | None | ~$0.0001/sec | ~$0.50/video |
| Resend Email | 100/day | $0.001/email | $0 |
| **Total (100 videos/month)** | | | **~$50** |

## 🧪 Testing

### Local Testing with Docker

```bash
cd serverless

# Place test video in test/sample_video.mp4
mkdir test
cp /path/to/your/video.mp4 test/sample_video.mp4

# Run test
docker-compose up

# Check results in output/
```

### Test Cloudflare Worker Locally

```bash
cd serverless/cloudflare
wrangler dev

# Worker will be at http://localhost:8787
```

### Manual API Testing

```bash
# Test health check
curl https://your-subdomain.workers.dev/health

# Test auth
curl -X POST https://your-subdomain.workers.dev/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"your_password"}'
```

## 📊 Monitoring

### Cloudflare Logs

```bash
cd serverless/cloudflare
wrangler tail

# Or in dashboard: https://dash.cloudflare.com
```

### RunPod Metrics

View in [RunPod Console](https://runpod.io/console/serverless):
- Request count
- Average duration
- GPU utilization
- Error rate

### Usage Analytics

Query KV storage for usage logs:
```javascript
// In Cloudflare dashboard → Workers → KV
// Browse LOGS namespace
```

## 🐛 Troubleshooting

### Common Issues

**Docker build fails**
- Ensure you're using RunPod base image
- Check CUDA compatibility
- Verify all Python files are copied

**RunPod endpoint not responding**
- Check endpoint status in console
- Verify Docker image was pushed
- Check RunPod logs for errors

**Cloudflare Worker errors**
- Check wrangler tail for logs
- Verify all secrets are set
- Check R2 bucket permissions

**Frontend can't connect**
- Update API_URL in index.html
- Check CORS headers in worker
- Verify auth token is valid

### Debug Mode

Enable debug logging:
```javascript
// In worker.js
const DEBUG = true;

// In handler.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔐 Security

- Password protected access
- JWT tokens for session management
- 48-hour automatic data deletion
- No persistent user data storage
- Encrypted communication (HTTPS)
- Isolated GPU processing environment

## 📈 Scaling

### Increase Capacity

1. **RunPod**: Increase "Max Workers" in endpoint settings
2. **Cloudflare**: Upgrade to Workers Paid plan ($5/month)
3. **Storage**: R2 automatically scales

### Optimize Costs

1. Use `medium` quality for faster processing
2. Set RunPod idle timeout to 5 seconds
3. Enable R2 lifecycle rules for cleanup
4. Monitor usage and adjust worker count

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test locally with docker-compose
4. Submit pull request

## 📄 License

This project is proprietary software for internal company use.

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Review RunPod/Cloudflare documentation
3. Contact system administrator

---

**Built with:** PyTorch, SMPL-X, MediaPipe, RunPod, Cloudflare Workers
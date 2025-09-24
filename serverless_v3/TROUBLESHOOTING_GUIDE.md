# V3 Serverless - Troubleshooting Guide

## Quick Diagnostics Checklist

Before diving into specific issues, verify these components:

```bash
# 1. Check proxy is running
curl http://localhost:5001/health

# 2. Check frontend is accessible
curl http://localhost:8000

# 3. Check Docker image exists
docker images | grep ergonomic-analysis-v3

# 4. Check RunPod endpoint status
# Visit: https://runpod.ai/console/serverless
```

## Common Issues and Solutions

### 🔴 Frontend Issues

#### Issue: "Failed to fetch" error
**Symptoms**:
- Browser console shows CORS errors
- Upload button doesn't work

**Solutions**:
```bash
# 1. Ensure proxy is running
cd serverless_v3
python full_proxy.py

# 2. Use correct URL
# ✅ http://localhost:8000/index-with-proxy.html
# ❌ file:///C:/path/to/index.html

# 3. Check proxy port (should be 5001)
netstat -an | findstr 5001
```

---

#### Issue: Page loads but nothing happens
**Symptoms**:
- Blank page or no response to clicks
- No errors in console

**Solutions**:
```bash
# 1. Clear browser cache
# Ctrl+F5 or Cmd+Shift+R

# 2. Check JavaScript console
# F12 → Console tab

# 3. Verify both servers running
tasklist | findstr python
```

---

### 🔴 Upload Issues

#### Issue: Upload starts but never completes
**Symptoms**:
- Progress bar stuck at 20%
- "Uploading..." message doesn't change

**Diagnosis**:
```python
# Test R2 credentials
import boto3
client = boto3.client(
    's3',
    endpoint_url='https://[account-id].r2.cloudflarestorage.com',
    aws_access_key_id='[key]',
    aws_secret_access_key='[secret]'
)
print(client.list_buckets())
```

**Solutions**:
1. Check R2 credentials in `.env` file
2. Verify bucket exists: `ergonomic-analysis`
3. Check network connectivity to CloudFlare

---

### 🔴 Processing Issues

#### Issue: Processing fails immediately
**Symptoms**:
- Status changes to "failed" within seconds
- No processing progress shown

**Check RunPod logs**:
```bash
# Via dashboard: RunPod Console → Endpoints → Logs

# Common errors:
# 1. FileNotFoundError: /app/models/smplx/SMPLX_NEUTRAL.npz
#    → Rebuild Docker image with models

# 2. ModuleNotFoundError: No module named 'mediapipe'
#    → Check requirements in Dockerfile

# 3. CUDA out of memory
#    → Reduce batch size or video resolution
```

---

#### Issue: SMPL-X model not found
**Error**: `FileNotFoundError: /app/models/smplx/SMPLX_NEUTRAL.npz`

**Solutions**:
```bash
# 1. Verify model exists locally
ls models/smplx/SMPLX_NEUTRAL.npz

# 2. Rebuild Docker image
cd serverless_v3
rebuild_with_models.bat

# 3. Verify in Docker image
docker run -it vaclavikmasa/ergonomic-analysis-v3:v3-with-models \
  ls -la /app/models/smplx/

# 4. Update RunPod endpoint
# Dashboard → Endpoints → Update container image
```

---

### 🔴 Docker Issues

#### Issue: Docker build fails
**Symptoms**:
- "COPY failed: file not found"
- "no such file or directory"

**Solutions**:
```bash
# Build from correct directory
cd C:\Users\vaclavik\ruce7\pracovni_poloha_mesh
docker build -f serverless_v3/Dockerfile -t image:tag .
#                                                      ^ Note the dot!

# NOT from serverless_v3!
# ❌ cd serverless_v3 && docker build .
```

---

#### Issue: Docker push fails
**Symptoms**:
- "denied: requested access to the resource is denied"
- "unauthorized: authentication required"

**Solutions**:
```bash
# 1. Login to Docker Hub
docker login

# 2. Check image name matches username
# ✅ vaclavikmasa/ergonomic-analysis-v3:tag
# ❌ ergonomic-analysis-v3:tag

# 3. Verify push permissions
docker push vaclavikmasa/ergonomic-analysis-v3:test
```

---

### 🔴 RunPod Issues

#### Issue: Endpoint not responding
**Symptoms**:
- API calls timeout
- No logs appearing

**Diagnosis**:
```bash
# Test direct API call
curl -X POST https://api.runpod.ai/v2/d1mtcfjymab45g/run \
  -H "Authorization: Bearer [API_KEY]" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "test"}}'
```

**Solutions**:
1. Check RunPod credits/balance
2. Verify API key is correct
3. Check endpoint is "Ready" in dashboard
4. Restart endpoint if needed

---

#### Issue: Wrong endpoint ID
**Symptoms**:
- 404 errors from RunPod API
- "Endpoint not found"

**Solutions**:
```python
# Update in handler_v3.py
RUNPOD_ENDPOINT_ID = 'd1mtcfjymab45g'  # Your actual endpoint ID

# Update in frontend JavaScript
const ENDPOINT_ID = 'd1mtcfjymab45g';
```

---

### 🔴 Network Issues

#### Issue: Port already in use
**Error**: `[Errno 10048] Only one usage of each socket address`

**Solutions**:
```bash
# 1. Find process using port
netstat -ano | findstr :5001

# 2. Kill process
taskkill /PID [process_id] /F

# 3. Use different port
# Edit full_proxy.py:
app.run(host='0.0.0.0', port=5002)  # Change port
```

---

### 🔴 CloudFlare R2 Issues

#### Issue: R2 bucket not accessible
**Symptoms**:
- Upload URL generation fails
- "Access Denied" errors

**Solutions**:
```bash
# 1. Verify bucket exists
# CloudFlare Dashboard → R2 → Buckets

# 2. Check API token permissions
# Needs: Object Read & Write

# 3. Test with AWS CLI
aws s3 ls s3://ergonomic-analysis/ \
  --endpoint-url https://[account-id].r2.cloudflarestorage.com
```

---

## Debug Commands Reference

### Test Individual Components

```python
# Test SMPL-X model loading
python -c "
import numpy as np
model = np.load('models/smplx/SMPLX_NEUTRAL.npz', allow_pickle=True)
print(f'Model loaded, shape params: {model[\"shape\"].shape}')
"

# Test MediaPipe
python -c "
import mediapipe as mp
mp_pose = mp.solutions.pose
print('MediaPipe loaded successfully')
"

# Test RunPod connection
python test_deployment.py
```

### Check System Status

```bash
# Windows
# Check Python processes
tasklist | findstr python

# Check ports
netstat -an | findstr "5001 8000"

# Check Docker
docker ps -a

# Linux/Mac
# Check processes
ps aux | grep python

# Check ports
lsof -i :5001 -i :8000

# Check Docker
docker ps -a
```

### View Logs

```bash
# Proxy logs
# Terminal running full_proxy.py

# Frontend logs
# Browser: F12 → Console

# RunPod logs
# Dashboard: https://runpod.ai/console/serverless → Endpoints → Logs

# Docker logs
docker logs [container_id]
```

## Emergency Recovery

### Complete System Reset

```bash
# 1. Stop all services
taskkill /f /im python.exe

# 2. Clear Docker cache
docker system prune -a

# 3. Rebuild everything
cd serverless_v3
rebuild_with_models.bat

# 4. Restart services
START_V3.bat
```

### Rollback to Previous Version

```bash
# Use previous Docker image
docker pull vaclavikmasa/ergonomic-analysis-v3:v3latestv2

# Update RunPod endpoint to use old image
```

## Performance Optimization

### Slow Processing
```python
# Reduce quality in handler_v3.py
quality_settings = {
    'medium': {  # Use this instead of 'ultra'
        'max_iterations': 50,
        'convergence_threshold': 1e-3
    }
}
```

### Memory Issues
```python
# Add to handler_v3.py
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

### Network Timeouts
```python
# Increase timeout in full_proxy.py
response = requests.post(url, timeout=300)  # 5 minutes
```

## Getting Help

### Log Collection for Support

```bash
# Collect all relevant logs
cd serverless_v3
mkdir debug_logs

# Copy proxy output
# Copy RunPod logs (from dashboard)
# Copy browser console (F12 → Console → Right-click → Save as...)

# System info
systeminfo > debug_logs/system.txt
docker version > debug_logs/docker.txt
pip freeze > debug_logs/pip_packages.txt
```

### Support Channels

1. **GitHub Issues**: Report bugs with logs
2. **RunPod Discord**: For RunPod-specific issues
3. **CloudFlare Support**: For R2 issues

---

## Prevention Tips

### Before Deployment
- ✅ Test locally first
- ✅ Verify all credentials
- ✅ Check Docker image size (<10GB)
- ✅ Ensure models are included

### During Operation
- ✅ Monitor RunPod credits
- ✅ Check logs regularly
- ✅ Keep backups of working images
- ✅ Document any changes

### After Issues
- ✅ Document the solution
- ✅ Update this guide
- ✅ Share with team

---

**Remember**: Most issues are related to:
1. CORS/proxy not running
2. Missing SMPL-X models
3. Wrong directories/ports
4. Credentials/permissions

**When in doubt**: Run `START_V3.bat` for automated setup!

---

*Last updated: September 23, 2025*
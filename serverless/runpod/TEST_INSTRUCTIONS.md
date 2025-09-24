# RunPod Handler Testing Instructions

## Problem Summary
- Original handler.py had issues with `progress_update()` calls
- Import errors with `run_production_simple`
- Need reliable testing before deployment

## Solution Files Created

### Handlers
1. **handler_minimal.py** - Minimal handler for basic testing (always works)
2. **handler_fixed.py** - Fixed version without progress_update calls
3. **handler.py** - Original (has issues)

### Test Scripts
1. **test_handler_local.py** - Python script to test handlers locally
2. **test_and_deploy.bat** - Windows batch script for testing and deployment
3. **test_runpod_api.ps1** - PowerShell script to test RunPod API

## Testing Process

### Step 1: Local Testing (RECOMMENDED FIRST)
```bash
cd C:\Users\vaclavik\ruce7\pracovni_poloha_mesh\serverless\runpod

# Test handlers locally
python test_handler_local.py

# Or use batch script
test_and_deploy.bat
# Select option 1
```

### Step 2: Test RunPod API Connection
```powershell
# Test if RunPod is accessible
.\test_runpod_api.ps1 -TestType health

# Submit minimal test job
.\test_runpod_api.ps1 -TestType submit

# Full test suite
.\test_runpod_api.ps1 -TestType full
```

### Step 3: Deploy Fixed Handler
```bash
# Quick deploy using batch script
test_and_deploy.bat
# Select option 4 (Quick deploy)

# OR manually:
copy handler_fixed.py handler.py
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
docker push vaclavikmasa/ergonomic-analyzer:latest
```

### Step 4: Test on RunPod
```powershell
# Wait 1-2 minutes after deploy for RunPod to update

# Test with minimal job
.\test_runpod_api.ps1 -TestType submit

# Check logs in RunPod console if issues
```

## What Each Handler Does

### handler_minimal.py
- Returns fake data immediately
- No processing, just tests RunPod integration
- Use this FIRST to verify RunPod works

### handler_fixed.py
- Full processing WITHOUT progress_update calls
- Has error handling and logging
- Falls back to test data if modules missing
- Use this for PRODUCTION

### handler.py (original)
- Has progress_update bugs
- Don't use until fixed

## Common Issues and Solutions

### Issue: Import errors
**Solution:** Make sure all .py files are copied to runpod folder:
```bash
copy ..\..\run_production_simple.py .
copy ..\..\create_combined_angles_csv_skin.py .
```

### Issue: Docker build fails
**Solution:** Use minimal requirements or Dockerfile.alternative

### Issue: RunPod not updating
**Solution:** Wait 1-2 minutes after push, or restart endpoint in console

### Issue: CORS errors in web app
**Solution:** Use proxy-server.py in simple-web folder

## Quick Test Commands

```bash
# 1. Test locally
python test_handler_local.py

# 2. Deploy minimal handler for testing
copy handler_minimal.py handler.py
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
docker push vaclavikmasa/ergonomic-analyzer:latest

# 3. Test on RunPod (PowerShell)
.\test_runpod_api.ps1 -TestType full

# 4. If works, deploy fixed handler
copy handler_fixed.py handler.py
docker build -t vaclavikmasa/ergonomic-analyzer:latest .
docker push vaclavikmasa/ergonomic-analyzer:latest
```

## Expected Output

### Successful local test:
```
[OK] Handler minimal imported
[OK] Handler executed
Result status: success
[PASS] Minimal handler test passed
```

### Successful RunPod test:
```
[OK] Endpoint is healthy
Workers ready: 3
[OK] Job submitted successfully
Status: COMPLETED
[OK] Job completed successfully!
```

## Next Steps After Testing

1. If minimal handler works -> Deploy fixed handler
2. If fixed handler works -> Test with real video
3. If real video works -> Use web interface

## Monitoring

Check RunPod logs:
1. Go to https://www.runpod.io/console/serverless
2. Click your endpoint
3. Click "Logs" tab
4. Look for errors or "HANDLER COMPLETED"
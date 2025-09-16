# 🚀 Deployment Instructions

## Current Situation
- ✅ **Flask app works** in simulation/fallback mode
- ❌ **RunPod API** needs valid credentials
- ❌ **Cloudflare R2** needs correct API token format
- ✅ **Email** works perfectly

## What Goes Where?

### 1️⃣ **Deploy to Render.com** (Web Interface)
This is your Flask web application that users interact with.

**Steps:**
1. Create account at https://render.com
2. Connect your GitHub repository
3. Create new Web Service
4. Use these settings:
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Add environment variables in Render dashboard:
   ```
   FLASK_SECRET_KEY=(generate new one)
   SMTP_USERNAME=vaclavik.renturi@gmail.com
   SMTP_PASSWORD=xaizlwiznvkqyypm
   # Add RunPod credentials when you have them
   ```

**Files needed on Render:**
- All files in `flask_runpod_app/` directory
- This is your web interface and API

### 2️⃣ **Deploy to RunPod** (GPU Processing)
This is where the actual 3D pose analysis runs on GPU.

**What to upload to RunPod:**
```
pracovni_poloha_mesh/
├── production_3d_pipeline_clean.py
├── quick_test_3_frames.py
├── models/
│   └── smplx/
│       ├── SMPLX_NEUTRAL.npz
│       ├── SMPLX_MALE.npz
│       └── SMPLX_FEMALE.npz
├── trunk_angle_calculator_skin.py
├── neck_angle_calculator_skin.py
├── create_combined_angles_csv_skin.py
└── requirements_runpod.txt (create this)
```

**Steps for RunPod:**
1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" → Choose GPU (RTX 4090 recommended)
3. Select "RunPod Pytorch 2.1" template
4. Set at least 50GB persistent storage
5. Deploy and note the Pod ID
6. SSH into pod and upload your processing scripts
7. Install dependencies:
   ```bash
   pip install mediapipe==0.10.8
   pip install smplx==0.1.28
   pip install open3d==0.18.0
   pip install trimesh==4.0.0
   ```

### 3️⃣ **Getting Valid Credentials**

#### Fix RunPod:
1. **Create RunPod Pod first** (see above)
2. **Get new API key:**
   - Go to https://www.runpod.io/console/user/settings
   - Create new API key
   - Update in Render environment variables

#### Fix Cloudflare R2:
1. **Create proper API token:**
   - Go to https://dash.cloudflare.com/
   - R2 → Manage R2 API Tokens
   - Create token with Object Read & Write permissions
   - **IMPORTANT**: The token should be 32 characters
   - Update in Render environment variables

## 🎯 Immediate Next Steps:

### Option A: Test Locally First (Recommended)
Continue using the app locally in simulation mode until you're ready for production.

### Option B: Deploy to Render Now
1. Push code to GitHub
2. Deploy to Render (works even without RunPod)
3. Users can upload videos and see simulated results
4. Add RunPod later when ready

### Option C: Full Production Setup
1. **First**: Create RunPod pod and get valid credentials
2. **Second**: Create Cloudflare R2 bucket and get proper API token
3. **Third**: Deploy to Render with all credentials
4. **Fourth**: Upload processing scripts to RunPod

## 📋 Quick Checklist:

To get everything working in production, you need:

- [ ] RunPod account with credits
- [ ] Create RunPod pod (GPU instance)
- [ ] Get valid RunPod API key
- [ ] Upload processing scripts to RunPod
- [ ] Cloudflare account
- [ ] Create R2 bucket
- [ ] Generate proper R2 API token (32 chars)
- [ ] Render account
- [ ] Deploy Flask app to Render
- [ ] Configure all environment variables

## 💡 Recommendation:

**Start with Option B** - Deploy to Render now with simulation mode. This gives you:
- Public URL for testing
- Real user interface
- Email notifications working
- Can add GPU processing later

The app is **designed to work without GPU** initially, so you can:
1. Deploy now
2. Test with users
3. Add GPU processing when ready
4. No code changes needed!

## 🔧 Testing Before Production:

```bash
# Test locally with mock GPU
cd flask_runpod_app
python app.py
# Upload test video at http://localhost:5000

# When ready for GPU:
python test_runpod_api.py  # After getting valid credentials
```

## 📝 Environment Variables Summary:

For `.env` file (local) or Render dashboard (production):
```env
# Required for basic operation
FLASK_SECRET_KEY=generate-a-secure-random-key
SMTP_USERNAME=vaclavik.renturi@gmail.com
SMTP_PASSWORD=xaizlwiznvkqyypm
EMAIL_FROM=vaclavik.renturi@gmail.com

# Optional - add when you have valid credentials
RUNPOD_API_KEY=your-valid-api-key-here
RUNPOD_POD_ID=your-pod-id-here
R2_ACCOUNT_ID=605252007a9788aa8b697311c0bcfec6
R2_ACCESS_KEY_ID=your-32-char-access-key
R2_SECRET_ACCESS_KEY=your-32-char-secret-key
R2_BUCKET_NAME=flaskrunpod
```

---

**Your app is ready to deploy!** Choose your path:
- **Easy**: Deploy to Render now (works immediately)
- **Full**: Get RunPod/R2 credentials first, then deploy everything
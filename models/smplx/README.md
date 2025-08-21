# SMPL-X Models Directory

## 📥 Required Files

You need to download the following SMPL-X model files and place them in this directory:

- `SMPLX_NEUTRAL.npz` (~10 MB)
- `SMPLX_MALE.npz` (~10 MB)
- `SMPLX_FEMALE.npz` (~10 MB)

## 🔗 Download Source

1. **Visit**: https://smpl-x.is.tue.mpg.de/
2. **Register** for academic/research account
3. **Download** the "SMPL-X v1.1 npz+pkl" package
4. **Extract** the `.npz` files to this directory

## 📁 Expected Structure

After download, this directory should contain:
```
models/smplx/
├── README.md (this file)
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
└── SMPLX_FEMALE.npz
```

## ⚠️ Important Notes

- **File Size**: Each model is approximately 10MB
- **License**: SMPL-X models require separate licensing agreement
- **Usage**: For research and non-commercial purposes only
- **Format**: Use `.npz` files (not `.pkl`) for best compatibility

## 🧪 Verification

After placing the files, you can verify them with:

```python
import os
from pathlib import Path

models_dir = Path("models/smplx")
required_files = ["SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"]

for file in required_files:
    file_path = models_dir / file
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024*1024)
        print(f"✅ {file}: {size_mb:.1f} MB")
    else:
        print(f"❌ {file}: Missing")
```

## 🚀 RunPod Upload Methods

### Method 1: SCP Upload
```bash
scp -P [SSH_PORT] models/smplx/*.npz root@[POD_IP]:/workspace/pracovni_poloha_mesh/models/smplx/
```

### Method 2: Jupyter Upload
1. Open Jupyter interface on RunPod
2. Navigate to `/workspace/pracovni_poloha_mesh/models/smplx/`
3. Use "Upload" button to upload each `.npz` file

### Method 3: Wget from Cloud Storage
```bash
# If you host files on Google Drive, Dropbox, etc.
cd models/smplx/
wget "YOUR_DOWNLOAD_LINK" -O SMPLX_NEUTRAL.npz
wget "YOUR_DOWNLOAD_LINK" -O SMPLX_MALE.npz
wget "YOUR_DOWNLOAD_LINK" -O SMPLX_FEMALE.npz
```

**Without these model files, the pipeline will fall back to simplified wireframe visualization.**
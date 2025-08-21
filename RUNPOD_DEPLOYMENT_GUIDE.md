# 🚀 RunPod GPU Deployment Guide
# 3D Human Mesh Pipeline Migration

## 📋 PŘÍPRAVA PRO RUNPOD

### 1. GitHub Repository Setup

Nejprve nahrajte projekt na GitHub:

```bash
# Vytvořte nový GitHub repository: "3d-human-mesh-pipeline"
# Poté v lokální složce:

git init
git add .
git commit -m "Initial 3D human mesh pipeline implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/3d-human-mesh-pipeline.git
git push -u origin main
```

### 2. Požadované soubory pro RunPod
- ✅ `production_3d_pipeline_clean.py` - Hlavní pipeline
- ✅ `requirements_runpod.txt` - GPU závislosti 
- ✅ `setup_runpod.py` - Automatická instalace
- ✅ `test_gpu_pipeline.py` - GPU test script
- ✅ `models/smplx/` - SMPL-X modely (upload ručně)

---

## 🖥️ RUNPOD SETUP KROK ZA KROKEM

### KROK 1: RunPod Account & GPU Selection

1. **Registrace:** https://runpod.io/
2. **GPU výběr:** Doporučuji:
   - **RTX 4090** (24GB VRAM) - nejlepší performance/cena
   - **RTX 3090** (24GB VRAM) - levnější alternativa
   - **A6000** (48GB VRAM) - pro velmi dlouhá videa

### KROK 2: Template Selection

**DOPORUČENÉ TEMPLATE:**
- **"PyTorch 2.0"** nebo **"CUDA Development"**
- **Ubuntu 22.04** (LTS - stabilní a podporované)
- **CUDA 11.8+** pre-installed
- **Python 3.9-3.11** kompatibilní

**NEDOPORUČUJI Windows** - Linux je výrazně rychlejší pro ML úlohy.

### KROK 3: Pod Configuration

```yaml
Template: PyTorch 2.0
GPU: RTX 4090 (24GB)
CPU: 16 vCPU
RAM: 32GB
Storage: 50GB (pro modely a videa)
Ports: 22 (SSH), 8888 (Jupyter), 8000 (Claude Code)
```

---

## 🔧 INSTALACE NA RUNPOD

### KROK 4: První připojení (SSH nebo Jupyter)

Po spuštění podu:

```bash
# SSH připojení (terminál)
ssh root@[POD_IP] -p [SSH_PORT]

# Nebo použijte Jupyter terminál přes web interface
```

### KROK 5: Environment Setup

```bash
# Aktualizace systému
apt update && apt upgrade -y

# Instalace základních nástrojů
apt install -y git curl wget htop ffmpeg

# Clone repository
git clone https://github.com/YOUR_USERNAME/3d-human-mesh-pipeline.git
cd 3d-human-mesh-pipeline

# Ověření CUDA
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### KROK 6: Python Environment

```bash
# Vytvoření conda environment
conda create -n mesh_pipeline python=3.9 -y
conda activate mesh_pipeline

# Nebo použijte venv
python -m venv mesh_env
source mesh_env/bin/activate
```

### KROK 7: Instalace závislostí

```bash
# Spusťte automatický setup script
python setup_runpod.py

# Nebo manuálně:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install smplx[all] trimesh open3d mediapipe opencv-python
pip install matplotlib scipy scikit-image imageio[ffmpeg]
```

---

## 📁 SMPL-X MODELS UPLOAD

### KROK 8: Upload SMPL-X modelů

**Metoda 1: SCP Upload**
```bash
# Z lokálního PC
scp -P [SSH_PORT] -r models/smplx/ root@[POD_IP]:/workspace/3d-human-mesh-pipeline/models/
```

**Metoda 2: Jupyter Upload**
1. Otevřete Jupyter interface
2. Navigate to `/workspace/3d-human-mesh-pipeline/models/`
3. Vytvořte složku `smplx`
4. Upload files: `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`

**Metoda 3: Google Drive/Dropbox**
```bash
# Stáhnout z cloud storage
wget "YOUR_DRIVE_LINK" -O models/smplx/SMPLX_NEUTRAL.npz
# Opakovat pro všechny modely
```

---

## 🧪 TESTOVÁNÍ NA GPU

### KROK 9: GPU Test

```bash
# Aktivovat environment
conda activate mesh_pipeline  # nebo source mesh_env/bin/activate

# Spustit GPU test
python test_gpu_pipeline.py

# Ověřit výsledky
ls -la gpu_test_output/
```

Expected output:
```
GPU Test Results:
✅ CUDA available: True
✅ GPU: RTX 4090 (24GB)
✅ SMPL-X models loaded
✅ 3 meshes generated in 8.5 seconds
✅ Average: 2.8 seconds per frame
```

---

## 🎯 CLAUDE CODE SETUP NA RUNPOD

### KROK 10: Claude Code Installation

```bash
# Stáhnout Claude Code
curl -fsSL https://claude.ai/install.sh | sh

# Nebo manual install
wget https://github.com/anthropics/claude-code/releases/latest/download/claude-code-linux-amd64.tar.gz
tar -xzf claude-code-linux-amd64.tar.gz
sudo mv claude-code /usr/local/bin/

# Ověření
claude-code --version
```

### KROK 11: Claude Code Configuration

```bash
# Inicializace v project directory
cd /workspace/3d-human-mesh-pipeline
claude-code init

# Nastavení API key (z Claude.ai settings)
export ANTHROPIC_API_KEY="your-api-key-here"

# Nebo do ~/.bashrc pro permanentní nastavení
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### KROK 12: Claude Code Testing

```bash
# Spustit Claude Code
claude-code

# Test v Claude Code session:
# - Ověřit GPU přístup
# - Spustit pipeline test
# - Kontrola výsledků
```

---

## 🚀 PRODUKČNÍ NASAZENÍ

### KROK 13: Video Processing

```bash
# Upload test video
scp -P [SSH_PORT] your_video.mp4 root@[POD_IP]:/workspace/3d-human-mesh-pipeline/

# Spustit pipeline
python production_3d_pipeline_clean.py

# Nebo s custom parametry
python -c "
from production_3d_pipeline_clean import MasterPipeline
pipeline = MasterPipeline(device='cuda')
pipeline.execute_full_pipeline('your_video.mp4', max_frames=900, quality='ultra')
"
```

### KROK 14: Download Results

```bash
# Download z RunPod na lokální PC
scp -P [SSH_PORT] -r root@[POD_IP]:/workspace/3d-human-mesh-pipeline/production_output/ ./
```

---

## 💰 COST OPTIMIZATION

### RunPod Pricing Tips:
- **Spot instances** - až 70% levnější, ale můžou být přerušeny
- **On-demand** - dražší, ale garantované
- **RTX 4090**: ~$0.50-0.80/hour (spot) vs $1.50-2.00/hour (on-demand)

### Estimated Costs:
- **30-second video**: $0.05-0.15 (5-8 minutes processing)
- **2-minute video**: $0.30-0.80 (30-45 minutes processing)
- **Development/testing**: $2-5/day

---

## 🔒 SECURITY & BACKUP

### Data Protection:
```bash
# Backup výsledků
rsync -av production_output/ backup/
tar -czf results_backup.tar.gz production_output/

# Git commit výsledků
git add . && git commit -m "GPU processing results"
git push
```

### Pod Management:
- **Save template** po úspěšném setupu
- **Stop pod** když nepoužíváte (account billing!)
- **Snapshot storage** pro důležitá data

---

## ❗ TROUBLESHOOTING

### Časté problémy:

**CUDA not available:**
```bash
nvidia-smi
pip install torch --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu118
```

**SMPL-X models not found:**
```bash
ls -la models/smplx/
chmod 644 models/smplx/*.npz
```

**Memory issues:**
```bash
# Monitoring
htop
nvidia-smi -l 1

# Reduce batch size in code
# Use frame_skip=3 instead of 2
```

**Network issues:**
```bash
# Test connectivity
ping google.com
curl -I https://github.com

# Reinstall packages
pip install --upgrade --force-reinstall [package]
```

---

## 📊 PERFORMANCE MONITORING

### Monitoring Commands:
```bash
# GPU utilization
watch -n 1 nvidia-smi

# System resources
htop

# Process monitoring
ps aux | grep python

# Disk space
df -h
```

### Expected Performance (RTX 4090):
- **Frame processing**: 2-3 seconds/frame
- **Memory usage**: 4-8GB GPU, 8-12GB RAM
- **Storage**: ~1MB per generated frame
- **Network**: Minimal after initial setup

---

## 🎉 DEPLOYMENT CHECKLIST

- [ ] RunPod account vytvořen
- [ ] RTX 4090 pod spuštěn s Ubuntu 22.04
- [ ] GitHub repository nahráno
- [ ] Environment setup dokončen
- [ ] SMPL-X modely uploadovány
- [ ] GPU test proběhl úspěšně
- [ ] Claude Code nainstalován a nakonfigurován
- [ ] Test video zpracováno
- [ ] Výsledky staženy a ověřeny

**Po dokončení budete mít plně funkční GPU pipeline pro 3D human mesh processing!** 🚀
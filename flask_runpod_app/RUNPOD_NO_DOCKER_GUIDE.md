# 🚀 RunPod bez Dockeru - Conda & GitHub

## Řešení: On-Demand Pod s Network Storage

### Jak to funguje:
1. **Flask app (Render)** přijme video
2. **Spustí RunPod pod** přes API (jen když je potřeba)
3. **Pod má persistent storage** s conda environment
4. **GitHub sync** - automaticky stáhne nejnovější kód
5. **Zpracuje video** a vypne se
6. **Platíte jen za použitý čas**

## Výhody tohoto řešení:
- ✅ **Žádný Docker** - jen conda & pip
- ✅ **GitHub deployment** - `git pull` pro update
- ✅ **Levné** - pod běží jen při zpracování
- ✅ **Persistent storage** - conda environment zůstává
- ✅ **Automatické** - vše řídí Flask app

## Nastavení krok za krokem:

### 1. Vytvořit RunPod Network Volume (jednou)
```bash
1. RunPod Console → Storage → Network Volumes
2. Create Network Volume (10GB)
3. Název: "pose-analysis-storage"
4. Region: vyberte nejbližší
```

### 2. Vytvořit RunPod Template
```bash
1. RunPod Console → Templates
2. Create Template:
   - Name: "Pose Analysis Conda"
   - Image: runpod/pytorch:2.0.1-py3.10-cuda11.8.0
   - Volume Mount Path: /workspace
   - Start Command: bash /workspace/startup.sh
```

### 3. Upload na GitHub
```bash
# Vytvořte GitHub repo a pushněte kód
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pose-analysis
git push -u origin main
```

### 4. První setup podu (jednou)
```python
# Spustit pod s network volume
import requests

api_key = "YOUR_RUNPOD_API_KEY"
headers = {"Authorization": api_key}

# GraphQL mutation pro vytvoření podu
mutation = """
mutation {
  podRentInterruptable(input: {
    templateId: "YOUR_TEMPLATE_ID"
    networkVolumeId: "YOUR_VOLUME_ID" 
    gpuTypeId: "NVIDIA RTX 3070"
    dataCenterId: "EU-CZ-1"
    minVcpuCount: 4
    minMemoryInGb: 20
    minGpuCount: 1
  }) {
    id
  }
}
"""

# Setup conda environment při prvním spuštění
```

### 5. Startup script pro pod
```bash
#!/bin/bash
# /workspace/startup.sh

# Pull latest code
cd /workspace
if [ ! -d "pose-analysis" ]; then
    git clone https://github.com/YOUR_USERNAME/pose-analysis.git
fi
cd pose-analysis
git pull

# Activate conda (už existuje na volume)
source /workspace/conda/bin/activate pose_analysis

# Start job processor
python process_jobs.py
```

### 6. Flask app spouští pod on-demand
```python
# V Flask app (core/runpod_on_demand.py)
class RunPodOnDemand:
    def process_video(self, video_path):
        # 1. Upload video na R2/S3
        video_url = upload_to_storage(video_path)
        
        # 2. Start pod if not running
        pod_id = self.start_pod_if_needed()
        
        # 3. Send job to pod
        job_id = self.send_job_to_pod(pod_id, video_url)
        
        # 4. Pod processes and auto-stops after idle
        return job_id
    
    def start_pod_if_needed(self):
        # Check if pod exists and running
        status = self.get_pod_status()
        
        if status == "EXITED":
            # Resume pod (takes 10-30 seconds)
            self.resume_pod()
        elif status == "NOT_EXISTS":
            # Create new pod with template
            self.create_pod_from_template()
        
        return self.pod_id
```

## Cenový model:
- **Pod běží jen při zpracování** (cca 1-2 min per video)
- **RTX 3070**: $0.24/hod = $0.008 per video
- **Network Storage**: $0.10/GB/měsíc (10GB = $1/měsíc)
- **Celkem**: ~$10/měsíc pro 1000 videí

## Deployment Flow:

### Krok 1: Deploy Flask na Render
```bash
# Push Flask app na GitHub
cd flask_runpod_app
git init
git add .
git commit -m "Flask app"
git push

# Connect to Render.com
# Deploy as Web Service
```

### Krok 2: Setup RunPod
```bash
# Vytvořit Network Volume (přes UI)
# Vytvořit Template (přes UI)
# První run pro setup conda
```

### Krok 3: Test
```bash
# Flask app automaticky:
1. Přijme video
2. Spustí pod
3. Počká na zpracování
4. Stáhne výsledky
5. Pod se vypne (idle timeout)
```

## Klíčové soubory:

```
flask_runpod_app/
├── core/
│   └── runpod_on_demand.py  # Řídí pody
├── runpod_setup/
│   ├── environment.yml       # Conda environment
│   ├── setup_pod.sh          # První setup
│   └── startup.sh            # Při každém startu
└── .github/
    └── workflows/
        └── deploy.yml        # Auto-deploy to pod
```

## GitHub Actions pro auto-deploy:
```yaml
name: Deploy to RunPod
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Trigger pod update
        run: |
          # Call Flask API to trigger pod restart
          curl -X POST https://your-app.onrender.com/api/update-pod
```

## Shrnutí:
- **Žádný Docker** ✅
- **Conda environment** ✅  
- **GitHub deployment** ✅
- **Levné** (platíte jen za použití) ✅
- **Automatické** ✅

Toto je **nejlepší řešení bez Dockeru**!
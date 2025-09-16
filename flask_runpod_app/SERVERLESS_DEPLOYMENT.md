# 🚀 RunPod Serverless Deployment Guide

## Proč Serverless je lepší než Pody

| Feature | Pody | Serverless |
|---------|------|------------|
| **Dostupnost GPU** | ❌ Může selhat | ✅ Vždy dostupné |
| **Cena** | $0.44/hod (i když neběží) | $0.00025/sec (jen při použití) |
| **Škálování** | Manuální | Automatické |
| **Cold start** | 1-2 minuty | 10-20 sekund |
| **Údržba** | Musíte spravovat | Zero maintenance |

## Architektura

```
Uživatel → Flask (Render) → RunPod Serverless → GPU Processing → Výsledky
```

## Kroky k nasazení

### 1. Vytvořit Docker Image

```bash
cd flask_runpod_app/serverless

# Build Docker image
docker build -t your-dockerhub/pose-analysis:latest .

# Push to Docker Hub
docker push your-dockerhub/pose-analysis:latest
```

### 2. Vytvořit Serverless Endpoint na RunPod

1. Jděte na https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Nastavte:
   - **Container Image**: `your-dockerhub/pose-analysis:latest`
   - **GPU Type**: RTX 4090 (nebo A5000)
   - **Max Workers**: 3
   - **Idle Timeout**: 5 sekund
   - **Container Disk**: 20 GB
4. Deploy!
5. Zkopírujte **Endpoint ID**

### 3. Aktualizovat Flask App

V `.env` přidejte:
```env
RUNPOD_ENDPOINT_ID=your-endpoint-id-here
RUNPOD_USE_SERVERLESS=true
```

### 4. Deploy Flask na Render

```bash
# Push na GitHub
git add .
git commit -m "Add serverless support"
git push

# Deploy na Render.com
# Automaticky se nasadí
```

## Jak to funguje

### Upload Flow:
1. **Uživatel nahraje video** → Flask app
2. **Flask nahraje video** na R2/S3
3. **Flask zavolá serverless** endpoint s URL videa
4. **Serverless stáhne video**, zpracuje na GPU
5. **Serverless nahraje výsledky** na R2/S3
6. **Flask stáhne výsledky** a pošle uživateli

### Výhody:
- ✅ **Žádné čekání na GPU** - vždy dostupné
- ✅ **Platíte jen za použití** - 100x levnější pro občasné použití
- ✅ **Automatické škálování** - zvládne 100+ uživatelů
- ✅ **Žádná údržba** - RunPod se stará o vše

## Cenová kalkulace

### Pody:
- RTX 4090: $0.44/hod = **$316/měsíc** (non-stop)
- Musí běžet pořád nebo čekat na start

### Serverless:
- RTX 4090: $0.00025/sec
- 1 video (60 sec processing) = $0.015
- 1000 videí/měsíc = **$15/měsíc**

**Úspora: 95%!**

## Testování

### Lokální test (simulace):
```bash
cd flask_runpod_app
python app.py
# Funguje i bez endpoint ID
```

### Test s reálným endpointem:
```python
from core.runpod_serverless_client import RunPodServerlessClient

client = RunPodServerlessClient(
    api_key="your-api-key",
    endpoint_id="your-endpoint-id"
)

success, job_id = client.create_job(
    video_url="https://example.com/video.mp4",
    output_key="results/test"
)

# Check status
status = client.get_job_status(job_id)
print(status)
```

## Monitoring

RunPod Console zobrazuje:
- Počet requestů
- GPU využití
- Cena
- Logy
- Metriky

## Troubleshooting

### Container se nespustí:
- Zkontrolujte Docker image je public
- Ověřte CUDA verzi v Dockerfile

### Timeout:
- Zvyšte timeout v endpointu
- Optimalizujte model loading

### Out of memory:
- Snižte batch size
- Použijte větší GPU

## Závěr

**Serverless je ideální pro produkci:**
- Spolehlivé (vždy funguje)
- Levné (platíte jen za použití)
- Škálovatelné (automaticky)
- Bez údržby

**Doporučení:** Použijte serverless místo podů!
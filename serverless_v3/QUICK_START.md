# 🚀 QUICK START - V3 Deployment (10 minut)

## Automatická instalace (DOPORUČENO)

```bash
cd serverless_v3
python setup_wizard.py
```

Wizard vás provede celým procesem krok po kroku.

---

## Manuální instalace

### 1️⃣ CloudFlare R2 (3 minuty)

1. **Vytvořte účet**: https://dash.cloudflare.com/sign-up
2. **Vytvořte R2 bucket**:
   - Jděte na R2 → Create Bucket
   - Název: `ergonomic-analysis`
3. **Získejte API klíče**:
   - R2 → Manage R2 API Tokens → Create Token
   - Permissions: Object Read & Write
   - **Zkopírujte**:
     - Account ID: `abc123...`
     - Access Key ID: `xyz789...`
     - Secret Access Key: `secret...`

### 2️⃣ RunPod (5 minut)

1. **Vytvořte účet**: https://www.runpod.io/console/signup
2. **Získejte API key**:
   - Settings → API Keys → + API Key
   - **Zkopírujte API key**: `runpod_xxx...`
3. **Vytvořte Serverless Endpoint**:
   - Serverless → + New Endpoint
   - **Nastavení**:
     ```
     Container Image: vaclavik/ergonomic-analysis-v3:latest
     GPU: RTX 4090 24GB
     Container Disk: 20 GB
     Max Workers: 3
     Idle Timeout: 5 seconds
     Execution Timeout: 3600 seconds
     ```
4. **Přidejte Environment Variables** (klikněte Edit):
   ```
   STORAGE_PROVIDER=r2
   R2_ACCOUNT_ID=váš_account_id
   R2_ACCESS_KEY_ID=váš_access_key
   R2_SECRET_ACCESS_KEY=váš_secret_key
   R2_BUCKET_NAME=ergonomic-analysis
   ```
5. **Deploy** → Zkopírujte **Endpoint ID**: `abc123xyz`

### 3️⃣ Konfigurace Frontendu (2 minuty)

1. **Otevřete** `frontend/index.html`
2. **Najděte řádky 162-163** a nahraďte:
   ```javascript
   const RUNPOD_ENDPOINT = 'https://api.runpod.ai/v2/abc123xyz/runsync';
   const RUNPOD_API_KEY = 'runpod_xxx...';
   ```
3. **Uložte** soubor

### 4️⃣ Spuštění

**Otevřete** `frontend/index.html` v prohlížeči a můžete začít!

---

## 📊 Testování

1. **Malý test** (< 10 MB):
   - Nahrajte krátké video (10 sekund)
   - Sledujte progress
   - Stáhněte výsledek

2. **Velký test** (> 1 GB):
   - Nahrajte dlouhé video
   - Ověřte, že nedojde k timeoutu

---

## 💡 Tipy

### Kde najít věci:

**CloudFlare R2 Dashboard:**
```
https://dash.cloudflare.com → R2
```

**RunPod Dashboard:**
```
https://www.runpod.io/console/serverless
```

### Co když něco nefunguje:

1. **"Failed to upload"**
   - Zkontrolujte R2 credentials v RunPodu
   - Ověřte, že bucket existuje

2. **"Processing stuck"**
   - Podívejte se na RunPod logs
   - Zkontrolujte, že worker má dost paměti

3. **"Cannot connect"**
   - Ověřte API klíče ve frontendu
   - Zkontrolujte Endpoint ID

---

## 💰 Ceny

- **10 min video**: ~$0.08 (RunPod GPU)
- **Storage**: ~$0.015/GB/měsíc (R2)
- **Download**: ZDARMA (R2 advantage!)

---

## 🎯 Hotovo!

Máte plně funkční serverless pipeline pro zpracování videí libovolné velikosti! 🎉

**Potřebujete pomoc?** Podívejte se na:
- `DEPLOYMENT_README.md` - detailní dokumentace
- `test_local.py` - testování komponent
- RunPod logs - debugging
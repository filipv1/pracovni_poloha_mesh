# ⚠️ Kontrola API klíčů

## RunPod API Key - POTŘEBA OVĚŘIT

Poskytnutý RunPod API klíč **nevrací platnou odpověď**:
```
rpa_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Možné důvody:
1. **API klíč je neplatný nebo expirovaný**
2. **Chybný formát** (možná chybí část klíče)
3. **Účet není aktivní**

### Jak získat správný API klíč:

1. **Přihlaste se na RunPod.io**
   - https://www.runpod.io/console/user/settings

2. **Jděte do Settings → API Keys**

3. **Vytvořte nový API key:**
   - Klikněte "Create API Key"
   - Pojmenujte ho "Flask App"
   - Zkopírujte CELÝ klíč (začíná obvykle `rp_` nebo `rpa_`)

4. **Aktualizujte .env:**
   ```env
   RUNPOD_API_KEY=váš_nový_api_klíč_zde
   ```

## Cloudflare R2 - NEKOMPLETNÍ

Máte pouze:
- ✅ Account ID: `605252007a9788aa8b697311c0bcfec6`
- ✅ Access Key ID: `FdX1lJ9yo8GNE-3NvPKpcoL1dW_3sXzSF6VEqbcQ`
- ❌ **Chybí Secret Access Key**

### Jak získat Secret Access Key:

1. **Cloudflare Dashboard**
   - https://dash.cloudflare.com/

2. **R2 → Manage R2 API Tokens**

3. **Vytvořte nový token** (pokud nemáte secret key):
   - Create API token
   - Permissions: Object Read & Write
   - Uložte si **Secret Access Key** (zobrazí se pouze jednou!)

4. **Doplňte do .env:**
   ```env
   R2_SECRET_ACCESS_KEY=váš_secret_key_zde
   ```

## Email - POTŘEBA GMAIL APP PASSWORD

Pro email notifikace potřebujete:
- ✅ Username: `vaclavik.renturi@gmail.com`
- ❌ **Chybí App Password**

### Jak získat Gmail App Password:

1. **Ujistěte se, že máte zapnuté 2FA**
   - https://myaccount.google.com/security

2. **Vytvořte App Password**
   - https://myaccount.google.com/apppasswords
   - Vyberte "Mail" → "Other"
   - Název: "Flask RunPod"
   - Zkopírujte 16místné heslo

3. **Doplňte do .env:**
   ```env
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx  # Bez mezer!
   ```

## 🔴 DŮLEŽITÉ: Co potřebujete udělat

1. **RunPod**: Získat platný API klíč a Pod ID
2. **Cloudflare R2**: Doplnit Secret Access Key
3. **Gmail**: Vytvořit App Password

## Test po doplnění

Po doplnění všech údajů:

```bash
# Test RunPod
python test_runpod_api.py

# Restart aplikace
python app.py
```

Pak v prohlížeči:
- http://localhost:5000/api/test/runpod
- http://localhost:5000/api/test/storage
- http://localhost:5000/api/test/email

---

**Aplikace funguje i bez těchto klíčů**, ale pouze v lokálním režimu bez GPU processingu, cloud storage a emailů.
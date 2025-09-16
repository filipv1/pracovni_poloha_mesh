# Configuration Status

## ✅ Co je již nastaveno v .env:

### RunPod
- **API Key**: `[CONFIGURED]` ✅

### Cloudflare R2
- **Account ID**: `605252007a9788aa8b697311c0bcfec6` ✅
- **Access Key ID**: `FdX1lJ9yo8GNE-3NvPKpcoL1dW_3sXzSF6VEqbcQ` ✅  
- **Bucket Name**: `flaskrunpod` ✅

### Email
- **SMTP Server**: `smtp.gmail.com` ✅
- **Username**: `vaclavik.renturi@gmail.com` ✅

## ❌ Co je potřeba doplnit:

### 1. RunPod Pod ID
```env
RUNPOD_POD_ID=your-pod-id-here
```
**Kde najít**: 
- Přihlaste se do RunPod dashboard
- Najděte váš existující pod nebo vytvořte nový
- Pod ID je v URL nebo v detailech podu

### 2. Cloudflare R2 Secret Access Key
```env
R2_SECRET_ACCESS_KEY=your-secret-key-here
```
**Kde najít**:
- Cloudflare dashboard → R2 → Manage R2 API tokens
- Secret key jste dostali při vytváření API tokenu
- Pokud jej nemáte, vytvořte nový token

### 3. Gmail App Password
```env
SMTP_PASSWORD=your-16-char-app-password
```
**Jak získat**:
1. Jděte na: https://myaccount.google.com/apppasswords
2. Vyberte "Mail" a "Other (Custom name)"
3. Zadejte název: "Flask RunPod App"
4. Zkopírujte 16místné heslo (bez mezer)

### 4. (Volitelné) Resend API Key
```env
RESEND_API_KEY=re_xxxxx
```
**Pokud máte Resend účet** (jako backup pro email)

## 📝 Rychlé instrukce:

1. **Otevřete `.env` soubor** v editoru
2. **Doplňte chybějící hodnoty** označené výše
3. **Uložte soubor**
4. **Restartujte aplikaci**:
   ```bash
   # Zastavte aplikaci (Ctrl+C)
   # Spusťte znovu
   python app.py
   ```

## 🧪 Test konfigurace:

Po doplnění všech údajů otestujte:

### Test RunPod:
```bash
curl http://localhost:5000/api/test/runpod
```

### Test R2 Storage:
```bash
curl http://localhost:5000/api/test/storage
```

### Test Email:
```bash
curl http://localhost:5000/api/test/email
```

## 📊 Aktuální stav služeb:

| Služba | Status | Poznámka |
|--------|--------|----------|
| Flask App | ✅ Běží | http://localhost:5000 |
| Database | ✅ Funkční | SQLite s 10 uživateli |
| RunPod | ⚠️ Částečně | Chybí Pod ID |
| Cloudflare R2 | ⚠️ Částečně | Chybí Secret Key |
| Email | ⚠️ Částečně | Chybí App Password |
| Job Queue | ✅ Funkční | FIFO processing |
| SSE Updates | ✅ Funkční | Real-time progress |

---

**Poznámka**: Aplikace funguje i bez těchto API klíčů v lokálním testovacím režimu. Pro plnou funkcionalitu (GPU processing, cloud storage, email notifikace) je potřeba doplnit všechny údaje.
# Flask RunPod Application - Setup Instructions

## ✅ Aplikace je připravena!

Aplikace byla úspěšně implementována a otestována. Nyní je potřeba dokončit konfiguraci.

## 🔧 Kroky k dokončení

### 1. Nainstalujte závislosti (pokud ještě nejsou)

```bash
cd flask_runpod_app
pip install Flask Flask-Login Flask-SQLAlchemy boto3 python-dotenv
```

### 2. Vytvořte a nakonfigurujte .env soubor

```bash
# Zkopírujte příklad
copy .env.example .env

# Otevřete .env v editoru a doplňte:
```

**Minimální konfigurace pro lokální testování:**
```env
FLASK_SECRET_KEY=your-very-secret-key-change-this
DATABASE_URL=sqlite:///app.db
```

**Pro plnou funkcionalitu doplňte:**

#### RunPod (volitelné - pro GPU processing):
```env
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_POD_ID=your-pod-id
```

#### Cloudflare R2 (volitelné - pro cloud storage):
```env
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=pose-analysis-files
```

#### Email notifikace (volitelné):
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### 3. Spusťte aplikaci

```bash
python app.py
```

Aplikace poběží na: **http://localhost:5000**

## 👤 Přihlašovací údaje

| Uživatel | Heslo | Role |
|----------|-------|------|
| **admin** | admin123 | Admin |
| **demo** | demo123 | User |
| user1-8 | user123 | User |

## 📁 Struktura souborů

```
flask_runpod_app/
├── app.py                    # Hlavní aplikace
├── models.py                 # Databázové modely
├── config.py                 # Konfigurace
├── auth.py                   # Autentizace
├── core/                     # Core moduly
│   ├── runpod_client.py     # RunPod komunikace
│   ├── storage_client.py    # R2 storage
│   ├── job_processor.py     # Queue processor
│   ├── email_service.py     # Email notifikace
│   └── progress_tracker.py  # Real-time updates
├── templates/               # HTML šablony
├── static/                  # CSS a JavaScript
├── runpod_scripts/         # GPU processing skripty
└── uploads/                # Dočasné soubory

```

## 🚀 Funkce aplikace

1. **Upload videí** - Drag & drop interface pro MP4 (max 5GB)
2. **Processing queue** - FIFO fronta pro zpracování
3. **Real-time progress** - SSE pro živé sledování průběhu
4. **Download výsledků** - PKL mesh data + XLSX analýza
5. **Admin dashboard** - Přehled systému a logů
6. **Email notifikace** - Automatické upozornění po dokončení

## ⚙️ Režimy provozu

### Lokální režim (bez API klíčů)
- ✅ Upload a správa souborů
- ✅ Queue management
- ✅ Základní processing (simulace)
- ❌ GPU processing
- ❌ Cloud storage
- ❌ Email notifikace

### Plný režim (s API klíči)
- ✅ Vše z lokálního režimu
- ✅ RunPod GPU processing
- ✅ Cloudflare R2 storage
- ✅ Email notifikace
- ✅ 7denní retence souborů

## 🐛 Řešení problémů

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Database error"
```bash
# Smazat starou databázi a vytvořit novou
del app.db
python -c "from app import initialize_app; initialize_app()"
```

### Port již používán
```bash
# Změnit port v app.py na řádku:
app.run(debug=True, host='0.0.0.0', port=5001)  # Změnit na jiný port
```

## 📊 Testování

Spusťte test script:
```bash
python run_test.py
```

## 🌐 Nasazení na produkci

Pro nasazení na Render.com:
1. Pushněte kód na GitHub
2. Propojte s Render.com
3. Nastavte environment proměnné
4. Deploy!

---

**Aplikace je plně funkční a připravena k použití!**
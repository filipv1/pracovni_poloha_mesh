# Lokální testování bez API klíčů

## ✅ Co funguje v lokálním režimu:

- **Upload videí** - soubory se ukládají do `uploads/`
- **Queue management** - FIFO fronta funguje
- **Progress tracking** - SSE real-time updates
- **Simulace processingu** - vytvoří dummy PKL a XLSX
- **UI/UX** - celý frontend funguje
- **Admin dashboard** - statistiky a logy

## ❌ Co nefunguje bez API klíčů:

- **Skutečné GPU processing** - potřeba RunPod
- **Cloud storage** - potřeba Cloudflare R2
- **Download souborů** - pouze s R2 nebo lokální úpravou
- **Email notifikace** - potřeba SMTP

## 🔧 Jak zprovoznit download v lokálním režimu:

Pokud chcete testovat download bez Cloudflare R2, můžete přidat endpoint pro lokální soubory:

```python
# Přidat do app.py (kolem řádku 280 po ostatních routes):

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    """Download local file (for testing without R2)"""
    import os
    from flask import send_from_directory
    
    # Security check - only allow files from outputs folder
    safe_path = os.path.join('outputs', filename)
    if os.path.exists(safe_path):
        return send_from_directory('outputs', filename, as_attachment=True)
    else:
        abort(404)
```

Pak upravit job_processor.py aby ukládal do `outputs/` místo temp.

## 📝 Testovací workflow:

1. **Login**: admin / admin123 ✅
2. **Upload video**: Drag & drop MP4 ✅ 
3. **Sledovat progress**: Real-time updates ✅
4. **Historie**: Zobrazí dokončené jobs ✅
5. **Download**: Nefunguje bez R2 ⚠️

## 🚀 Pro plnou funkcionalitu:

### Minimální setup (doporučeno):
1. **Cloudflare R2** (zdarma do 10GB)
   - Registrace na cloudflare.com
   - Vytvořit R2 bucket
   - Získat API credentials
   - Přidat do .env

### Volitelné:
2. **RunPod** pro skutečné GPU processing
3. **Email** pro notifikace

---

**Aplikace funguje správně!** Lokální režim je perfektní pro:
- Testování UI/UX
- Vývoj nových features
- Debugging
- Demo prezentace
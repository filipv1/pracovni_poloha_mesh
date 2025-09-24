# ⚠️ FIX R2 CORS - MUSÍŠ UDĚLAT V CLOUDFLARE!

## PROBLÉM:
R2 bucket blokuje upload z browseru (CORS)

## ŘEŠENÍ:

### 1. Jdi do CloudFlare Dashboard
https://dash.cloudflare.com

### 2. Najdi R2 → tvůj bucket "ergonomic-analysis"

### 3. Klikni na "Settings" → "CORS"

### 4. Přidej tuto CORS policy:

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

### 5. Klikni "Save"

## ALTERNATIVA - Použij proxy pro všechno:

Pokud nechceš nastavovat CORS, můžeme udělat proxy i pro R2 upload.
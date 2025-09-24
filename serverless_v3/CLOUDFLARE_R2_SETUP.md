# CloudFlare R2 Setup - Detailní návod

## 1. Získání Account ID

### Způsob A - Přes R2 Dashboard:
1. Přihlaste se na: https://dash.cloudflare.com
2. V levém menu klikněte na **R2** (může být pod "Storage" nebo "Developer Platform")
3. **Account ID** uvidíte hned nahoře na stránce
4. Zkopírujte ho (vypadá jako: `a1b2c3d4e5f6...`)

### Způsob B - Přes Account Settings:
1. Klikněte na své jméno vpravo nahoře
2. Vyberte **"Manage Account"**
3. Account ID je zobrazen na stránce

### Způsob C - Přímý odkaz:
Jděte na: https://dash.cloudflare.com/?to=/:account/r2
Account ID bude v URL nebo na stránce

## 2. Aktivace R2 (pokud ještě není aktivní)

1. Jděte na: https://dash.cloudflare.com/sign-up/r2
2. Klikněte **"Get Started"** nebo **"Enable R2"**
3. R2 je ZDARMA pro první 10 GB storage a 1 milion požadavků

## 3. Vytvoření Bucketu

1. V R2 dashboardu klikněte **"Create bucket"**
2. **Název**: `ergonomic-analysis`
3. **Location**: Automatic (nechte defaultní)
4. Klikněte **"Create bucket"**

## 4. Získání API Credentials

1. V R2 dashboardu klikněte **"Manage R2 API Tokens"**
2. Klikněte **"Create API Token"**
3. Vyplňte:
   - **Token name**: `ergonomic-v3`
   - **Permissions**: `Object Read & Write`
   - **Specify bucket**: `ergonomic-analysis`
   - **TTL**: ponechte prázdné (neomezená platnost)
4. Klikněte **"Create API Token"**
5. **DŮLEŽITÉ**: Okamžitě zkopírujte:
   - **Access Key ID**: `xxx...`
   - **Secret Access Key**: `yyy...`
   - Tyto údaje se zobrazí POUZE JEDNOU!

## 5. Shrnutí - Co potřebujete

Pro setup_wizard.py budete potřebovat:

```
R2_ACCOUNT_ID=        # Z kroku 1
R2_ACCESS_KEY_ID=     # Z kroku 4
R2_SECRET_ACCESS_KEY= # Z kroku 4
R2_BUCKET_NAME=ergonomic-analysis
```

## Troubleshooting

### "R2 není v menu"
- R2 musíte nejdřív aktivovat na: https://dash.cloudflare.com/sign-up/r2
- Je zdarma pro malé projekty

### "Nevidím Account ID"
- Zkuste: https://dash.cloudflare.com/?to=/:account/settings
- Nebo v URL adrese hledejte část za `/account/`

### "API Token nefunguje"
- Ověřte, že má permissions: Object Read & Write
- Ověřte, že je pro správný bucket
- Zkuste vytvořit nový token

## Alternativa - AWS S3

Pokud preferujete AWS S3:
1. Ve wizardu zvolte možnost 2
2. Budete potřebovat AWS účet
3. S3 má poplatky za egress ($0.09/GB)

## Kontakt

Pokud máte problémy, vytvořte issue na GitHubu nebo použijte AWS S3 jako alternativu.
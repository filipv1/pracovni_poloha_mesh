# Řešení RunPod Throttled Workers

## Problém
Všechny workery jsou ve stavu "throttled" - RunPod omezuje provoz kvůli:
1. Docker Hub rate limit při stahování image
2. Příliš mnoho restartů workerů
3. Chyby při spouštění kontejnerů

## Okamžité řešení

### 1. Spusťte diagnostiku:
```bash
cd serverless\simple-web
python test_runpod_direct.py
```

### 2. V RunPod konzoli:
1. Jděte na: https://www.runpod.io/console/serverless
2. Klikněte na váš endpoint (dfcn3rqntfybuk)
3. Zkontrolujte záložku **"Logs"** pro chyby

### 3. Restartujte endpoint:
V RunPod konzoli:
1. Klikněte **"Edit Endpoint"**
2. Změňte **Min Workers** na 0
3. Klikněte **"Save"**
4. Počkejte 30 sekund
5. Změňte **Min Workers** zpět na 1
6. Klikněte **"Save"**

### 4. Alternativní Docker image:
Pokud je problém s Docker Hub rate limit:

V RunPod konzoli změňte Docker image na:
- `runpod/base:0.4.0-cuda11.8.0` (základní image bez našeho kódu)

Pak zkuste znovu nahrát náš image později.

## Dlouhodobé řešení

### Použijte GitHub Container Registry:
```bash
# 1. Vytvořte GitHub Personal Access Token
# https://github.com/settings/tokens
# Potřebuje: write:packages

# 2. Přihlaste se
docker login ghcr.io -u YOUR_GITHUB_USERNAME

# 3. Přetagujte image
docker tag vaclavikmasa/ergonomic-analyzer:latest ghcr.io/YOUR_USERNAME/ergonomic-analyzer:latest

# 4. Nahrajte
docker push ghcr.io/YOUR_USERNAME/ergonomic-analyzer:latest

# 5. V RunPod použijte novou adresu:
# ghcr.io/YOUR_USERNAME/ergonomic-analyzer:latest
```

## Status workerů:
- **idle**: Připraven k práci ✅
- **running**: Zpracovává úlohu ⚙️
- **throttled**: Omezen/Pozastaven ⚠️
- **initializing**: Startuje 🔄

## Časté příčiny throttling:
1. **Docker pull rate limit** - počkejte 6 hodin nebo použijte jiný registry
2. **Příliš mnoho chyb** - zkontrolujte logy
3. **Nedostatek paměti** - snižte počet workerů
4. **Chybějící dependencies** - zkontrolujte Dockerfile

## Testování bez RunPodu:
Pokud RunPod nefunguje, můžete testovat lokálně:

```bash
cd serverless\runpod
python test_handler_with_mock.py
```
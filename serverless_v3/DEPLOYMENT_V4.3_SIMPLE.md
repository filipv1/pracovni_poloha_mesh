# V4.3 - Nasazení bez generování videí

## Rychlé nasazení:

```bash
cd serverless_v3
build_v4.3.bat
```

Pak v RunPod konzoli aktualizuj endpoint na **v4.3**.

## Co se změnilo:

- **Odstraněno:** Generování 4 videí (problém s PKL formátem)
- **Výstupy:** PKL, CSV, XLSX
- **Rychlejší:** Zpracování bez videí je rychlejší

## Výsledné soubory ke stažení:

1. **mesh.pkl** - 3D mesh data
2. **angles.csv** - Úhlová data
3. **ergonomic_analysis.xlsx** - Ergonomická analýza

## Testování:
1. Spusť: `python full_proxy.py`
2. Otevři: `frontend/index-with-proxy.html`
3. Nahraj video a stáhni 3 výstupní soubory
# Plán Implementace: Paralelní Zpracování 3D Meshe

**Datum:** 22. 9. 2025
**Autor:** Gemini
**Cílový soubor:** `run_production_simple_p.py`

## 1. Shrnutí a Cíl

Tento dokument popisuje krok-za-krokem plán pro implementaci nové, vysoce výkonné pipeline pro generování 3D meshů z videa. Cílem je vytvořit nový soubor `run_production_simple_p.py`, který bude využívat paralelní zpracování pro maximální rychlost při zachování vysoké kvality a plynulosti animace.

**Zvolená architektura:**
Po důkladné analýze byla zvolena dvou-fázová architektura, která je nejlepším kompromisem mezi jednoduchostí implementace, rychlostí a kvalitou:

1.  **Fáze 1: Sekvenční detekce s interním vyhlazováním MediaPipe.** V této fázi se rychle projdou všechny snímky videa a extrahují se 3D landmarky. Využijeme vestavěného a vysoce optimalizovaného časového vyhlazování v MediaPipe, abychom získali plynulou a kompletní sekvenci koster.
2.  **Fáze 2: Masivně paralelní SMPL-X Fitting.** S již vyhlazenými a připravenými daty můžeme spustit výpočetně náročný fitting SMPL-X modelu pro všechny snímky najednou v jediné dávce na GPU.

**Odůvodnění:** Tento přístup je nejefektivnější, protože:
*   **Nevyžaduje implementaci složitých algoritmů** (Kalmanovy filtry, interpolace).
*   Využívá **specializované a optimalizované funkce** knihovny MediaPipe.
*   **Odděluje zodpovědnosti:** MediaPipe řeší tracking, SMPL-X řeší fitting.
*   Umožňuje **plnou paralelizaci** nejpomalejší části procesu.

---

## 2. Architektonické změny a Refaktoring Kódu

Než začneme psát novou logiku, je nutné upravit stávající třídy, aby podporovaly dávkové zpracování. Všechny změny budeme provádět v novém souboru `run_production_simple_p.py`.

### Krok 2.1: Úprava třídy `HighAccuracySMPLXFitter`

**Cíl:** Přetvořit fitter na bezstavovou (stateless) komponentu schopnou zpracovat dávku (batch) snímků najednou.

*   **Odstranit časovou logiku:**
    *   Z konstruktoru (`__init__`) odeberte atributy `self.param_history`, `self.max_history` a `self.temporal_alpha`.
*   **Upravit `fit_mesh_to_landmarks`:**
    *   Metoda již nebude přijímat jeden `target_joints`, ale dávku: `target_joints_batch`. Tento tensor bude mít tvar `(N, num_joints, 3)`, kde `N` je počet snímků v dávce.
    *   V inicializaci parametrů (`body_pose`, `global_orient`, `transl`, `betas`) kompletně odstraňte `if/else` blok pro `param_history`. Vždy použijte "studený start" (`torch.zeros(...)`), ale s novým rozměrem pro dávku, např. `body_pose = torch.zeros((N, 63), ...)`.
    *   V výpočtu chybové funkce (`total_loss`) kompletně odstraňte výpočet a přičítání `temporal_loss`.
*   **Upravit inicializaci SMPL-X modelu:**
    *   Konstruktor `__init__` třídy `HighAccuracySMPLXFitter` musí přijímat nový argument `batch_size`.
    *   Při vytváření instance `smplx.SMPLX` předejte tento `batch_size`, aby byl model připraven na dávkové zpracování: `self.smplx_model = smplx.SMPLX(..., batch_size=batch_size)`.
*   **Upravit návratovou hodnotu:**
    *   Metoda `fit_mesh_to_landmarks` nyní vrátí dávku výsledků. Místo jednoho slovníku `mesh_result` vrátí seznam slovníků, jeden pro každý snímek v dávce.

### Krok 2.2: Úprava třídy `MasterPipeline`

**Cíl:** Přestavět hlavní třídu tak, aby orchestrovala novou dvou-fázovou pipeline.

*   **Přejmenovat hlavní metodu:** Přejmenujte `execute_full_pipeline` na `execute_parallel_pipeline`, aby byl zřejmý její účel.
*   **Rozdělit logiku:** Vytvořte dvě nové privátní metody, které budou reprezentovat jednotlivé fáze:
    1.  `_phase1_detect_and_smooth_landmarks(self, video_path, max_frames, frame_skip)`
    2.  `_phase2_fit_parallel_mesh(self, all_landmarks)`
*   Metoda `execute_parallel_pipeline` bude tyto dvě metody postupně volat a zpracovávat jejich výstupy.

---

## 3. Detailní Implementace Krok za Krokem

### Krok 3.1: Implementace Fáze 1 - `_phase1_detect_and_smooth_landmarks`

**Cíl:** Získat plynulou sekvenci 3D landmarků z videa.

1.  **Inicializace MediaPipe:**
    *   Vytvořte instanci `mp.solutions.pose.Pose`.
    *   Nastavte parametry pro maximální kvalitu a plynulost:
        ```python
        pose = self.mp_pose.Pose(
            static_image_mode=False,      # Důležité pro tracking mezi snímky
            smooth_landmarks=True,        # Explicitně zapíná vestavěné vyhlazování
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5, # Mírně snížíme pro zachycení více póz
            min_tracking_confidence=0.5
        )
        ```
2.  **Zpracování videa:**
    *   Otevřete video soubor pomocí `cv2.VideoCapture`.
    *   Vytvořte prázdný seznam `all_landmarks = []`.
    *   Spusťte `while` smyčku pro čtení snímků z videa.
    *   V každé iteraci volejte `results = pose.process(rgb_frame)`.
    *   Přidejte `results.pose_world_landmarks` do seznamu `all_landmarks`. Pokud MediaPipe na snímku nic nedetekuje, `pose_world_landmarks` bude `None`, což je v pořádku.
3.  **Návratová hodnota:**
    *   Metoda vrátí seznam `all_landmarks`.

### Krok 3.2: Implementace Fáze 2 - `_phase2_fit_parallel_mesh`

**Cíl:** Zpracovat dávku vyhlazených landmarků a vygenerovat 3D meshe.

1.  **Příprava dat:**
    *   Vstupem je seznam `all_landmarks` z Fáze 1.
    *   Vytvořte nový seznam `valid_landmarks` a `original_indices`, který bude obsahovat pouze ty landmarky, které nejsou `None`. Zároveň si uložte jejich původní indexy, abyste mohli později rekonstruovat celou sekvenci.
    *   Pokud je `valid_landmarks` prázdný, ukončete zpracování.
    *   Použijte `PreciseMediaPipeConverter` v smyčce k převedení každého prvku v `valid_landmarks` na pole kloubů SMPL-X.
    *   Všechny výsledné pole kloubů spojte do jednoho velkého NumPy pole a následně převeďte na PyTorch tensor `target_joints_batch` s tvarem `(len(valid_landmarks), num_joints, 3)`.
2.  **Inicializace Fitteru:**
    *   Získejte velikost dávky: `batch_size = len(valid_landmarks)`.
    *   Vytvořte instanci upravené třídy `HighAccuracySMPLXFitter` a předejte jí `batch_size`.
3.  **Spuštění Fittingu:**
    *   Zavolejte `list_of_mesh_results = self.mesh_fitter.fit_mesh_to_landmarks(target_joints_batch)`.
4.  **Rekonstrukce sekvence:**
    *   Vytvořte finální `mesh_sequence` o původní délce videa, naplněnou hodnotami `None`.
    *   Pomocí `original_indices` vložte výsledky z `list_of_mesh_results` na správná místa v `mesh_sequence`.
5.  **Návratová hodnota:**
    *   Metoda vrátí kompletní `mesh_sequence`.

### Krok 3.3: Spojení v `execute_parallel_pipeline`

**Cíl:** Orchestrovat celý proces.

1.  Zavolejte `all_landmarks = self._phase1_detect_and_smooth_landmarks(...)`.
2.  Zavolejte `mesh_sequence = self._phase2_fit_parallel_mesh(all_landmarks)`.
3.  Pokud je `mesh_sequence` validní, pokračujte stávající logikou pro:
    *   Uložení výsledků do `.pkl` souboru.
    *   Uložení statistik.
    *   Vytvoření finálního videa pomocí `self.visualizer.create_professional_video(mesh_sequence, ...)`.
4.  Vraťte finální slovník s výsledky.

### Krok 3.4: Úprava `main` funkce

*   Vytvořte kopii souboru `run_production_simple.py` a pojmenujte ji `run_production_simple_p.py`.
*   V `main` funkci v novém souboru změňte volání `pipeline.execute_full_pipeline(...)` na `pipeline.execute_parallel_pipeline(...)`.
*   Proveďte všechny výše popsané úpravy tříd v tomto novém souboru.

---

Tento plán poskytuje pevný základ pro úspěšnou a efektivní implementaci. Výsledný kód bude nejen výrazně rychlejší, ale také čistší a lépe strukturovaný.

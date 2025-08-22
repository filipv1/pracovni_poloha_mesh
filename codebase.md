# 📚 CODEBASE DOCUMENTATION - pracovni_poloha_mesh
**Last Updated:** 2025-08-22  
**Current Commit:** 1d3f984 (Fix pracovni_poloha2 integration and add mesh analysis tools)

## 🎯 Project Overview

**Cíl projektu:** Vytvoření 3D human mesh modelu z 2D videa pomocí MediaPipe a SMPL-X pro analýzu pracovní polohy.

**Vstup:** MP4 video jednoho člověka při práci  
**Výstup:** 3D mesh vizualizace pohybu člověka v čase (video nebo frames)

## 🏗️ Architecture

### Core Pipeline Flow
```
Input Video → MediaPipe (33 3D landmarks) → SMPL-X Fitting → 3D Mesh → Visualization
```

### Main Components

#### 1. **production_3d_pipeline_clean.py** - Hlavní pipeline
- `MasterPipeline` class - orchestrátor celého procesu
- Podporuje CPU i GPU (CUDA)
- Quality modes: ultra/high/medium
- Batch processing capability

#### 2. **PreciseMediaPipeConverter** - MediaPipe → SMPL-X konverze
- Mapuje 33 MediaPipe bodů na 22 SMPL-X joints
- Anatomicky přesné mapování s confidence weights
- Temporal smoothing pro plynulé přechody

#### 3. **EnhancedSMPLXFitter** - SMPL-X mesh fitting
- Multi-stage optimization (3 fáze):
  1. Global pose & translation (80 iterací)
  2. Body pose fine-tuning (100 iterací)  
  3. Final refinement (70 iterací)
- Adam optimizer s adaptive learning rates
- Temporal consistency pomocí parameter history

#### 4. **ProfessionalVisualizer** - Vizualizace
- Open3D pro 3D rendering (ray-tracing kvalita)
- Matplotlib pro 3D animace
- Dual visualization (standalone + overlay)

## 🛠️ Environment Setup

### Conda Environment
```bash
conda activate trunk_analysis
```

**Python Version:** 3.9.23

### Key Dependencies
- **torch** >= 2.0.0 (PyTorch pro SMPL-X)
- **smplx** >= 0.1.28 (SMPL-X body models)
- **mediapipe** >= 0.10.8 (pose detection)
- **open3d** >= 0.18.0 (3D visualization)
- **opencv-python** >= 4.8.0 (video processing)
- **trimesh** >= 4.0.0 (mesh operations)
- **numpy**, **scipy**, **matplotlib**

### SMPL-X Models
Umístění: `models/smplx/`
- SMPLX_NEUTRAL.npz (10MB)
- SMPLX_MALE.npz (10MB)
- SMPLX_FEMALE.npz (10MB)
- Nutno stáhnout z: https://smpl-x.is.tue.mpg.de/

## 📁 Project Structure

```
pracovni_poloha_mesh/
├── production_3d_pipeline_clean.py    # Hlavní pipeline
├── quick_test_3_frames.py            # Rychlý test (3 frames)
├── fix_video_output.py                # Video output diagnostika
├── models/smplx/                     # SMPL-X modely
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
├── pracovni_poloha2/                 # MediaPipe trunk analysis
│   ├── main.py                       # Entry point
│   └── src/                          # Source moduly
├── requirements_runpod.txt           # GPU dependencies
├── test.mp4                          # Test video
└── vysledek.gif                      # Cílový výsledek (reference)
```

## 🚀 Usage

### Basic Test (3 frames)
```bash
conda activate trunk_analysis
python quick_test_3_frames.py
```

### Full Pipeline
```python
from production_3d_pipeline_clean import MasterPipeline

pipeline = MasterPipeline(
    smplx_path="models/smplx",
    device='cuda',  # nebo 'cpu'
    gender='neutral'
)

results = pipeline.execute_full_pipeline(
    'input_video.mp4',
    output_dir='output',
    max_frames=100,
    frame_skip=2,
    quality='high'
)
```

### Output Files
- `[video]_meshes.pkl` - Kompletní mesh data
- `[video]_3d_animation.mp4` - 3D animace
- `[video]_final_mesh.png` - Finální mesh render
- `sample_frame_*.png` - Individual frames

## 🐛 Known Issues

### 1. **FFmpeg Not Available on Windows**
- Video output nefunguje bez FFmpeg
- Řešení: Instalovat FFmpeg nebo použít OpenCV fallback
- Fallback implementace v `fix_video_output.py`

### 2. **Pipeline Timeout**
- Test může timeoutovat při video generování
- Důvod: Pomalé CPU zpracování (33s/frame na Intel GPU)
- Řešení: Použít GPU nebo snížit quality/frames

### 3. **Missing Video Overlay**
- Současná implementace generuje pouze standalone 3D mesh
- Chybí overlay na původní video (jako ve vysledek.gif)
- TODO: Implementovat mesh overlay funkci

## 🎯 TODO - Co zbývá implementovat

### Priority 1 - Video Output
- [ ] Opravit video generování na Windows
- [ ] Implementovat FFmpeg fallback
- [ ] Přidat progress bar pro dlouhé procesy

### Priority 2 - Mesh Overlay
- [ ] Implementovat overlay mesh na původní video
- [ ] Synchronizovat mesh s video frames
- [ ] Přidat alpha blending pro průhlednost

### Priority 3 - Optimalizace
- [ ] GPU batch processing
- [ ] Frame caching pro rychlejší zpracování
- [ ] Paralelní zpracování více frames

### Priority 4 - Rozšíření
- [ ] Multi-person support
- [ ] Hand/face tracking (SMPL-X podporuje)
- [ ] Real-time processing možnost

## 💡 Tips pro vývojáře

### Testování
1. Vždy používat `conda activate trunk_analysis`
2. Pro rychlý test: `python quick_test_3_frames.py`
3. Kontrolovat SMPL-X modely v `models/smplx/`

### Debugging
- MediaPipe landmarks: Zkontrolovat confidence hodnoty
- SMPL-X fitting: Sledovat loss hodnoty při optimalizaci
- Video output: Použít `fix_video_output.py` pro diagnostiku

### Performance
- CPU: ~33 sekund/frame
- GPU (RTX 4090): ~2-3 sekundy/frame
- Doporučeno: frame_skip=2 pro rychlejší zpracování

## 🔗 Related Documentation

- **README.md** - Uživatelská dokumentace
- **FINAL_IMPLEMENTATION_REPORT.md** - Technická zpráva
- **RUNPOD_DEPLOYMENT_GUIDE.md** - GPU deployment
- **zadani.txt** - Původní zadání projektu
- **vysledek.gif** - Reference výsledku

## 📝 Notes

- Projekt byl revertován z poškozeného stavu na commit 1d3f984
- Hlavní priorita: Implementovat mesh overlay na video (jako vysledek.gif)
- Framework volba: SMPL-X (lepší než EasyMocap pro náš use case)
- Vyhýbat se: ROMP, HyBrik (dependency hell)

## 🤝 Contributors

- Filip V. - Project owner
- Claude Code - Implementation assistance

---

**Pro další práci:** Začít s opravou video output a implementací mesh overlay funkce.
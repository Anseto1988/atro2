# AstroAI Suite — Technische Architektur

## 1. Tech-Stack Entscheidung

| Schicht | Technologie | Begruendung |
|---------|------------|------------|
| **UI** | PySide6 (Qt6) | Native cross-platform, GPU-beschleunigtes Display, direktes Python-Oekosystem ohne IPC-Overhead |
| **Backend/Core** | Python 3.12+ | Vollstaendiges wissenschaftliches Oekosystem (astropy, numpy, rawpy) |
| **AI Framework** | PyTorch 2.x | Bestes CUDA + MPS (Apple Metal) Support, aktive Community, ONNX-Export |
| **Bildverarbeitung** | astropy, rawpy, scipy | FITS/XISF, RAW-Formate (CR2/NEF/ARW) |
| **Build** | Poetry | Reproducible dependencies |
| **Distribution** | PyInstaller | Single-file bundles fuer Win/Mac/Linux |
| **CI/CD** | GitHub Actions | Matrix-Builds auf allen 3 Plattformen |

**Abgelehnte Alternativen:**
- Electron: zu schwer, JS-Overhead ungeeignet fuer GPU-intensive Workflows
- Tauri: IPC-Overhead zwischen Rust und Python AI-Backend zu komplex
- TensorFlow: schlechterer MPS/Metal-Support als PyTorch

## 2. Modulare Architektur

```
UI Layer (PySide6 Qt6)
     |  Qt Signals/Slots
Core Processing Pipeline
  |-- astroai/core/        (E1) IO, Kalibrierung
  |-- astroai/engine/      (E2) Stacking, Registration
  |-- astroai/inference/   (E2) Scoring, AI-Backends
  +-- astroai/processing/  (E3) Denoise, Stretch, Stars
     |  torch.device abstraction
GPU Backend Abstraction
  |-- CUDA (Nvidia)
  |-- MPS / Metal (Apple Silicon)
  +-- CPU Fallback
```

### Layer-Verantwortlichkeiten

**Core Engine** (astroai/core/) - Epic 1
- io/: FITS/XISF/RAW Lesen und Schreiben (astropy + rawpy)
- pipeline/: Abstraktes Pipeline-Interface mit Fortschritts-Callbacks
- calibration/: Metadaten-Matching, Dark/Flat-Anwendung

**KI-Inference Layer** (astroai/inference/) - Epic 2
- backends/: GPU-Abstraktion (cuda/mps/cpu)
- models/: Model-Wrapper mit Hot-Reload (ModelRegistry)
- scoring/: HFR-Berechnung, Sternrundheit, Wolken-Detektion

**Stacking Engine** (astroai/engine/) - Epic 2
- registration/: KI-basierte Bildregistrierung (strukturbasiert)
- stacking/: Sigma-Clipping, Mean/Median Kombination

**Processing Layer** (astroai/processing/) - Epic 3
- denoise/: Neural Denoising (U-Net basiert, ONNX)
- stretch/: Intelligente Histogramm-Streckung (MTF/STF)
- stars/: Starless-Trennung, Stern-Reduktion

**UI Layer** (astroai/ui/) - Epic 4
- main/: MainWindow + App-Einstiegspunkt
- widgets/: Custom Qt-Widgets (ImageViewer, HistogramWidget, WorkflowGraph)
- resources/: Dark-Theme QSS, Icons

## 3. Datenfluss

```
FITS/RAW Input
  -> core.io: Lesen + Normalisierung
  -> core.calibration: Dark/Flat Matching + Anwendung
  -> inference.scoring: KI Frame-Bewertung (HFR, Wolken)
  -> engine.registration: AI-Alignment
  -> engine.stacking: Frame-Kombination
  -> processing.denoise: Neural Denoising
  -> processing.stretch: Intelligent Stretch
  -> processing.stars: Star-Management (optional)
  -> Output: FITS/TIFF
```

## 4. CI/CD Grundgeruest

Workflow: .github/workflows/build.yml
- Trigger: push/PR auf main und develop
- Matrix: ubuntu-latest, windows-latest, macos-latest x python-3.12
- Test Stage: ruff lint -> mypy type-check -> pytest unit tests
- Build Stage (nur main): PyInstaller bundle -> Artifact upload
- GPU-Tests: Self-hosted Runners (Phase 2, nach MVP)

## 5. KI-Modell-Strategie

- Format: ONNX fuer plattformuebergreifende Inferenz
- Verteilung: Modelle NICHT im Git-Repo, Download bei Erststart
- Hot-Reload: ModelRegistry erlaubt Updates ohne App-Neustart
- Backend: torch.device Abstraktion -> automatisch cuda/mps/cpu

## 6. Epic Subtask-Delegation

| Epic | Titel | Assignee |
|------|-------|---------|
| E1 | Core Engine: FITS/RAW I/O + Kalibrierung + Pipeline | Backend Developer |
| E2 | KI-Inference: Frame Scoring + AI Alignment + Stacking | Full-Stack Developer |
| E3 | Neural Processing: Denoise + Stretch + Stars | Backend Developer |
| E4 | UI Layer: PySide6 MainWindow + Dark Theme + Workflow Graph | Frontend Developer |

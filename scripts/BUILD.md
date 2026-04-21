# AstroAI Suite — Windows Standalone Build

## Voraussetzungen

- Python 3.12 (64-bit)
- Poetry (`pip install poetry`)
- ~3 GB Festplattenspeicher für das Bundle (torch + PySide6 sind groß)
- Windows 10/11 x86_64

## Build-Schritte

```bash
# 1. Dependencies installieren
poetry install --with dev --sync

# 2. ASTAP-Binary herunterladen (Plate Solving Engine)
poetry run python scripts/download_astap.py

# 3. PyInstaller-Bundle erzeugen
poetry run pyinstaller scripts/astroai.spec --noconfirm
```

Das Bundle wird unter `dist/AstroAI/` erzeugt.

## Ergebnis

| Datei                  | Beschreibung                              |
|------------------------|-------------------------------------------|
| `dist/AstroAI/AstroAI.exe` | Hauptanwendung (GUI, kein Konsolenfenster) |
| `dist/AstroAI/_internal/`  | Gebündelte Libraries und Daten            |

Gesamtgröße: ~1.2 GB (hauptsächlich PyTorch + MKL + PySide6).

## Testen

```bash
# EXE starten — GUI-Fenster sollte sich öffnen
dist/AstroAI/AstroAI.exe
```

Erwartetes Verhalten: Hauptfenster öffnet sich ohne Import- oder DLL-Fehler.

## Bekannte Probleme

| Problem | Status | Workaround |
|---------|--------|------------|
| ASTAP-Binary 404 bei `download_astap.py` | Gelöst | GitHub-Release `Anseto1988/astap-bin v0.2.1` publiziert. URLs und SHA256-Checksummen in `astap_binary.py` aktualisiert. |
| MKL-DLL-Warnungen (mpich2mpi, impi, msmpi) | Unkritisch | Optionale MPI/HPC-Bibliotheken die im Desktop-Build nicht benötigt werden. |
| Bundle-Größe ~1.2 GB | Designbedingt | PyTorch + MKL. Kann durch `torch` CPU-only Wheel oder `--exclude` in der Spec reduziert werden. |

## CI/CD

Der Release-Workflow (`.github/workflows/release.yml`) führt den gleichen Build automatisch bei Tag-Push (`v*`) aus und erstellt ein ZIP-Artifact.

## Spec-Datei

Die PyInstaller-Konfiguration ist in `scripts/astroai.spec`:

- **Entry Point:** `astroai/ui/main/app.py`
- **Hidden Imports:** Alle dynamisch geladenen Module (Plate Solving, GPU Engine, Processing Steps, UI Widgets)
- **Data Files:** UI-Ressourcen (QSS Themes) + ASTAP-Binaries
- **Excludes:** tkinter, matplotlib, IPython, jupyter, pytest, mypy, ruff

1\. Produkt-Übersicht \& Zielsetzung

Die AstroAI Suite ist eine plattformübergreifende Software zur Verarbeitung von Astrofotografien. Sie differenziert sich durch "AI-First"-Workflows, die manuelle, fehleranfällige Schritte (wie Frame-Sortierung und Rauschentfernung) automatisieren.



2\. Kern-Features \& Funktionale Anforderungen

EPIC 1: Universelle Datenverarbeitung

FR-1.1: FITS/RAW Support: Native Unterstützung für .fits, .xisf, sowie gängige DSLR-RAW-Formate (CR2, NEF, ARW).



FR-1.2: Metadaten-Extraktion: Automatisches Auslesen von EXIF/FITS-Headern (Belichtungszeit, Gain/ISO, Temperatur).



EPIC 2: KI-gestütztes Stacking \& Kalibrierung

FR-2.1: Neural Frame Scoring: Ein KI-Modell bewertet Frames nach HFR (Sternschärfe), Rundheit der Sterne und Wolkenbedeckung.



FR-2.2: AI-Alignment: Registrierung von Bildern basierend auf struktureller Mustererkennung (funktioniert auch bei extremem Bildrauschen, wo Stern-Detektoren scheitern).



FR-2.3: Smart-Calibration: Automatisches Matching von Korrekturframes (Darks/Flats) zum Lightframe-Stack basierend auf Metadaten.



EPIC 3: Neural Processing (Post-Processing)

FR-3.1: AI-Denoise: Ein spezialisiertes neuronales Netz zur Entfernung von thermischem Rauschen und Photonenrauschen.



FR-3.2: AI-Star-Management: Funktionen zum Entfernen (Starless) oder Verkleinern von Sternen per Klick.



FR-3.3: Intelligent Stretch: Ein automatischer Algorithmus, der das Bild basierend auf dem Histogramm "streckt", um Details sichtbar zu machen, ohne das Rauschen zu verstärken.



EPIC 4: User Experience (UX)

FR-4.1: Modern UI: Dunkles Interface (Reduzierung von blauem Licht für die Nachtnutzung).



FR-4.2: Visual Workflow: Ein Fortschrittsbalken oder Graph, der zeigt, in welchem Stadium (Kalibrierung, Stacking, Processing) sich das Projekt befindet.



3\. Nicht-funktionale Anforderungen

Performance: GPU-Beschleunigung (CUDA für Nvidia, Metal für Apple Silicon) ist für die KI-Inferenz zwingend erforderlich.



Stabilität: Das Stacking von hunderten Gigabytes an Daten darf nicht zum Absturz führen (effizientes Memory-Management).



4\. Priorisierung (MoSCoW)

Must Have: FITS-Support, KI-Stacking, AI-Denoise, Basis-UI.



Should Have: Automatisches Kalibrierungs-Matching, Starless-Funktion.



Could Have: ~~Synthetische Flat-Generierung~~ ✅ (v2.3.0-alpha implementiert), Mobile App Companion.



Won't Have (v1.0): Cloud-Stacking (alles lokal auf dem Rechner des Users).



5\. Releases

| Version | Tag | Link | Features |
|---------|-----|------|----------|
| v1.1.0-alpha | v1.1.0-alpha | [GitHub Release](https://github.com/Anseto1988/atro2/releases/tag/v1.1.0-alpha) | E3–E6, Security S-01–S-04, Licensing API |
| v2.0.0-alpha | v2.0.0-alpha | [GitHub Release](https://github.com/Anseto1988/atro2/releases/tag/v2.0.0-alpha) | F-1 Plate Solving, F-2 GPU-Kalibrierung, F-3 Drizzle Super-Resolution, F-4 Mosaic-Workflow |
| v2.1.0-alpha | v2.1.0-alpha | [GitHub Release](https://github.com/Anseto1988/atro2/releases/tag/v2.1.0-alpha) | F-5 Photometrische Farbkalibrierung (SPCC), Fixes |
| v2.2.0-alpha | v2.2.0-alpha | [GitHub Release](https://github.com/Anseto1988/atro2/releases/tag/v2.2.0-alpha) | F-Comet Dual-Tracking Comet Stacking, Test-Suite 1278 Tests |
| v2.3.0-alpha | v2.3.0-alpha | [GitHub Release](https://github.com/Anseto1988/atro2/releases/tag/v2.3.0-alpha) | F-PipelineRunner/Builder, F-FullPipelineRun, F-RegistrationStep/StackingStep, F-FrameSelect, F-SynFlat, F-StarProcessing, F-DenoisePanel, F-StretchPanel, F-BackgroundRemoval, F-ExportPanel, F-LiveWorkflowTracking, F-FrameListPanel, F-ImageStats, F-LiveHistogram, F-ToneCurves, F-WcsHoverCoords, F-SessionNotes, F-FrameExposureImport, F-FrameContextMenu, F-FITSMetadata, F-ManualFrameReject, F-PreviewCompare, F-PipelineCancel — 2349 Tests, 99% Coverage |
| v2.5.0-alpha | v2.5.0-alpha | — | F-CalibMatchBatch, F-CalibScan, F-FrameExportStats, F-PipelinePreset, F-PresetUI, F-ProjectValidation, F-ProjectSummary, F-BuiltinPresets, F-SmartCalibUI, F-CalibAutoMatch, F-DragDropFrameImport, F-ShortcutsDialog, F-SavePreviewImage, F-FramePreviewOnClick, F-FrameNotesField, F-CalibStatusLabel, F-ImportFolderAction, F-RawDSLR, F-AIStarAlignment — ~3200 Tests, CI/CD |
| v2.6.0-alpha | v2.6.0-alpha | — | F-OnnxRegistry, F-StarNetTiling, F-CatalogCache, F-LivePreview, F-ProcessingHistory, F-Sharpening — +168 Tests, ~3348+ Gesamt |
| v2.7.0-alpha | — | — | F-NoiseEstimator (MAD-Sigma-Clipping, SNR, Auto-Detect), F-SelectiveColorSaturation (HSV, 7 Hue-Ranges, Qt-Panel), F-AdaptiveDenoise (NoiseEstimator-Pipeline-Step, Adaptive-Checkbox), F-WhiteBalance (R/G/B Multiplier, Qt-Panel), F-AsinHStretch (arcsinh Stretch, black_point, linked) — +270 Tests, ~3673+ Gesamt |
| v2.8.0-alpha | — | — | F-MTFStretch (Midtone Transfer Function, Auto-BTF), F-BackgroundNeutralization (Auto/ROI-Modus, per-Kanal), F-LocalContrastEnhancement (CLAHE pure numpy, Luminanz/pro-Kanal) — +208 Tests, ~3881+ Gesamt |


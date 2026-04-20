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



Could Have: Synthetische Flat-Generierung, Mobile App Companion.



Won't Have (v1.0): Cloud-Stacking (alles lokal auf dem Rechner des Users).


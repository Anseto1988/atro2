Project Brief: AstroAI Suite (Arbeitstitel)

Executive Summary

AstroAI Suite ist eine All-in-One Astrofotografie-Software, die die mächtigen Funktionen klassischer Tools wie Siril, DeepSkyStacker und AutoStakkert! mit modernster Künstlicher Intelligenz verbindet. Das Ziel ist es, den komplexen Workflow der Astrofotografie (Kalibrierung, Stacking, Post-Processing) durch neuronale Netze zu automatisieren und zu perfektionieren, während eine moderne, intuitive Benutzeroberfläche sowohl Anfänger als auch Profis anspricht.



Problemstellung

Aktuelle Software-Lösungen im Bereich Astrofotografie leiden unter folgenden Problemen:



Steile Lernkurve: Einsteiger sind von mathematischen Parametern und komplexen Workflows (z.B. in Siril) überfordert.



Veraltete UIs: Programme wie DeepSkyStacker oder AutoStakkert! nutzen veraltete Oberflächen, die den Workflow bremsen.



Fehleranfälliges Stacking: Satellitenspuren, Rauschen und schlechtes Seeing zerstören oft Stunden an Belichtungszeit, da herkömmliche Algorithmen an ihre Grenzen stoßen.



Fragmentierung: Nutzer müssen oft zwischen 3-4 Programmen wechseln, um ein fertiges Bild zu erhalten.



Lösungsvorschlag

Die AstroAI Suite bietet eine integrierte Plattform, die den gesamten Prozess abdeckt:



KI-gestütztes Stacking: Intelligente Frame-Bewertung und Registrierung durch neuronale Netze, die Strukturen besser erkennen als klassische Punkt-Algorithmen.



Neural Denoising \& Restoration: Ein natives KI-Modell, das spezifisch auf Astro-Daten trainiert wurde, um Rauschen zu entfernen und Details (Deconvolution) ohne Artefakte wiederherzustellen.



Automatisierte Kalibrierung: Intelligentes Dateimanagement, das Lights, Darks und Flats automatisch zuordnet und bei Bedarf synthetische Korrekturbilder berechnet.



Moderne UX: Ein konsistentes "Dark-Mode" Interface mit visuellen Workflow-Graphen.



Zielbenutzer

Einsteiger: Die durch einen "One-Click"-KI-Workflow schnell beeindruckende Ergebnisse sehen wollen.



Fortgeschrittene: Die volle Kontrolle über alle Parameter behalten möchten, aber KI-Tools zur Zeitersparnis und Qualitätssteigerung nutzen.



Planeten- \& Deep-Sky-Fotografen: Die eine einzige Software für alle Objekttypen benötigen.



MVP Scope (Mindestanforderungen)

Kern-Engine: Verarbeitung von FITS, RAW (DSLR) und Astro-Kamera Formaten.



Stacking-Modul: Basierend auf KI-Ausrichtung und Qualitätsprüfung.



Neural Denoise: Ein funktionierendes Modell zur Rauschunterdrückung.



Post-Processing Basics: Histogramm-Transformation, Streckung, Hintergrund-Extraktion.



UI: Grundlegendes modernes Layout mit Projektverwaltung.



Technische Überlegungen

Plattform: Cross-Plattform (Windows/Mac/Linux) – idealerweise mit GPU-Beschleunigung (CUDA/Metal) für die KI-Modelle.



Architektur: Modularer Aufbau, um KI-Modelle unabhängig von der Kern-Engine aktualisieren zu können.



KI-Stack: Integration von Frameworks wie PyTorch oder TensorFlow für die Inferenz der neuronalen Netze.



Rationale \& Entscheidungen

Warum YOLO-Modus? Wir haben die Vision im Brainstorming sehr klar definiert. Dieses Dokument fasst das "Was" und "Warum" prägnant zusammen, damit der Produktmanager (John) direkt die funktionalen Anforderungen (FRs) ableiten kann.



KI-Fokus: Ich habe sichergestellt, dass die KI nicht als "Filter", sondern als integraler Bestandteil des Stackings und Entrauschens definiert ist.


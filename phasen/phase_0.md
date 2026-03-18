# Phase 0 – Repo & Infrastruktur

**Dauer:** Tag 0

## Ziele

- [ ] Repository strukturieren
- [ ] Entwicklungsumgebung einrichten
- [ ] Abhängigkeiten installieren
- [ ] Qwen3-32B Base Model laden

## Aufgaben

### 1. Repository Setup
- Verzeichnisstruktur anlegen
- `.gitignore` konfigurieren
- `requirements.txt` oder `pyproject.toml` erstellen
- README.md mit Projektübersicht

### 2. Infrastruktur
- GPU-Zugriff verifizieren (NVIDIA H100 80 GB)
- CUDA-Treiber prüfen
- Python-Umgebung erstellen (empfohlen: conda/venv)

### 3. Framework Installation
- Axolotl oder Unsloth installieren
- PyTorch mit CUDA-Support
- Transformers-Bibliothek
- Weitere Dependencies (PEFT, Accelerate, etc.)

### 4. Base Model Laden
- Qwen3-32B von HuggingFace herunterladen
- Modell-Integrität prüfen
- Inference-Test durchführen

## Deliverables

- [ ] Funktionierende Entwicklungsumgebung
- [ ] Base Model verfügbar
- [ ] Erste Inference möglich

## Erfolgskriterien

- `import torch` funktioniert mit CUDA
- Modell kann geladen werden
- Einfache Prompt-Generierung läuft

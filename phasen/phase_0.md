# Phase 0 – Repo & Infrastruktur (RTX 3050 Optimiert)

**Dauer:** Tag 0

## Ziele

- [ ] Repository strukturieren
- [ ] Entwicklungsumgebung einrichten
- [ ] Abhängigkeiten installieren
- [ ] Kleines Qwen-Modell für Pipeline-Tests laden (0.6B–3B)

## Aufgaben

### 1. Repository Setup
- Verzeichnisstruktur anlegen
- `.gitignore` konfigurieren
- `requirements.txt` oder `pyproject.toml` erstellen
- README.md mit Projektübersicht

### 2. Infrastruktur
- GPU-Zugriff verifizieren (NVIDIA RTX 3050 4–8 GB)
- CUDA-Treiber prüfen
- Python-Umgebung erstellen (empfohlen: conda/venv)

### 3. Framework Installation
- PyTorch mit CUDA-Support
- Transformers-Bibliothek
- Weitere Dependencies (PEFT, Accelerate, bitsandbytes)
- Optional: llama.cpp für GGUF-Quantisierung

### 4. Base Model Laden (Klein für Testing)
- **Qwen3-0.6B** oder **Qwen3-1.7B** für Smoke Tests
- **Qwen2.5-3B-Instruct** für realistischere Tests
- Modell-Integrität prüfen
- Inference-Test durchführen

## Modell-Empfehlungen für RTX 3050

| Modell | Größe | Verwendung | Quantisierung |
|--------|-------|------------|---------------|
| Qwen3-0.6B | ~1.2 GB | Pipeline-Validierung | FP16/INT8 |
| Qwen3-1.7B | ~3.5 GB | Erste fachliche Tests | FP16/INT8 |
| Qwen2.5-3B-Instruct | ~6 GB | Realistische Tests | Q4_K_M GGUF |

## Deliverables

- [ ] Funktionierende Entwicklungsumgebung
- [ ] Kleines Test-Modell verfügbar
- [ ] Erste Inference möglich
- [ ] Pipeline validiert (Datenformat, Prompt-Template, Eval)

## Erfolgskriterien

- `import torch` funktioniert mit CUDA
- Modell kann geladen werden (< 4 GB VRAM)
- Einfache Prompt-Generierung läuft
- Epistemic Mode Detection funktioniert

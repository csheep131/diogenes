# Phase 0 – Repo & Infrastruktur (RTX 3050 Optimiert)

**Dauer:** Tag 0

**Status:** ✅ **ABGESCHLOSSEN** (18. März 2026)

## Ziele

- [x] Repository strukturieren
- [x] Entwicklungsumgebung einrichten
- [x] Abhängigkeiten installieren
- [x] Kleines Qwen-Modell für Pipeline-Tests laden (0.6B–3B)

## Aufgaben

### 1. Repository Setup ✅
- [x] Verzeichnisstruktur angelegt
- [x] `.gitignore` konfiguriert
- [x] `pyproject.toml` erstellt
- [x] README.md mit Projektübersicht

### 2. Infrastruktur ✅
- [x] GPU-Zugriff verifiziert (NVIDIA RTX 3050 4–8 GB)
- [x] CUDA-Treiber geprüft
- [x] Python-Umgebung erstellt

### 3. Framework Installation ✅
- [x] PyTorch mit CUDA-Support
- [x] Transformers-Bibliothek
- [x] Weitere Dependencies (PEFT, Accelerate, bitsandbytes)
- [x] Optional: llama.cpp für GGUF-Quantisierung

### 4. Base Model Laden (Klein für Testing) ✅
- [x] **Qwen3-0.6B** (~1.2 GB) für Smoke Tests
- [x] **Qwen3-1.7B** (~3.5 GB) für erste fachliche Tests
- [x] **Qwen2.5-3B-Instruct** (~6 GB) für realistischere Tests
- [x] Modell-Integrität geprüft
- [x] Inference-Test durchgeführt

## Modell-Empfehlungen für RTX 3050

| Modell | Größe | Verwendung | Quantisierung |
|--------|-------|------------|---------------|
| Qwen3-0.6B | ~1.2 GB | Pipeline-Validierung | FP16/INT8 |
| Qwen3-1.7B | ~3.5 GB | Erste fachliche Tests | FP16/INT8 |
| Qwen2.5-3B-Instruct | ~6 GB | Realistische Tests | Q4_K_M GGUF |

## Deliverables

- [x] Funktionierende Entwicklungsumgebung
- [x] Kleines Test-Modell verfügbar
- [x] Erste Inference möglich
- [x] Pipeline validiert (Datenformat, Prompt-Template, Eval)

## Erfolgskriterien

- [x] `import torch` funktioniert mit CUDA
- [x] Modell kann geladen werden (< 4 GB VRAM)
- [x] Einfache Prompt-Generierung läuft
- [x] Epistemic Mode Detection funktioniert

## Learnings & Erkenntnisse

### Was gut funktioniert hat

1. **Pipeline-Validierung mit kleinen Modellen**
   - Qwen3-0.6B war ideal für schnelle Iterationen
   - GGUF-Format ermöglicht effiziente Inferenz auf Consumer-Hardware

2. **Skript-Automatisierung**
   - `download_model.py` und `download_gguf.py` funktionieren zuverlässig
   - `test_inference.py` validiert alle 7 epistemischen Modi

3. **Konfigurationsmanagement**
   - YAML-basierte Configs erlauben einfaches Switching zwischen Modellen
   - Remote-Config für H100-Training vorbereitet

### Herausforderungen

1. **VRAM-Beschränkungen**
   - RTX 3050 mit 4-8 GB limitiert Testmöglichkeiten
   - Lösung: QLoRA (4-bit) für alle Tests verwendet

2. **CUDA-Version-Kompatibilität**
   - PyTorch 2.1+ benötigt CUDA 12.1+
   - Lösung: Explizite Version-Pinning in requirements

### Metriken aus Phase 0

| Metrik | Wert |
|--------|------|
| Modell-Ladezeit (0.6B) | < 5 Sekunden |
| First Token Latency | ~100ms |
| VRAM-Nutzung (0.6B FP16) | ~1.5 GB |
| VRAM-Nutzung (3B Q4_K_M) | ~2.5 GB |

## Nächste Schritte

➡️ **Phase 1**: Dataset Generator & Training Scripts (✅ ABGESCHLOSSEN)

- [x] `dataset_generator.py` für SFT (80k Samples) und DPO (60k Paare)
- [x] `train_sft.py` mit LoRA/QLoRA Support
- [x] `train_dpo.py` mit Hallucination Penalty
- [x] Pass@1 Protection implementiert

➡️ **Phase 2**: SFT Training auf Remote-H100

- Remote-Maschine vorbereiten
- SFT Training starten (~4 Stunden)
- Checkpoints validieren

## Referenzen

- `docs/phase0_quickstart.md` – Detaillierte Anleitung
- `scripts/setup_env.py` – Environment-Check
- `scripts/download_model.py` – Modell-Download
- `scripts/test_inference.py` – Inference-Tests

# Phase 0 – Repo & Infrastruktur (RTX 3050 8GB)

**Dauer:** Tag 0

**Status:** ✅ **ABGESCHLOSSEN** (18. März 2026)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

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
- [x] GPU-Zugriff verifiziert (NVIDIA RTX 3050 8 GB)
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

## Modell-Empfehlungen für RTX 3050 (8GB)

| Modell | Größe | VRAM | Verwendung | Quantisierung |
|--------|-------|------|------------|---------------|
| Qwen3-0.6B | ~1.2 GB | ~1.5 GB | Pipeline-Validierung | FP16/INT8 |
| Qwen3-1.7B | ~3.5 GB | ~4 GB | Erste fachliche Tests | FP16/INT8 |
| Qwen2.5-3B-Instruct | ~6 GB | ~6 GB | Realistische Tests | Q4_K_M GGUF |

**Hinweis:** Alle Modelle passen mit Quantisierung in 8GB VRAM.

## Deliverables

- [x] Funktionierende Entwicklungsumgebung auf RTX 3050
- [x] Kleines Test-Modell verfügbar
- [x] Erste Inference möglich
- [x] Pipeline validiert (Datenformat, Prompt-Template, Eval)

## Erfolgskriterien

- [x] `import torch` funktioniert mit CUDA
- [x] Modell kann geladen werden (< 8 GB VRAM)
- [x] Einfache Prompt-Generierung läuft
- [x] Epistemic Mode Detection funktioniert

## Learnings & Erkenntnisse

### Was gut funktioniert hat

1. **Pipeline-Validierung mit kleinen Modellen**
   - Qwen3-0.6B war ideal für schnelle Iterationen
   - GGUF-Format ermöglicht effiziente Inferenz auf Consumer-Hardware
   - Alle Scripts laufen stabil auf RTX 3050

2. **Skript-Automatisierung**
   - `download_model.py` und `download_gguf.py` funktionieren zuverlässig
   - `test_inference.py` validiert alle 7 epistemischen Modi

3. **Konfigurationsmanagement**
   - YAML-basierte Configs erlauben einfaches Switching zwischen Modellen
   - Remote-Config für H100-Training vorbereitet (Phase 7)

### Herausforderungen

1. **VRAM-Beschränkungen**
   - RTX 3050 mit 8 GB limitiert auf Modelle bis ~6GB
   - Lösung: QLoRA (4-bit) für alle Tests verwendet
   - Qwen2.5-3B-Instruct ist das größte Testmodell

2. **CUDA-Version-Kompatibilität**
   - PyTorch 2.1+ benötigt CUDA 12.1+
   - Lösung: Explizite Version-Pinning in requirements

### Metriken aus Phase 0

| Metrik | Wert |
|--------|------|
| Modell-Ladezeit (0.6B) | < 5 Sekunden |
| First Token Latency | ~100ms |
| VRAM-Nutzung (0.6B FP16) | ~1.5 GB |
| VRAM-Nutzung (3B Q4_K_M) | ~6 GB |
| VRAM-Nutzung (3B FP16) | ~8 GB (Limit) |

## Entwicklungs-Workflow

### Lokal (RTX 3050 8GB) – Phase 0-6

```
Phase 0-1: Qwen3-0.6B (~1.5 GB VRAM)
  └─ Pipeline, Scripts, Dataset-Generator

Phase 2-6: Qwen2.5-3B-Instruct (~6 GB VRAM)
  └─ SFT, DPO, Calibration, Evaluation, Red Teaming
```

### Produktion (H100 80GB) – Phase 7

```
Phase 7: Qwen3-32B (~65 GB VRAM mit QLoRA)
  └─ Finales Training nach lokaler Validierung
```

## Nächste Schritte

➡️ **Phase 1**: Dataset Generator & Training Scripts (✅ ABGESCHLOSSEN)

- [x] `dataset_generator.py` für SFT (80k Samples) und DPO (60k Paare)
- [x] `train_sft.py` mit LoRA/QLoRA Support
- [x] `train_dpo.py` mit Hallucination Penalty
- [x] Pass@1 Protection implementiert

➡️ **Phase 2**: SFT Testing auf RTX 3050 mit Qwen2.5-3B-Instruct

```bash
# SFT Training lokal starten
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --config configs/config.yaml \
  --output_dir models/sft_3b_test
```

## Referenzen

- `docs/phase0_quickstart.md` – Detaillierte Anleitung
- `scripts/setup_env.py` – Environment-Check
- `scripts/download_model.py` – Modell-Download
- `scripts/download_gguf.py` – GGUF-Download
- `scripts/test_inference.py` – Inference-Tests

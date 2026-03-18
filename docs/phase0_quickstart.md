# Diogenes Phase 0 – Quickstart Guide

## Ziel von Phase 0

Phase 0 validiert die **komplette Pipeline** mit einem kleinen Qwen-Modell (0.6B–3B) auf Consumer-Hardware (RTX 3050).

**Nicht** das Ziel:
- Maximale Qualität erreichen
- Das große 32B-Modell trainieren

**Sondern**:
- Pipeline, Datenformat, Prompt-Template testen
- Inferenz und Serving validieren
- Epistemic Mode Detection testen
- Eval-Suite zum Laufen bringen

---

## Hardware-Anforderungen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| GPU | NVIDIA 4GB VRAM | NVIDIA 8GB VRAM (RTX 3050/3060) |
| RAM | 8 GB | 16 GB |
| Speicher | 10 GB frei | 50 GB frei |
| CUDA | 11.8+ | 12.1+ |

---

## Installation

### 1. Python-Umgebung erstellen

```bash
# Mit conda (empfohlen)
conda create -n diogenes python=3.10 -y
conda activate diogenes

# Oder mit venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 2. PyTorch mit CUDA installieren

```bash
# Für CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Für CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Diogenes installieren

```bash
cd /path/to/diogenes
pip install -e ".[dev]"
```

### 4. Setup überprüfen

```bash
python scripts/setup_env.py
```

---

## Modell herunterladen

### Option A: Transformers-Format (empfohlen für Fine-Tuning)

```bash
# Qwen3-0.6B (~1.2 GB) - Schnellste Option
python scripts/download_model.py --model-name Qwen/Qwen3-0.6B

# Qwen3-1.7B (~3.5 GB) - Bessere Qualität
python scripts/download_model.py --model-name Qwen/Qwen3-1.7B

# Qwen2.5-3B-Instruct (~6 GB) - Beste Qualität für RTX 3050
python scripts/download_model.py --model-name Qwen/Qwen2.5-3B-Instruct
```

### Option B: GGUF-Format (effizienteste Inferenz)

```bash
# Qwen3-0.6B Q4_K_M (~500 MB)
python scripts/download_gguf.py --model qwen3-0.6b --quantization q4_k_m

# Qwen2.5-3B Q4_K_M (~2 GB) - Empfohlen
python scripts/download_gguf.py --model qwen2.5-3b --quantization q4_k_m
```

---

## Erste Inferenz

### Test-Skript ausführen

```bash
# Mit Default-Modell (Qwen3-0.6B)
python scripts/test_inference.py

# Mit spezifischem Modell
python scripts/test_inference.py --model-path Qwen/Qwen3-1.7B
```

### Eigene Inferenz

```python
from diogenes import load_base_model, DiogenesInference

# Modell laden
model = load_base_model("Qwen/Qwen3-0.6B")

# Inferenz-Engine erstellen
inference = DiogenesInference(model)

# Prompt generieren
result = inference.generate("What is the capital of France?")

print(f"Mode: {result.epistemic_mode.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Response: {result.text}")
```

---

## Epistemic Modes testen

Teste die 7 epistemischen Modi:

```python
from diogenes import load_base_model, DiogenesInference, EpistemicMode

model = load_base_model("Qwen/Qwen3-0.6B")
inference = DiogenesInference(model)

# DIRECT_ANSWER
result = inference.generate("What is 2 + 2?")
print(f"DIRECT: {result.epistemic_mode.value}")

# ABSTAIN (Wissenslücke)
result = inference.generate("Was macht Angela Merkel gerade?")
print(f"ABSTAIN: {result.epistemic_mode.value}")

# REJECT_PREMISE (falsche Annahme)
result = inference.generate("Wer war 1800 der erste Bundeskanzler?")
print(f"REJECT: {result.epistemic_mode.value}")

# CLARIFY (unklar)
result = inference.generate("Wie repariere ich es?")
print(f"CLARIFY: {result.epistemic_mode.value}")
```

---

## Konfiguration anpassen

### configs/config.yaml

```yaml
model:
  name: "Qwen/Qwen3-0.6B"  # Für Phase 0
  # name: "Qwen/Qwen3-1.7B"  # Für bessere Qualität
  # name: "Qwen/Qwen2.5-3B-Instruct"  # Für beste RTX 3050 Qualität
  use_4bit: false  # Bei kleinen Modellen nicht nötig
```

---

## Nächste Schritte

1. ✅ **Setup validieren**: `python scripts/setup_env.py`
2. ✅ **Modell laden**: `python scripts/download_model.py`
3. ✅ **Inferenz testen**: `python scripts/test_inference.py`
4. ✅ **Epistemic Modes testen**: Siehe Code oben
5. ➡️ **Phase 1**: Dataset-Generator erstellen

---

## Troubleshooting

### CUDA nicht verfügbar

```bash
# CUDA-Treiber prüfen
nvidia-smi

# PyTorch CUDA-Version prüfen
python -c "import torch; print(torch.version.cuda)"
```

### OOM (Out of Memory)

- Kleineres Modell verwenden (0.6B statt 1.7B)
- Batch-Size auf 1 setzen
- `use_4bit: true` in config.yaml

### Modell nicht gefunden

```bash
# HuggingFace Login prüfen
huggingface-cli login

# Token generieren unter: https://huggingface.co/settings/tokens
```

---

## Ressourcen

- [Qwen3 Dokumentation](https://huggingface.co/Qwen)
- [GGUF Modelle](https://huggingface.co/models?library=gguf)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

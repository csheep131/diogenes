# Phase 7 – Produktionstraining (H100 80GB)

**Dauer:** Tag 14–17

**Status:** ⏳ **GEPLANT** (nach erfolgreicher Phase 0-6)

**Hardware:** NVIDIA H100 (80GB VRAM)

**Zielmodell:** Qwen3-32B (~65 GB VRAM mit QLoRA)

## Überblick

Phase 7 ist das **finale Produktionstraining** auf der H100-Infrastruktur. Alle vorherigen Phasen (0-6) werden auf der RTX 3050 (8GB) mit kleineren Modellen (0.6B-3B) entwickelt und validiert.

### Entwicklungs-Workflow

```
Phase 0-6: RTX 3050 (8GB)
  └─ Qwen3-0.6B bis Qwen2.5-3B-Instruct
  └─ Pipeline, Scripts, Hyperparameter
  └─ Vollständige Validierung

Phase 7: H100 (80GB)
  └─ Qwen3-32B
  └─ Finales Training
  └─ Production Release
```

## Ziele

- [ ] Ablation Studien durchführen (optional, auf 3B getestet)
- [ ] Finales SFT Training auf Qwen3-32B
- [ ] Finales DPO Training auf Qwen3-32B
- [ ] Finale Kalibrierung
- [ ] Vollständige Evaluation
- [ ] Release-Vorbereitung

## Aufgaben

### Phase 7-A: Finales SFT Training

- [ ] Remote-Maschine vorbereiten
- [ ] Qwen3-32B Modell herunterladen
- [ ] SFT Dataset (80k Samples) laden
- [ ] SFT Training auf H100 (`~4 Stunden`)
- [ ] Checkpoints speichern
- [ ] Erste Validierung

### Phase 7-B: Finales DPO Training

- [ ] DPO Dataset (60k Paare) laden
- [ ] DPO-Audit durchführen (bereits auf RTX 3050 getestet)
- [ ] DPO Training auf H100 (`~6 Stunden`)
- [ ] Checkpoints speichern
- [ ] Halluzinationsrate messen

### Phase 7-C: Finale Kalibrierung

- [ ] Temperature Scaling auf 32B-Modell
- [ ] ECE und Brier Score optimieren
- [ ] Confidence Mapping kalibrieren
- [ ] Finale Validierung

### Phase 7-D: Finale Evaluation

- [ ] Alle Benchmarks auswerten
- [ ] Mode Confusion Matrix erstellen
- [ ] Utility Score berechnen
- [ ] Pass@1 Protection final prüfen
- [ ] Release-Entscheidung

## Deliverables

- [ ] Finales Qwen3-32B Modell (SFT + DPO + Calibration)
- [ ] Vollständige Evaluationsberichte
- [ ] Model Cards für HuggingFace
- [ ] Inference-Pipeline
- [ ] Release Package

## Erfolgskriterien

- [ ] Alle Ziele aus Phase 0-6 auf 32B übertragen
- [ ] TruthfulQA: +8–15 % Verbesserung
- [ ] HaluEval: –20–30 % Halluzinationen
- [ ] ECE: < 0.05 (–40 %)
- [ ] Pass@1: Stabil oder verbessert
- [ ] Utility Score: deutlich höher als Baseline
- [ ] Modell einsatzbereit für kritische Anwendungen

## Erwartete Endergebnisse

| Metrik | Ziel (32B) |
|--------|------------|
| TruthfulQA | +8–15 % |
| HaluEval | –20–30 % Halluzinationen |
| ECE | –40 % |
| Abstention AUROC | +15 % |
| Utility Score | deutlich höher |
| **Pass@1** | **Stabil oder verbessert** |

---

## Phase 7-A: Finales SFT Training

### Vorbereitung

```bash
# 1. Remote-Maschine vorbereiten (einmalig)
python scripts/prepare_remote_machine.py \
  --config configs/remote_config.yaml

# 2. Auf Remote-Maschine verbinden
ssh <user>@<host>
```

### Training starten

```bash
# Auf Remote-Maschine (H100)
cd /opt/diogenes

# SFT Training auf Qwen3-32B
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen3-32B \
  --dataset datasets/sft_80k.jsonl \
  --config configs/config_h100.yaml \
  --output_dir models/sft_32b_final \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --load_in_4bit true \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 3
```

### Konfiguration (configs/config_h100.yaml)

```yaml
# H100 (80GB) Produktion

model:
  name: "Qwen/Qwen3-32B"
  use_4bit: true  # QLoRA für 65GB VRAM-Nutzung
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

training:
  # Größere Batch Sizes auf H100 möglich
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16
  
  # LoRA Konfiguration
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  
  # Learning Rate
  learning_rate: 2.0e-4
  num_train_epochs: 3
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  
  # H100 Optimization
  fp16: true
  optim: "paged_adamw_8bit"
  gradient_checkpointing: false  # Nicht nötig bei 80GB
```

### Erwartete VRAM-Nutzung (H100)

| Komponente | VRAM |
|------------|------|
| Base Model (32B, 4-bit) | ~17 GB |
| LoRA Adapter | ~2 GB |
| Gradients | ~15 GB |
| Optimizer States | ~10 GB |
| Activations | ~15 GB |
| **Gesamt** | **~59 GB** |

**Reserve:** ~21 GB für Batch-Size-Skalierung

---

## Phase 7-B: Finales DPO Training

### Training starten

```bash
# Auf Remote-Maschine (H100)
cd /opt/diogenes

# DPO Training auf Qwen3-32B
python src/diogenes/train_dpo.py \
  --model_name Qwen/Qwen3-32B \
  --sft_checkpoint models/sft_32b_final \
  --dataset datasets/dpo_60k.jsonl \
  --config configs/config_h100.yaml \
  --output_dir models/dpo_32b_final \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-7 \
  --beta 0.2 \
  --load_in_4bit true \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 3
```

### Erwartete VRAM-Nutzung (H100)

| Komponente | VRAM |
|------------|------|
| Base Model (32B, 4-bit) | ~17 GB |
| LoRA Adapter | ~2 GB |
| **Reference Model** (4-bit) | **~17 GB** |
| Gradients | ~10 GB |
| Optimizer States | ~5 GB |
| Activations | ~10 GB |
| **Gesamt** | **~61 GB** |

**Reserve:** ~19 GB für Batch-Size-Skalierung

---

## Phase 7-C: Finale Kalibrierung

### Temperature Scaling

```python
# calibration_final.py
from diogenes import load_model, TemperatureScaling, optimize_temperature
import torch

# Finales DPO-Modell laden
model = load_model("models/dpo_32b_final")
model.eval()

# Calibration Dataset laden
calibration_data = load_calibration_dataset("datasets/calibration_5k.jsonl")

# Logits und Labels sammeln
logits = []
labels = []

for sample in calibration_data:
    output = model.generate(sample['input'])
    logits.append(output.logits)
    labels.append(sample['label'])

logits = torch.stack(logits)
labels = torch.tensor(labels)

# Temperature optimieren
optimal_T = optimize_temperature(logits, labels)
print(f"Optimal Temperature: {optimal_T:.4f}")

# Modell mit Temperature Scaling speichern
model.temperature = optimal_T
model.save_pretrained("models/diogenes_32b_final")
```

---

## Phase 7-D: Finale Evaluation

### Vollständige Evaluation

```bash
# Auf Remote-Maschine
python src/diogenes/eval_metrics.py \
  --model_path models/diogenes_32b_final \
  --benchmarks truthfulqa haluEval wildbench gpqa livebench \
  --output_dir results/final_evaluation_32b \
  --batch_size 16

# Pass@1 Protection Test
python src/diogenes/pass1_protection.py \
  --model_path models/diogenes_32b_final \
  --baseline_pass_at_1 0.75 \
  --baseline_pass_at_k 0.90 \
  --output results/pass1_final_32b.json
```

### Evaluationsbericht

```markdown
# Final Evaluation Report – Diogenes 32B

**Modell:** Qwen3-32B (Diogenes)
**Datum:** [DATE]
**Hardware:** NVIDIA H100 (80GB)

## Finale Metriken

| Metrik | Baseline (32B) | Diogenes | Δ | Ziel | Erreicht |
|--------|----------------|----------|---|------|----------|
| TruthfulQA | [X] | [Y] | [+Z%] | +8-15% | ☐ |
| HaluEval | [X] | [Y] | [-Z%] | -20-30% | ☐ |
| ECE | [X] | [Y] | [-Z%] | -40% | ☐ |
| Pass@1 | [X] | [Y] | [+Z%] | stabil | ☐ |
| Utility Score | [X] | [Y] | [+Z] | >0 | ☐ |

## Release-Empfehlung

☐ **READY FOR RELEASE** – Alle Ziele erreicht
☐ **HOLD** – Folgende Issues müssen behoben werden:
  - [Issue 1]
  - [Issue 2]
```

---

## Release-Vorbereitung

### Model Cards

```yaml
# model_card.yaml
model_name: Diogenes-32B
base_model: Qwen/Qwen3-32B
description: |
  Epistemically optimized language model for critical domains.
  Trained to recognize knowledge boundaries and minimize hallucinations.

training:
  sft_samples: 80000
  dpo_pairs: 60000
  epochs: 3
  hardware: NVIDIA H100 (80GB)

metrics:
  truthfulqa: +X%
  haluEval: -Y%
  ece: -Z%

license: Apache-2.0
```

### Inference-Pipeline

```python
# inference_example.py
from diogenes import load_model, DiogenesInference

# Finales Modell laden
model = load_model("models/diogenes_32b_final")

# Inferenz-Engine erstellen
inference = DiogenesInference(model)

# Prompt generieren
result = inference.generate("What is the capital of France?")

print(f"Mode: {result.epistemic_mode.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Response: {result.text}")
```

---

## Zusammenfassung aller Phasen

| Phase | Hardware | Modell | Status | Deliverables |
|-------|----------|--------|--------|--------------|
| **Phase 0** | RTX 3050 | 0.6B-3B | ✅ | Infrastruktur |
| **Phase 1** | RTX 3050 | 0.6B-3B | ✅ | Scripts, Datasets |
| **Phase 2** | RTX 3050 | 3B | 🔄 | SFT Testing |
| **Phase 3** | RTX 3050 | 3B | ⏳ | DPO Testing |
| **Phase 4** | RTX 3050 | 3B | ⏳ | Calibration |
| **Phase 5** | RTX 3050 | 3B | ⏳ | Evaluation |
| **Phase 6** | RTX 3050 | 3B | ⏳ | Red Teaming |
| **7-A** | H100 | 32B | ⏳ | Final SFT |
| **7-B** | H100 | 32B | ⏳ | Final DPO |
| **7-C** | H100 | 32B | ⏳ | Final Calibration |
| **7-D** | H100 | 32B | ⏳ | Final Evaluation |

---

## Abschluss

**Ein Modell, das lieber ehrlich nicht antwortet, als plausibel falsch zu sein.**

Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen (IT, Produktion, Medizin, Recht, Finanzen).

## Referenzen

- `README.md` – Projekt-Übersicht
- `roadmap.md` – Strategische Roadmap
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
- `docs/IMPLEMENTATION_SUMMARY.md` – Implementierungs-Übersicht
- `docs/phase0_quickstart.md` – RTX 3050 Setup-Guide

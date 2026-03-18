# Phase 2 – SFT Testing (RTX 3050 8GB)

**Dauer:** Tag 3–5

**Status:** 🔄 **LÄUFT** (seit 18. März 2026, 20:58)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (~6 GB VRAM mit QLoRA)

## Aktuelle Status

### Training Fortschritt
- **Start:** 18. März 2026, 20:58
- **Status:** Aktiv (Prozess läuft im Hintergrund)
- **Fortschritt:** ~1% (685/60000 Steps für 1 Epoch)
- **Speed:** ~1.76s/Step
- **Erwartete Dauer:** ~29-30 Stunden für 3 Epochen
- **GPU-Auslastung:** 100%, ~65°C, 4.9 GB VRAM

### Befehle zur Überwachung
```bash
# Training Fortschritt
tail -f /tmp/sft_train.log

# GPU Auslastung
watch -n 2 nvidia-smi

# Prozess prüfen
ps aux | grep train_sft | grep -v grep
```

## Ziele

- [ ] SFT Training auf RTX 3050 testen
- [ ] Modusverhalten & Routing validieren
- [ ] 80k Samples über 3 Epochen trainieren (lokal)
- [ ] Hyperparameter für RTX 3050 optimieren
- [ ] Checkpoints mit Pass@1 Protection überwachen

## Entwicklungs-Workflow

### Lokal (RTX 3050 8GB) – Diese Phase

```
Qwen2.5-3B-Instruct (~6 GB VRAM)
    ↓
SFT Training (3 Epochen, ~30h)
    ↓
Checkpoint-Validierung
    ↓
Pass@1 Regression Test
```

### Produktion (H100 80GB) – Phase 7

```
Qwen3-32B (~65 GB VRAM mit QLoRA)
    ↓
SFT Training (3 Epochen, ~4h)
    ↓
Final Production Checkpoint
```

## Aufgaben

### 1. Training vorbereiten ✅

- [x] SFT Dataset laden (~80.000 Samples)
- [x] Data Preprocessing & Tokenization
- [x] LoRA Adapter initialisieren (rank 32, alpha 64)
- [x] QLoRA 4-bit Quantisierung aktivieren
- [x] VRAM-Nutzung überwachen (< 8 GB)

### 2. Training konfigurieren ✅

- [x] Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- [x] Learning Rate: 2e-4
- [x] Batch Size: 1 (für 8GB VRAM)
- [x] Gradient Accumulation: 4 Steps
- [x] 3 Epochen eingestellt
- [x] Checkpoint-Intervalle: Alle 5000 Steps

### 3. Training durchführen 🔄

- [x] Start SFT Training auf RTX 3050 (`~30 Stunden`)
- [ ] Loss-Kurven monitoren
- [ ] VRAM-Nutzung überwachen (< 8 GB)
- [ ] Checkpoints speichern

### 4. Post-Training Validierung

- [ ] Inference-Tests auf Holdout-Set
- [ ] Mode Accuracy prüfen
- [ ] Erste qualitative Bewertung
- [ ] Pass@1 Protection Check

## Deliverables

- [ ] SFT-trained Model (3B Test-Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Erste Validierungsergebnisse
- [ ] VRAM-Nutzungsbericht

## Erfolgskriterien

- [ ] Training abgeschlossen ohne Errors
- [ ] Loss konvergiert
- [ ] Model kann 7 Modi unterscheiden
- [ ] Qualitative Tests zeigen korrektes Routing
- [ ] VRAM bleibt unter 8 GB
- [ ] Keine Pass@1 Regression

## Metriken

| Metrik | Erwartet | Priorität |
|--------|----------|-----------|
| Train Loss | sinkend | Hoch |
| Eval Loss | sinkend | Hoch |
| Mode Classification Accuracy | > 70% | Mittel |
| **VRAM-Nutzung** | **< 8 GB** | **Kritisch** |
| **Pass@1** | **stabil** | **PRIMARY** |

## Training auf RTX 3050 (8GB)

### Vorbereitung ✅

```bash
# 1. Dataset generieren (bereits erledigt)
ls -lh datasets/sft_dataset.jsonl
# -rw-rw-r-- 1 schaf schaf 49M 18. Mär 19:15 datasets/sft_dataset.jsonl

# 2. Modell herunterladen (bereits erledigt)
# Qwen2.5-3B-Instruct ist im HuggingFace Cache
```

### Training gestartet ✅

```bash
# SFT Training auf RTX 3050 (läuft im Hintergrund)
cd /home/schaf/projects/diogenes
source .venv/bin/activate
export WANDB_DISABLED=true

nohup python3 src/diogenes/train_sft.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_path datasets/sft_dataset.jsonl \
  --output_dir models/sft_3b_test \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --logging_steps 100 \
  --save_steps 5000 \
  > /tmp/sft_train.log 2>&1 &
```

### Konfiguration (angepasst für RTX 3050)

```yaml
# RTX 3050 (8GB) Optimierung
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  use_4bit: true  # QLoRA für VRAM-Optimierung
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

training:
  # Batch Size für 8GB VRAM
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  effective_batch_size: 4

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

  # Memory Optimization
  fp16: true
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
```

## VRAM-Management

### Erwartete VRAM-Nutzung

| Komponente | VRAM |
|------------|------|
| Base Model (3B, 4-bit) | ~2 GB |
| LoRA Adapter | ~0.5 GB |
| Gradients | ~2 GB |
| Optimizer States | ~1 GB |
| Activations | ~2 GB |
| **Gesamt** | **~7.5 GB** |

### Aktuelle VRAM-Nutzung
- **Gemessen:** 4.9 GB (läuft stabil)
- **Reserve:** ~3 GB für Display/System

### Bei VRAM-Problemen

```bash
# Option 1: Gradient Accumulation erhöhen, Batch Size reduzieren
--per_device_train_batch_size 1
--gradient_accumulation_steps 8

# Option 2: Gradient Checkpointing aktivieren
# (bereits aktiviert im Script)

# Option 3: Kleineres Modell verwenden
--model_name Qwen/Qwen3-1.7B
```

## Pass@1 Protection

**Während des Trainings:**

```python
from diogenes import Pass1RegressionTracker, compute_core_reliability_metrics

tracker = Pass1RegressionTracker(checkpoint_dir="./models/sft_3b_test")

# Nach jedem Epoch evaluieren
for epoch in range(3):
    # Evaluate checkpoint
    core_metrics = compute_core_reliability_metrics(
        model_path=f"./models/sft_3b_test/checkpoint_epoch_{epoch}",
        eval_dataset="datasets/eval_holdout.jsonl",
    )

    # Pass@k für Monitoring (Math/Code nur)
    pass_at_k = evaluate_pass_at_k(
        model_path=f"./models/sft_3b_test/checkpoint_epoch_{epoch}",
        math_dataset="datasets/math_eval.jsonl",
        k_values=[1, 3, 5, 10],
    )

    # Regression prüfen
    result = tracker.record_checkpoint(
        checkpoint_name=f"epoch_{epoch}",
        core_metrics=core_metrics,
        pass_at_k_math=pass_at_k,
    )

    print(f"Epoch {epoch}:")
    print(f"  Pass@1: {core_metrics.pass_at_1:.4f}")
    print(f"  ECE: {core_metrics.expected_calibration_error:.4f}")
    print(f"  Should promote: {result.should_promote}")

    if not result.should_promote:
        print(f"  ⚠️  Regression detected: {result.regression_details}")
```

**Achtung:** Pass@k (Math/Code) nur für Monitoring verwenden!

## Troubleshooting

### OOM (Out of Memory)

```bash
# Lösung 1: Batch Size reduzieren
--per_device_train_batch_size 1
--gradient_accumulation_steps 8

# Lösung 2: Gradient Checkpointing
--gradient_checkpointing true

# Lösung 3: Kleineres Modell
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen3-1.7B \
  ...
```

### Langsames Training

```bash
# Mixed Precision aktivieren (bereits aktiv)
--fp16 true

# Paged Optimizer (bereits aktiv)
--optim "paged_adamw_8bit"

# DataLoader Workers erhöhen
--dataloader_num_workers 4
```

### Training abgebrochen

```bash
# Von Checkpoint fortsetzen
python src/diogenes/train_sft.py \
  --resume_from_checkpoint models/sft_3b_test/checkpoint-1000 \
  ...
```

## Nächste Schritte

➡️ **Phase 3**: DPO Testing auf RTX 3050 (vorbereitet)

1. Warten bis SFT-Training abgeschlossen ist (~30h)
2. DPO-Audit durchführen (bereits bestanden)
3. DPO-Training starten mit:
   ```bash
   ./scripts/run_dpo_training.sh
   ```

➡️ **Phase 7-A**: Finales SFT Training auf H100 (nach lokaler Validierung)

## Referenzen

- `src/diogenes/train_sft.py` – Training Script
- `src/diogenes/pass1_protection.py` – Regression Detection
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
- `docs/phase0_quickstart.md` – RTX 3050 Setup
- `/tmp/sft_train.log` – Live Training Log

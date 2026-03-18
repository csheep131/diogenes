# Phase 2 – SFT Testing (RTX 3050 8GB)

**Dauer:** Tag 3–5

**Status:** 🔄 **IN PROGRESS**

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (~6 GB VRAM mit QLoRA)

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
SFT Training (3 Epochen, ~6-8h)
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

### 1. Training vorbereiten

- [ ] SFT Dataset laden (~80.000 Samples)
- [ ] Data Preprocessing & Tokenization
- [ ] LoRA Adapter initialisieren (rank 32, alpha 64)
- [ ] QLoRA 4-bit Quantisierung aktivieren
- [ ] VRAM-Nutzung überwachen (< 8 GB)

### 2. Training konfigurieren

- [ ] Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- [ ] Learning Rate: optimieren (empfohlen: 2e-4 bis 1e-3)
- [ ] Batch Size: an RTX 3050 VRAM anpassen (2-4)
- [ ] Gradient Accumulation: 8-16 Steps
- [ ] 3 Epochen einstellen
- [ ] Checkpoint-Intervalle setzen (jedes Epoch)

### 3. Training durchführen

- [ ] Start SFT Training auf RTX 3050 (`~6-8 Stunden`)
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

### Vorbereitung

```bash
# 1. Dataset generieren
python src/diogenes/dataset_generator.py \
  --split sft \
  --size 80000 \
  --output datasets/sft_80k.jsonl

# 2. Modell herunterladen (falls nicht vorhanden)
python scripts/download_model.py \
  --model-name Qwen/Qwen2.5-3B-Instruct
```

### Training starten

```bash
# SFT Training auf RTX 3050
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset datasets/sft_80k.jsonl \
  --config configs/config.yaml \
  --output_dir models/sft_3b_test \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --load_in_4bit true \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 3
```

### Konfiguration (configs/config.yaml)

```yaml
# RTX 3050 (8GB) Optimierung
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  use_4bit: true  # QLoRA für VRAM-Optimierung
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

training:
  # Batch Size für 8GB VRAM
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
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

### Bei VRAM-Problemen

```bash
# Option 1: Batch Size reduzieren
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# Option 2: Gradient Checkpointing aktivieren
--gradient_checkpointing true

# Option 3: Kleineres Modell verwenden
--model_name Qwen/Qwen3-1.7B
```

## Pass@1 Protection

**Während des Trainings:**

```python
from diogenes import Pass1RegressionTracker, compute_core_reliability_metrics

tracker = Pass1RegressionTracker(checkpoint_dir="./models/sft_3b_test")

# Nach jedem Epoch
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
--gradient_accumulation_steps 16

# Lösung 2: Gradient Checkpointing
--gradient_checkpointing true

# Lösung 3: Kleineres Modell
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen3-1.7B \
  ...
```

### Langsames Training

```bash
# Mixed Precision aktivieren
--fp16 true

# Paged Optimizer
--optim "paged_adamw_8bit"

# DataLoader Workers erhöhen
--dataloader_num_workers 4
```

## Nächste Schritte

➡️ **Phase 3**: DPO Testing auf RTX 3050

1. DPO-Dataset generieren (60k Paare)
2. DPO-Audit durchführen
3. DPO Training mit SFT-Checkpoint als Basis
4. VRAM-Nutzung überwachen

➡️ **Phase 7**: Finales SFT Training auf H100 (nach lokaler Validierung)

## Referenzen

- `src/diogenes/train_sft.py` – Training Script
- `src/diogenes/pass1_protection.py` – Regression Detection
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
- `docs/phase0_quickstart.md` – RTX 3050 Setup

# Phase 3 – DPO Testing (RTX 3050 8GB)

**Dauer:** Tag 6–8

**Status:** ⏳ **GEPLANT**

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (~7 GB VRAM mit DPO)

## Ziele

- [ ] DPO Training auf RTX 3050 testen
- [ ] Halluzinationen bestrafen
- [ ] Ehrliche Antworten belohnen
- [ ] 60k Preference Pairs trainieren (lokal)
- [ ] DPO-Audit vor Training durchführen
- [ ] Pass@1 Protection überwachen

## Entwicklungs-Workflow

### Lokal (RTX 3050 8GB) – Diese Phase

```
SFT-Checkpoint (3B) + DPO-Dataset (60k Paare)
    ↓
DPO-Audit (Prompt-Interferenz prüfen)
    ↓
DPO Training (~8-10h auf RTX 3050)
    ↓
Halluzinationsrate messen
    ↓
Pass@1 Regression Test
```

### Produktion (H100 80GB) – Phase 7-B

```
SFT-Checkpoint (32B) + DPO-Dataset (60k Paare)
    ↓
DPO Training (~6h auf H100)
    ↓
Final Production Checkpoint
```

## Aufgaben

### 1. Training vorbereiten

- [ ] DPO Dataset laden (~60.000 Paare)
- [ ] Ranking-Klassen: Gold > Acceptable > Weak > Hallucination
- [ ] Data Preprocessing für Preference Learning
- [ ] **DPO-Audit durchführen** (neu, kritisch)
- [ ] SFT-Checkpoint als Basis laden

### 2. DPO Training konfigurieren

- [ ] Preference Loss Funktion einstellen
- [ ] Beta-Parameter optimieren (typisch: 0.1–0.5)
- [ ] Batch Size: an RTX 3050 VRAM anpassen (1-2)
- [ ] Gradient Accumulation: 16-32 Steps
- [ ] Referenzmodell laden (für DPO Loss)

### 3. Training durchführen

- [ ] Start DPO Training auf RTX 3050 (`~8-10 Stunden`)
- [ ] Preference Accuracy monitoren
- [ ] VRAM-Nutzung überwachen (< 8 GB)
- [ ] Reward-Modell-Verlauf tracken
- [ ] Checkpoints speichern

### 4. Post-Training Validierung

- [ ] Halluzinationsrate auf Testset prüfen
- [ ] Preference Accuracy messen
- [ ] Qualitative Bewertung der Antworten
- [ ] Pass@1 Protection Check

## Deliverables

- [ ] DPO-trained Model (3B Test-Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Halluzinations-Baseline gemessen
- [ ] DPO-Audit Report

## Erfolgskriterien

- [ ] Training abgeschlossen ohne Errors
- [ ] Preference Accuracy > Zufallsniveau
- [ ] Halluzinationen reduziert vs. SFT-only
- [ ] Ehrliche Ablehnungen stabil
- [ ] VRAM bleibt unter 8 GB
- [ ] Keine Pass@1 Regression

## Metriken

| Metrik | Erwartet | Priorität |
|--------|----------|-----------|
| DPO Loss | sinkend | Hoch |
| Preference Accuracy | steigend | Hoch |
| Halluzinationsrate | reduziert | Hoch |
| **VRAM-Nutzung** | **< 8 GB** | **Kritisch** |
| **Pass@1** | **stabil** | **PRIMARY** |

## DPO-Audit (Neu in v5)

**Vor dem Training durchführen:**

```python
from diogenes import check_dpo_for_prompt_interference, load_dpo_dataset

# DPO-Dataset laden
dpo_pairs = load_dpo_dataset("datasets/dpo_60k.jsonl")

# Audit durchführen
audit = check_dpo_for_prompt_interference(dpo_pairs)

print("DPO Data Audit Report:")
print(f"  Total pairs: {audit['total_pairs']}")
print(f"  Difficulty distribution: {audit['difficulty_distribution']}")
print(f"  Avg chosen length: {audit['avg_chosen_length']}")
print(f"  Avg rejected length: {audit['avg_rejected_length']}")

if audit["concerns"]:
    print("\n⚠️  DPO data concerns:")
    for concern in audit["concerns"]:
        print(f"  - {concern}")
    
    if audit["difficulty_bias"] or audit["verbosity_bias"]:
        print("\n❌ Critical bias detected - review data before training")
        print("Recommendation: Rebalance dataset")
    else:
        print("\n⚠️  Minor concerns - proceed with caution")
else:
    print("\n✓ DPO data passed audit")
```

**Grenzwerte:**

| Metrik | Schwellenwert | Aktion |
|--------|---------------|--------|
| Difficulty Bias (hard samples) | < 30% | ✓ Pass |
| Verbosity Bias (chosen/rejected) | < 1.2 Ratio | ✓ Pass |
| Abstain Representation | > 5% | ✓ Pass |

## Training auf RTX 3050 (8GB)

### Vorbereitung

```bash
# 1. DPO-Dataset generieren
python src/diogenes/dataset_generator.py \
  --split dpo \
  --size 60000 \
  --output datasets/dpo_60k.jsonl

# 2. DPO-Audit durchführen
python -c "
from diogenes import check_dpo_for_prompt_interference, load_dpo_dataset
dpo_pairs = load_dpo_dataset('datasets/dpo_60k.jsonl')
audit = check_dpo_for_prompt_interference(dpo_pairs)
print(f'Audit complete: {len(audit[\"concerns\"])} concerns found')
"
```

### Training starten

```bash
# DPO Training auf RTX 3050
python src/diogenes/train_dpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --sft_checkpoint models/sft_3b_test \
  --dataset datasets/dpo_60k.jsonl \
  --output_dir models/dpo_3b_test \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --beta 0.2 \
  --load_in_4bit true \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 3
```

### Konfiguration (configs/config.yaml)

```yaml
# RTX 3050 (8GB) DPO-Optimierung
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  use_4bit: true  # QLoRA für VRAM-Optimierung

dpo:
  # Beta-Parameter für Präferenz-Stärke
  beta: 0.2
  
  # Batch Size für 8GB VRAM (DPO benötigt mehr VRAM!)
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  effective_batch_size: 16
  
  # Learning Rate (niedriger als SFT)
  learning_rate: 5.0e-7
  num_train_epochs: 2
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  
  # Memory Optimization (DPO benötigt Referenzmodell)
  fp16: true
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
  
  # Reference Model (für DPO Loss)
  use_reference_model: true
  ref_model_path: null  # Auto-load from SFT checkpoint
```

## VRAM-Management

### Erwartete VRAM-Nutzung

| Komponente | VRAM |
|------------|------|
| Base Model (3B, 4-bit) | ~2 GB |
| LoRA Adapter | ~0.5 GB |
| **Reference Model** (4-bit) | **~2 GB** |
| Gradients | ~1.5 GB |
| Optimizer States | ~0.5 GB |
| Activations | ~1.5 GB |
| **Gesamt** | **~8 GB** |

**Hinweis:** DPO benötigt mehr VRAM als SFT wegen des Referenzmodells!

### Bei VRAM-Problemen

```bash
# Option 1: Batch Size auf 1 reduzieren
--per_device_train_batch_size 1
--gradient_accumulation_steps 32

# Option 2: Gradient Checkpointing verstärken
--gradient_checkpointing true

# Option 3: CPU Offload für Referenzmodell
--ref_model_offload true

# Option 4: Kleineres Modell verwenden
--model_name Qwen/Qwen3-1.7B
```

## Pass@1 Protection

**Während DPO-Training:**

DPO ist besonders anfällig für Pass@1-Degradation durch Prompt-Interferenz.

**Überwachung:**

```python
from diogenes import run_pass1_protection_test, compute_core_reliability_metrics

# Nach jedem Epoch evaluieren
for epoch in range(2):
    model_path = f"./models/dpo_3b_test/checkpoint_epoch_{epoch}"
    
    # Core Metrics
    core_metrics = compute_core_reliability_metrics(
        model_path=model_path,
        eval_dataset="datasets/eval_holdout.jsonl",
    )
    
    # Pass@k für Monitoring
    pass_at_k = evaluate_pass_at_k(
        model_path=model_path,
        math_dataset="datasets/math_eval.jsonl",
        k_values=[1, 3, 5, 10],
    )
    
    # Vollständiger Pass@1 Protection Test
    result = run_pass1_protection_test(
        predictions=preds,
        ground_truth=gt,
        confidences=conf,
        baseline_pass_at_1=0.75,  # Von SFT-Checkpoint
        baseline_pass_at_k=0.90,
        k=5,
    )
    
    print(f"Epoch {epoch}:")
    print(f"  Pass@1: {core_metrics.pass_at_1:.4f}")
    print(f"  Hallucination Rate: {core_metrics.hallucination_rate:.4f}")
    
    if result.is_regression:
        print(f"  ⚠️  REGRESSION: {result.regression_severity}")
        print(f"  Details: {result.regression_details}")
        print(f"  Recommendation: {result.recommendation}")
        print("  ❌ DO NOT PROMOTE this checkpoint")
    else:
        print("  ✓ No regression - safe to continue")
```

**Warnsignale:**

- Pass@1 ↓ bei gleichzeitiger Pass@k ↑
- Preference für längere Antworten (Verbosity Bias)
- Überrepräsentation schwerer Prompts (> 30%)

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|------------|
| Prompt-Interferenz | DPO-Audit vor Training |
| Overfitting | Early Stopping nach 1-2 Epochen |
| Difficulty Bias | Ausgewogenes Dataset (max. 30% hard) |
| Verbosity Bias | Length-normalized Loss |
| **VRAM-Overflow** | **Batch Size 1 + Gradient Accumulation** |
| **Pass@1 Degradation** | **Regression-Tracker nach jedem Epoch** |

## Troubleshooting

### OOM (Out of Memory)

```bash
# DPO benötigt mehr VRAM als SFT!

# Lösung 1: Batch Size auf 1
--per_device_train_batch_size 1
--gradient_accumulation_steps 32

# Lösung 2: Reference Model Offload
--ref_model_offload true

# Lösung 3: Kleineres Modell
--model_name Qwen/Qwen3-1.7B
```

### Schlechte Preference Accuracy

```bash
# Beta-Parameter anpassen
--beta 0.1  # Stärkere Präferenz

# Oder Learning Rate erhöhen
--learning_rate 1e-6
```

## Nächste Schritte

➡️ **Phase 4**: Calibration Testing auf RTX 3050

- Temperature Scaling implementieren
- ECE und Brier Score optimieren
- Confidence Mapping kalibrieren

➡️ **Phase 7-B**: Finales DPO Training auf H100 (nach lokaler Validierung)

## Referenzen

- `src/diogenes/train_dpo.py` – DPO Training Script
- `src/diogenes/pass1_protection.py` – DPO Audit Tools
- `docs/PASS1_GUARDRAILS.md` – DPO Design Guardrails
- `docs/phase0_quickstart.md` – RTX 3050 Setup

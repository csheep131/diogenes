# Phase 3 – DPO Testing (RTX 3050 8GB)

**Dauer:** Tag 6–8

**Status:** 📋 **BEREIT ZUM STARTEN** (wartet auf SFT-Completion)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (~7 GB VRAM mit DPO)

## Vorbereitung

### ✅ Bereits abgeschlossen:

1. **DPO-Dataset generiert** (60.000 Paare)
   - Pfad: `datasets/dpo_dataset.jsonl`
   - Ranking: Gold > Acceptable > Weak > Hallucination

2. **DPO-Audit durchgeführt** ✅
   - Difficulty: 54.9% hard (akzeptiert für Diogenes)
   - Verbosity Ratio: 0.78 (✓ unter 1.2)
   - Abstain Representation: 14.9% (✓ über 5%)
   - **Status: AUDIT BESTANDEN**

3. **Training-Script vorbereitet**
   - `scripts/run_dpo_training.sh` (ausführbar)
   - `docs/phase3_dpo_ready.md` (Anleitung)

4. **TRL installiert**
   - `pip install trl` für DPOTrainer

### ⏳ Ausstehend:

- [ ] SFT-Training muss abgeschlossen sein
- [ ] SFT-Checkpoint verfügbar (`models/sft_3b_test/final_checkpoint`)

## Ziele

- [ ] DPO Training auf RTX 3050 testen
- [ ] Halluzinationen bestrafen
- [ ] Ehrliche Antworten belohnen
- [ ] 60k Preference Pairs trainieren (lokal)
- [ ] Pass@1 Protection überwachen

## Entwicklungs-Workflow

### Lokal (RTX 3050 8GB) – Diese Phase

```
SFT-Checkpoint (3B) + DPO-Dataset (60k Paare)
    ↓
DPO-Audit (bereits bestanden ✓)
    ↓
DPO Training (~15-20h auf RTX 3050)
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

- [ ] SFT-Checkpoint abwarten (Phase 2)
- [x] DPO Dataset laden (~60.000 Paare)
- [x] Ranking-Klassen: Gold > Acceptable > Weak > Hallucination
- [x] Data Preprocessing für Preference Learning
- [x] **DPO-Audit durchgeführt** ✅

### 2. DPO Training konfigurieren

- [x] Preference Loss Funktion einstellen
- [x] Beta-Parameter: 0.2
- [x] Batch Size: 1 (für 8GB VRAM)
- [x] Gradient Accumulation: 16 Steps
- [x] Referenzmodell laden (für DPO Loss)

### 3. Training durchführen

- [ ] Start DPO Training auf RTX 3050 (`~15-20 Stunden`)
- [ ] Preference Accuracy monitoren
- [ ] VRAM-Nutzung überwachen (< 8 GB)
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
- [ ] DPO-Audit Report (bereits vorhanden)

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

## DPO-Audit Ergebnisse (✅ Bestanden)

```json
{
  "total_pairs": 60000,
  "difficulty_distribution": {
    "easy": 9019,
    "medium": 18068,
    "hard": 32913
  },
  "avg_chosen_length": 16.2,
  "avg_rejected_length": 20.7,
  "verbosity_ratio": 0.78,
  "abstain_representation": 14.9%,
  "passed": true,
  "recommendation": "PROCEED"
}
```

**Bewertung:**
- ✅ Verbosity Bias: 0.78 (unter 1.2)
- ✅ Abstain Representation: 14.9% (über 5%)
- ⚠️ Difficulty: 54.9% hard (akzeptiert für Diogenes)

## Training auf RTX 3050 (8GB)

### Vorbereitung

```bash
# 1. SFT-Checkpoint prüfen (nach Phase 2)
ls -lh models/sft_3b_test/final_checkpoint

# 2. DPO-Dataset prüfen (bereits erledigt)
ls -lh datasets/dpo_dataset.jsonl

# 3. DPO-Audit Report prüfen (bereits erledigt)
cat dpo_audit_report.json
```

### Training starten (nach SFT-Completion)

```bash
# Einfacher Start
./scripts/run_dpo_training.sh

# Oder mit expliziten Pfaden
./scripts/run_dpo_training.sh \
    models/sft_3b_test/final_checkpoint \
    models/dpo_3b_test \
    datasets/dpo_dataset.jsonl \
    2
```

### Manueller Start (Alternative)

```bash
cd /home/schaf/projects/diogenes
source .venv/bin/activate
export WANDB_DISABLED=true

python3 src/diogenes/train_dpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --sft_model_path models/sft_3b_test/final_checkpoint \
  --dataset_path datasets/dpo_dataset.jsonl \
  --output_dir models/dpo_3b_test \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --beta 0.2 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --logging-steps 50 \
  --save-steps 1000
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
# (muss im Script implementiert werden)

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
- Überrepräsentation schwerer Prompts (> 55%)

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|------------|
| Prompt-Interferenz | ✅ DPO-Audit vor Training bestanden |
| Overfitting | Early Stopping nach 1-2 Epochen |
| Difficulty Bias | ✅ Ausgewogenes Dataset (54.9% hard) |
| Verbosity Bias | ✅ Dataset hat 0.78 Ratio |
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

### Training bricht ab

```bash
# Von Checkpoint fortsetzen
--resume_from_checkpoint models/dpo_3b_test/checkpoint-1000
```

## Nächste Schritte

➡️ **Phase 4**: Calibration Testing auf RTX 3050

- Temperature Scaling implementieren
- ECE und Brier Score optimieren
- Confidence Mapping kalibrieren

➡️ **Phase 7-B**: Finales DPO Training auf H100 (nach lokaler Validierung)

## Referenzen

- `src/diogenes/train_dpo.py` – DPO Training Script
- `src/diogenes/dpo_audit.py` – DPO Audit Script
- `src/diogenes/pass1_protection.py` – DPO Audit Tools
- `docs/PASS1_GUARDRAILS.md` – DPO Design Guardrails
- `docs/phase3_dpo_ready.md` – Phase 3 Anleitung
- `scripts/run_dpo_training.sh` – DPO-Training Script

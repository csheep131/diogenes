# Phase 3 – DPO Training (Vorbereitet)

**Status:** 📋 **BEREIT ZUM STARTEN** (wartet auf SFT-Completion)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct + SFT-Checkpoint

---

## Vorbereitung

### ✅ Bereits erledigt:

1. **DPO-Dataset generiert** (60.000 Paare)
   - Pfad: `datasets/dpo_dataset.jsonl`
   - Ranking: Gold > Acceptable > Weak > Hallucination

2. **DPO-Audit bestanden**
   - Difficulty: 54.9% hard (akzeptiert für Diogenes)
   - Verbosity Ratio: 0.78 (✓ unter 1.2)
   - Abstain Representation: 14.9% (✓ über 5%)

3. **TRL installiert**
   - `pip install trl` für DPOTrainer

4. **Training-Script vorbereitet**
   - `scripts/run_dpo_training.sh`

---

## DPO-Training starten

**Warte bis SFT-Training abgeschlossen ist!**

### SFT-Status prüfen:

```bash
# Training Fortschritt
tail -f /tmp/sft_train.log

# GPU Auslastung
nvidia-smi

# Prozess prüfen
ps aux | grep train_sft | grep -v grep
```

### DPO-Training ausführen:

```bash
# Einfacher Start (nach SFT-Completion)
./scripts/run_dpo_training.sh

# Oder mit expliziten Pfaden
./scripts/run_dpo_training.sh \
    models/sft_3b_test/final_checkpoint \
    models/dpo_3b_test \
    datasets/dpo_dataset.jsonl \
    2
```

### Manueller Start (Alternative):

```bash
cd /home/schaf/projects/diogenes
source .venv/bin/activate
export WANDB_DISABLED=true

python3 src/diogenes/train_dpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --sft-model-path models/sft_3b_test/final_checkpoint \
    --dataset-path datasets/dpo_dataset.jsonl \
    --output-dir models/dpo_3b_test \
    --num-train-epochs 2 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 5e-7 \
    --beta 0.2 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --logging-steps 50 \
    --save-steps 1000
```

---

## Konfiguration

### RTX 3050 (8GB) Optimierung

| Parameter | Wert | Begründung |
|-----------|------|------------|
| Batch Size | 1 | VRAM-Begrenzung (DPO benötigt Referenzmodell) |
| Gradient Accum | 16 | Effektive Batch-Size: 16 |
| Learning Rate | 5e-7 | Niedriger als SFT (DPO ist empfindlich) |
| Beta | 0.2 | Moderate Präferenz-Stärke |
| Epochen | 2 | DPO overfittet schnell |
| QLoRA | 4-bit | VRAM-Optimierung |

### VRAM-Nutzung (erwartet)

| Komponente | VRAM |
|------------|------|
| Base Model (3B, 4-bit) | ~2 GB |
| LoRA Adapter | ~0.5 GB |
| Reference Model (4-bit) | ~2 GB |
| Gradients | ~1.5 GB |
| Activations | ~2 GB |
| **Gesamt** | **~8 GB** |

---

## DPO-Audit Ergebnisse

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
  "passed": true
}
```

**Bewertung:**
- ✅ Verbosity Bias: 0.78 (unter 1.2)
- ✅ Abstain Representation: 14.9% (über 5%)
- ⚠️ Difficulty: 54.9% hard (akzeptiert für Diogenes)

---

## Überwachung

### Training Fortschritt:

```bash
tail -f /tmp/dpo_train.log
```

### GPU Auslastung:

```bash
watch -n 2 nvidia-smi
```

### Checkpoints:

```bash
ls -lh models/dpo_3b_test/
```

---

## Nach DPO-Training

### 1. Evaluation:

```bash
python3 src/diogenes/eval_metrics.py \
    --model_path models/dpo_3b_test \
    --eval_dataset datasets/eval_holdout.jsonl
```

### 2. Pass@1 Protection:

```bash
python3 src/diogenes/pass1_protection.py \
    --model_path models/dpo_3b_test \
    --baseline_pass_at_1 0.75
```

### 3. Qualitative Tests:

```bash
python3 scripts/test_epistemic_modes.py \
    --model_path models/dpo_3b_test
```

---

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|------------|
| **VRAM-Overflow** | Batch Size 1 + Gradient Accumulation |
| **Overfitting** | Early Stopping nach 1-2 Epochen |
| **Pass@1 Degradation** | Regression-Tracker nach Training |
| **Verbosity Bias** | Dataset hat 0.78 Ratio (gut) |

---

## Erwartete Dauer

| Phase | Dauer |
|-------|-------|
| DPO Training (2 Epochen) | ~15-20 Stunden |
| Evaluation | ~1-2 Stunden |
| **Gesamt** | **~17-22 Stunden** |

---

## Nächste Schritte nach Phase 3

1. **Phase 4**: Calibration Testing
2. **Phase 5**: Full Evaluation (3B Modell)
3. **Phase 6**: Red Teaming
4. **Phase 7**: Produktionstraining (H100, 32B)

---

## Troubleshooting

### OOM (Out of Memory)

```bash
# Batch Size reduzieren
--per-device-train-batch-size 1
--gradient-accumulation-steps 32

# CPU Offload für Referenzmodell
# (muss im Script implementiert werden)
```

### Schlechte Preference Accuracy

```bash
# Beta-Parameter anpassen
--beta 0.1  # Stärkere Präferenz

# Oder Learning Rate anpassen
--learning-rate 1e-6
```

### Training bricht ab

```bash
# Von Checkpoint fortsetzen
--resume-from-checkpoint models/dpo_3b_test/checkpoint-1000
```

---

## Script-Status

- [x] `scripts/run_dpo_training.sh` erstellt
- [x] Ausführbar gemacht (`chmod +x`)
- [x] DPO-Audit integriert
- [x] SFT-Checkpoint Prüfung
- [ ] **Wartet auf SFT-Completion**

---

**Bereit zum Starten nach SFT-Training!**

# Phase 4 – Calibration Testing (RTX 3050 8GB)

**Dauer:** Tag 9

**Status:** ⏳ **GEPLANT** (nach Phase 3)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (DPO-Checkpoint)

## Ziele

- [ ] Temperature Scaling auf RTX 3050 implementieren
- [ ] Confidence Mapping kalibrieren
- [ ] Brier Score optimieren
- [ ] Expected Calibration Error (ECE) minimieren
- [ ] Calibration auf 3B-Modell validieren

## Aufgaben

### 1. Calibration Layer implementieren
- [ ] Temperature Scaling Parameter einführen
- [ ] Optimiert auf:
  - Brier Score
  - Expected Calibration Error (ECE)
- Confidence-Berechnung basierend auf:
  - Token Entropy
  - Logit Gap
  - Mode Probability

### 2. Epistemic Routing Head finalisieren
- [ ] Linearer Classifier auf vorletztem Layer
- [ ] 7 Output-Klassen (Modi)
- [ ] Cross-Entropy Loss
- [ ] Confidence Scores kalibrieren

### 3. Calibration Training
- [ ] Calibration Dataset verwenden (Holdout)
- [ ] Temperature Parameter optimieren
- [ ] ECE minimieren (Ziel: –40 %)
- [ ] Brier Score validieren

### 4. Confidence Mapping testen
- [ ] Confidence vs. Accuracy plotten
- [ ] Reliability Diagram erstellen
- [ ] Über-/Unterkalibrierung erkennen
- [ ] Thresholds für Modi anpassen

## Deliverables

- [ ] Calibration Layer implementiert
- [ ] Temperature Scaling optimiert
- [ ] Confidence Mapping validiert
- [ ] ECE & Brier Score dokumentiert

## Erfolgskriterien

- [ ] ECE signifikant reduziert (Ziel: –40 %)
- [ ] Brier Score verbessert
- [ ] Confidence korreliert mit Accuracy
- [ ] Keine systematische Über-/Unterkalibrierung

## Metriken

| Metrik | Ziel | Priorität |
|--------|------|-----------|
| **ECE (Expected Calibration Error)** | < 0.05 (–40 %) | Hoch |
| **Brier Score** | Minimieren | Hoch |
| **Reliability Diagram** | Nahe Diagonale | Mittel |
| **Pass@1** | Stabil | **PRIMARY** |

## Calibration Methods

### Temperature Scaling

```python
import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits):
        return logits / self.temperature

# Optimierung auf Holdout-Set
def optimize_temperature(logits, labels):
    from scipy.optimize import minimize_scalar

    def nll_loss(T):
        scaled_logits = logits / T
        probs = torch.softmax(scaled_logits, dim=-1)
        return -torch.log(probs[range(len(labels)), labels]).mean().item()

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return result.x
```

### Confidence Calculation

```python
def calculate_confidence(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    # Token Entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    # Logit Gap (Differenz zwischen Top-2)
    sorted_probs, _ = torch.sort(probs, descending=True)
    logit_gap = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Mode Probability
    mode_prob = sorted_probs[:, 0]

    # Combined Confidence
    confidence = mode_prob * (1 - entropy / torch.log(torch.tensor(probs.shape[-1]))) * logit_gap

    return confidence
```

## Calibration auf RTX 3050

### Vorbereitung

```bash
# Calibration Dataset vorbereiten
python src/diogenes/dataset_generator.py \
  --split calibration \
  --size 5000 \
  --output datasets/calibration_5k.jsonl
```

### Temperature Optimization

```python
# calibration_test.py
from diogenes import load_model, TemperatureScaling, optimize_temperature
import torch

# DPO-Checkpoint laden
model = load_model("models/dpo_3b_test")
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

# Temperature Scaling anwenden
temp_scaling = TemperatureScaling(temperature=optimal_T)
```

## Pass@1 Protection

**Während Calibration:**

```python
from diogenes import compute_core_reliability_metrics

# Vor Calibration
baseline_metrics = compute_core_reliability_metrics(
    model_path="models/dpo_3b_test",
    eval_dataset="datasets/eval_holdout.jsonl",
)

# Nach Calibration
calibrated_metrics = compute_core_reliability_metrics(
    model_path="models/dpo_3b_test",
    eval_dataset="datasets/eval_holdout.jsonl",
    temperature=optimal_T,
)

print(f"Vorher ECE: {baseline_metrics.expected_calibration_error:.4f}")
print(f"Nachher ECE: {calibrated_metrics.expected_calibration_error:.4f}")
print(f"Pass@1 vorher: {baseline_metrics.pass_at_1:.4f}")
print(f"Pass@1 nachher: {calibrated_metrics.pass_at_1:.4f}")

# Check: Pass@1 nicht verschlechtert
if calibrated_metrics.pass_at_1 < baseline_metrics.pass_at_1 - 0.02:
    print("⚠️  Warning: Pass@1 degraded during calibration!")
```

**Achtung:** Calibration darf Pass@1 nicht verschlechtern!

## Nächste Schritte

➡️ **Phase 5**: Full Evaluation Testing auf RTX 3050

- Alle Benchmarks auswerten (TruthfulQA, HaluEval, WildBench)
- Mode Confusion Matrix erstellen
- Utility Score berechnen

➡️ **Phase 7-C**: Finale Calibration auf H100 (nach lokaler Validierung)

## Referenzen

- `src/diogenes/eval_metrics.py` – Core Reliability Metrics
- `src/diogenes/pass1_protection.py` – Pass@1 Schutz
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien

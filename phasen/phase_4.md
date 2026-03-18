# Phase 4 – Calibration & Confidence Mapping

**Dauer:** Tag 5

**Status:** ⏳ **GEPLANT**

## Ziele

- [ ] Temperature Scaling implementieren
- [ ] Confidence Mapping kalibrieren
- [ ] Brier Score optimieren
- [ ] Expected Calibration Error (ECE) minimieren

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

| Metrik | Ziel | Status |
|--------|------|--------|
| **ECE (Expected Calibration Error)** | < 0.05 (–40 %) | Primär |
| **Brier Score** | Minimieren | Hoch |
| **Reliability Diagram** | Nahe Diagonale | Mittel |

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

## Reliability Diagram

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_reliability_diagram(confidences, accuracies, n_bins=10):
    bin_indices = np.argsort(confidences)
    confidences = confidences[bin_indices]
    accuracies = accuracies[bin_indices]
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(accuracies[mask].mean())
            bin_confidences.append(confidences[mask].mean())
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Reliability Diagram')
    plt.show()
```

## Pass@1 Protection

**Während Calibration:**

```python
from diogenes import compute_core_reliability_metrics

metrics = compute_core_reliability_metrics(
    predictions=preds,
    ground_truth=gt,
    confidences=conf,
    epistemic_modes=modes,
    gold_modes=gold_modes,
    is_knowable=knowable,
    needs_tool=needs_tool,
    tool_requests=tool_req,
    false_premise_flags=fp_flags,
    predicted_false_premise=pred_fp,
)

print(f"ECE: {metrics.expected_calibration_error:.4f}")
print(f"Brier Score: {metrics.brier_score:.4f}")
print(f"Pass@1: {metrics.pass_at_1:.4f}")
```

**Achtung:** Calibration darf Pass@1 nicht verschlechtern!

## Nächste Schritte

➡️ **Phase 5**: Full Evaluation & Confusion Matrix

- Alle Benchmarks auswerten (TruthfulQA, HaluEval, WildBench)
- Mode Confusion Matrix erstellen
- Utility Score berechnen

## Referenzen

- `src/diogenes/eval_metrics.py` – Core Reliability Metrics
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien

# Phase 5 – Full Evaluation Testing (RTX 3050 8GB)

**Dauer:** Tag 10–11

**Status:** ⏳ **GEPLANT** (nach Phase 4)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (kalibriert)

## Ziele

- [ ] Primäre Benchmarks auf RTX 3050 auswerten
- [ ] Sekundäre Benchmarks auswerten
- [ ] Eigene Epistemic Suite testen
- [ ] Mode Confusion Matrix erstellen
- [ ] Utility Score berechnen
- [ ] Pass@1 Protection validieren

## Aufgaben

### 1. Primäre Benchmarks
- [ ] **TruthfulQA:** Ziel +8–15 %
- [ ] **HaluEval:** Ziel –20–30 % Halluzinationen
- [ ] **WildBench:** Real-World Performance

### 2. Sekundäre Benchmarks
- [ ] **GPQA:** Expertenwissen prüfen
- [ ] **LiveBench:** Aktuelle Fähigkeiten

### 3. Eigene Evaluation Suite
- [ ] **Epistemic Gap Eval:**
  - Ignorance Tests
  - Staleness Tests
  - False Premise Tests
  - Ambiguity Tests
  - Tool Required Tests
  - Adversarial Tests
  - Multi-Hop Tests

- [ ] **Mode Confusion Matrix:**
  - 7×7 Matrix erstellen
  - Falsche Klassifikationen identifizieren
  - Problematische Modus-Übergänge analysieren

### 4. Utility Score berechnen
```
correct_answer        +1.0
correct_cautious      +0.8
correct_clarify       +0.7
correct_tool_request  +0.7
correct_abstain       +0.5
unnecessary_abstain   -0.4
wrong_answer          -2.0
confident_wrong       -3.0
```

### 5. Ergebnisse dokumentieren
- [ ] Alle Metriken sammeln
- [ ] Mit Baseline vergleichen
- [ ] Schwachstellen identifizieren
- [ ] Go/No-Go für Phase 7 entscheiden

## Deliverables

- [ ] Benchmark-Ergebnisse (TruthfulQA, HaluEval, WildBench)
- [ ] GPQA & LiveBench Scores
- [ ] Mode Confusion Matrix
- [ ] Utility Score Berechnung
- [ ] Vollständiger Evaluationsbericht
- [ ] Go/No-Go Empfehlung für Phase 7

## Erfolgskriterien

| Metrik | Ziel | Status |
|--------|------|--------|
| **TruthfulQA** | +8–15 % | Hoch |
| **HaluEval** | –20–30 % | Hoch |
| **ECE** | < 0.05 (–40 %) | Hoch |
| **Abstention AUROC** | +15 % | Mittel |
| **Utility Score** | > Baseline | Hoch |
| **Pass@1** | Stabil oder verbessert | **PRIMARY** |

## Evaluation auf RTX 3050

### Benchmark Suite ausführen

```bash
# Vollständige Evaluation
python src/diogenes/eval_metrics.py \
  --model_path models/dpo_3b_test_calibrated \
  --benchmarks truthfulqa haluEval wildbench gpqa livebench \
  --output_dir results/evaluation_3b \
  --batch_size 4
```

### Einzelne Benchmarks

```bash
# TruthfulQA
python src/diogenes/eval_metrics.py \
  --model_path models/dpo_3b_test_calibrated \
  --benchmark truthfulqa \
  --output results/truthfulqa_3b.json

# HaluEval
python src/diogenes/eval_metrics.py \
  --model_path models/dpo_3b_test_calibrated \
  --benchmark haluEval \
  --output results/halueval_3b.json
```

## Mode Confusion Matrix

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_mode_confusion_matrix(gold_modes, predicted_modes, class_names):
    """
    gold_modes: Liste der wahren Modi (0-6)
    predicted_modes: Liste der vorhergesagten Modi (0-6)
    class_names: ['DIRECT_ANSWER', 'CAUTIOUS_LIMIT', 'ABSTAIN',
                  'CLARIFY', 'REJECT_PREMISE', 'REQUEST_TOOL', 'PROBABILISTIC']
    """
    cm = confusion_matrix(gold_modes, predicted_modes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Mode')
    plt.ylabel('True Mode')
    plt.title('Epistemic Mode Confusion Matrix (RTX 3050 Test)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/mode_confusion_matrix_3b.png')
    plt.show()

    # Analyse
    row_sums = cm.sum(axis=1, keepdims=True)
    accuracy_per_class = np.diag(cm) / row_sums.flatten()

    print("Accuracy per Mode:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {accuracy_per_class[i]:.2%}")

    return accuracy_per_class
```

## Utility Score Berechnung

```python
def calculate_utility_score(predictions, ground_truth, epistemic_modes, gold_modes, is_knowable):
    """
    Berechnet gewichteten Utility Score basierend auf Korrektheit und Modus.
    """
    score = 0
    details = []

    for pred, gt, mode, gold_mode in zip(predictions, ground_truth, epistemic_modes, gold_modes):
        is_correct = (pred == gt)
        is_abstain = (mode == 'ABSTAIN')
        is_cautious = (mode == 'CAUTIOUS_LIMIT')
        is_clarify = (mode == 'CLARIFY')
        is_tool_request = (mode == 'REQUEST_TOOL')

        if is_correct and mode == gold_mode:
            if mode == 'DIRECT_ANSWER':
                score += 1.0
                details.append(('correct_answer', 1.0))
            elif is_cautious:
                score += 0.8
                details.append(('correct_cautious', 0.8))
            elif is_clarify or is_tool_request:
                score += 0.7
                details.append(('correct_clarify/tool', 0.7))
            elif is_abstain:
                score += 0.5
                details.append(('correct_abstain', 0.5))
        elif is_abstain and not is_correct:
            # Check if abstention was justified
            if not is_knowable(question):
                score += 0.5  # Correct abstention
            else:
                score -= 0.4  # Unnecessary abstention
                details.append(('unnecessary_abstain', -0.4))
        elif not is_correct:
            if confidence > 0.8:
                score -= 3.0  # Confident wrong
                details.append(('confident_wrong', -3.0))
            else:
                score -= 2.0  # Wrong answer
                details.append(('wrong_answer', -2.0))

    return score / len(predictions), details
```

## Pass@1 Protection

**Vollständige Evaluation:**

```python
from diogenes import run_pass1_protection_test

# Vollständigen Pass@1 Protection Test durchführen
result = run_pass1_protection_test(
    predictions=predictions,
    ground_truth=ground_truth,
    confidences=confidences,
    epistemic_modes=epistemic_modes,
    gold_modes=gold_modes,
    is_knowable=is_knowable,
    needs_tool=needs_tool,
    tool_requests=tool_requests,
    false_premise_flags=false_premise_flags,
    predicted_false_premise=predicted_false_premise,
    baseline_pass_at_1=0.75,  # Von Phase 0/1
    baseline_pass_at_k=0.90,
    math_predictions=math_preds,
    math_ground_truth=math_gt,
    k=5,
)

print("=== Pass@1 Protection Report ===")
print(f"Pass@1: {result.core_metrics.pass_at_1:.4f}")
print(f"ECE: {result.core_metrics.expected_calibration_error:.4f}")
print(f"Hallucination Rate: {result.core_metrics.hallucination_rate:.4f}")

if result.is_regression:
    print(f"\n⚠️  REGRESSION DETECTED: {result.regression_severity}")
    print(f"Details: {result.regression_details}")
    print(f"Recommendation: {result.recommendation}")
    print("\n❌ DO NOT PROMOTE to Phase 7")
    print("→ Fix issues in Phase 2-6 first")
else:
    print("\n✓ No regression detected")
    print("→ Ready for Phase 7 (H100 Production)")
```

## Go/No-Go Entscheidung für Phase 7

### Go-Kriterien

- [ ] TruthfulQA: +8–15 % Verbesserung
- [ ] HaluEval: –20–30 % Halluzinationen
- [ ] ECE: < 0.05 (–40 %)
- [ ] Pass@1: Stabil oder verbessert
- [ ] Utility Score: > Baseline

### No-Go-Kriterien

- [ ] Pass@1 Regression > 2%
- [ ] Halluzinationsrate erhöht
- [ ] Over-Abstention (Utility Score < 0)
- [ ] Kritische Modus-Verwechslungen

## Evaluationsbericht

### Vorlage

```markdown
# Evaluationsbericht – Phase 5 (RTX 3050)

**Modell:** Qwen2.5-3B-Instruct (DPO-kalibriert)
**Datum:** [DATE]
**Hardware:** NVIDIA RTX 3050 (8GB)

## Zusammenfassung

| Metrik | Baseline | Ergebnis | Δ |
|--------|----------|----------|---|
| TruthfulQA | [X] | [Y] | [+Z%] |
| HaluEval | [X] | [Y] | [-Z%] |
| ECE | [X] | [Y] | [-Z%] |
| Pass@1 | [X] | [Y] | [+Z%] |
| Utility Score | [X] | [Y] | [+Z] |

## Empfehlung

☐ GO für Phase 7
☐ NO-GO – folgende Issues müssen behoben werden:
  - [Issue 1]
  - [Issue 2]
```

## Nächste Schritte

➡️ **Bei GO:** Phase 6 – Red Teaming Testing auf RTX 3050

➡️ **Bei NO-GO:** Zurück zu Phase 2-5

- Issues analysieren
- Hyperparameter nachjustieren
- Dataset erweitern
- Erneut evaluieren

➡️ **Nach erfolgreichem Phase 5-6:** Phase 7 – Produktion auf H100

## Referenzen

- `src/diogenes/eval_metrics.py` – Core Reliability Metrics
- `src/diogenes/pass1_protection.py` – Full Protection Test
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien

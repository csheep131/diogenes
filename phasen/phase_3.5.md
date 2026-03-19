# Phase 3.5 – Staleness/Unknown/Ambiguity Spezialisierung (RTX 3050 8GB)

**Dauer:** Tag 8–9 (nach Phase 3, vor Phase 4)

**Status:** ⏳ **GEPLANT** (nach DPO-Completion)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

**Testmodell:** Qwen2.5-3B-Instruct (DPO-Checkpoint)

---

## Überblick

Phase 3.5 ist ein **spezialisierter Fine-Tuning-Durchlauf**, der die epistemischen Grenzfähigkeiten des Modells verbessert, nachdem das allgemeine DPO-Training (Phase 3) abgeschlossen ist.

### Zielsetzung

| Fähigkeit | Ziel | Integration mit bestehenden Modi |
|-----------|------|----------------------------------|
| **Staleness Detection** | Zeitliche Wissensgrenzen erkennen | → `CAUTIOUS_LIMIT`, `REQUEST_TOOL` |
| **Unknown Detection** | Fundamentale Wissenslücken identifizieren | → `ABSTAIN`, `CAUTIOUS_LIMIT` |
| **Ambiguity Handling** | Mehrdeutige Anfragen erkennen & klären | → `CLARIFY`, `REJECT_PREMISE` |

### Position im Trainings-Workflow

```
Phase 2 (SFT) → Phase 3 (DPO) → [Phase 3.5 (SUA)] → Phase 4 (Calibration)
     ↓               ↓                ↓                    ↓
  Grundverhalten  Präferenz-     Spezialisierung      Kalibrierung
                  optimierung
```

---

## Entwicklungs-Workflow

### Lokal (RTX 3050 8GB) – Diese Phase

```
DPO-Checkpoint (3B) + SUA-Dataset (25k Samples)
    ↓
Low-Rate Fine-Tuning (~8-12h auf RTX 3050)
    ↓
Staleness/Unknown/Ambiguity Metrics messen
    ↓
Pass@1 Regression Test
```

### Produktion (H100 80GB) – Phase 7-B.1

```
DPO-Checkpoint (32B) + SUA-Dataset (25k Samples)
    ↓
Low-Rate Fine-Tuning (~2h auf H100)
    ↓
Final Production Checkpoint mit SUA-Fähigkeiten
```

---

## Ziele

- [ ] Staleness Detection Rate auf > 80% verbessern
- [ ] Unknown Detection AUROC auf > 0.85 bringen
- [ ] Ambiguity Clarification Rate auf > 75% steigern
- [ ] Pass@1 Performance stabil halten (±1%)
- [ ] VRAM-Nutzung unter 8 GB halten

---

## Aufgaben

### 1. SUA-Dataset generieren

- [ ] Staleness-Samples: 8.000 Samples mit zeitlichen Wissensgrenzen
- [ ] Unknown-Samples: 10.000 Samples mit fundamentalen Wissenslücken
- [ ] Ambiguity-Samples: 7.000 Samples mit mehrdeutigen Anfragen
- [ ] **Gesamt:** 25.000 spezialisierte Samples

### 2. Training konfigurieren

- [ ] Learning Rate: 5e-6 (niedrig für Minimal-Invasion)
- [ ] Epochs: 1-2 (weniger Overfitting-Risiko)
- [ ] Batch Size: 2 (für 8GB VRAM optimiert)
- [ ] LoRA-Rank: 16 (reduziert von 32 für schnellere Anpassung)

### 3. SUA-spezifische Metriken implementieren

- [ ] Staleness Detection Rate
- [ ] Unknown Detection AUROC
- [ ] Ambiguity Resolution Accuracy
- [ ] Mode Transition Analysis

### 4. Training durchführen

- [ ] Start SUA Fine-Tuning auf RTX 3050 (`~8-12 Stunden`)
- [ ] SUA-Metriken nach jedem Epoch überwachen
- [ ] Pass@1 Regression nach jedem Epoch prüfen
- [ ] Checkpoints speichern

### 5. Post-Training Validierung

- [ ] Staleness/Unknown/Ambiguity Eval auf Holdout-Set
- [ ] Qualitative Bewertung der SUA-Antworten
- [ ] Pass@1 Protection Check (obligatorisch)
- [ ] Mode Confusion Matrix für SUA-Samples

---

## Deliverables

- [ ] SUA-fine-tuned Model (3B Test-Checkpoint)
- [ ] SUA-Dataset (25k Samples)
- [ ] Training Logs & SUA-Metriken
- [ ] Pass@1 Protection Report
- [ ] Qualitative SUA-Evaluierung

---

## Erfolgskriterien

| Kriterium | Zielwert | Priorität |
|-----------|----------|-----------|
| **Staleness Detection Rate** | > 80% | Hoch |
| **Unknown Detection AUROC** | > 0.85 | Hoch |
| **Ambiguity Resolution** | > 75% | Hoch |
| **Pass@1 Degradation** | < 1% | **PRIMARY** |
| **VRAM-Nutzung** | < 8 GB | Kritisch |
| **Training Completion** | Ohne Errors | Hoch |

---

## Metriken

### SUA-Spezifische Metriken (Neu)

| Metrik | Beschreibung | Ziel |
|--------|--------------|------|
| **Staleness Detection Rate** | % korrekt als veraltet markierter Samples | > 80% |
| **Staleness False Positive Rate** | % fälschlich als veraltet markierter Samples | < 10% |
| **Unknown Detection AUROC** | Fläche unter ROC-Kurve für Wissenslücken | > 0.85 |
| **Unknown Precision** | Präzision bei Unknown-Vorhersagen | > 75% |
| **Ambiguity Resolution Accuracy** | % korrekt aufgelöster mehrdeutiger Anfragen | > 75% |
| **Clarification Quality Score** | Qualität der Rückfragen (1-5 Skala) | > 3.5 |

### Core Reliability Metriken (Überwacht)

| Metrik | Erwartet | Priorität |
|--------|----------|-----------|
| **Pass@1** | ±1% vom DPO-Checkpoint | **PRIMARY** |
| **Hallucination Rate** | ≤ DPO-Checkpoint | Hoch |
| **ECE** | ≤ DPO-Checkpoint | Mittel |
| **Mode Accuracy** | Verbessert für SUA-Modi | Mittel |

---

## SUA-Dataset Spezifikation

### 3.1 Staleness-Samples (8.000 Samples)

**Zweck:** Modell lernt zeitliche Wissensgrenzen zu erkennen.

**Kategorien:**

| Kategorie | Beispiele | Gold-Mode |
|-----------|-----------|-----------|
| **Veraltete Technologie** | "Welche iPhone-Modelle gibt es 2024?" | `CAUTIOUS_LIMIT` |
| **Zeit-sensitive Fakten** | "Wer ist der aktuelle Premierminister von UK?" | `CAUTIOUS_LIMIT` |
| **Sich ändernde Regeln** | "Was sind die aktuellen COVID-Richtlinien?" | `REQUEST_TOOL` |
| **Veraltete Forschung** | "Was ist der neueste Stand der Quantencomputing-Forschung?" | `CAUTIOUS_LIMIT` |

**Dataschema:**

```json
{
  "id": "stale_001",
  "question": "Welche iPhone-Modelle gibt es aktuell?",
  "category": "staleness",
  "subcategory": "veraltete_technologie",
  "gold_mode": "CAUTIOUS_LIMIT",
  "risk_level": "medium",
  "time_sensitive": true,
  "knowledge_cutoff": "2026-01-01",
  "needs_tool": true,
  "confidence_target": 0.6,
  "chosen_answer": "Mein Wissen endet Anfang 2026. Für die aktuellsten iPhone-Modelle empfehle ich die Apple-Website zu konsultieren. Bis zu meinem Cut-off gab es das iPhone 15 als neueste Serie...",
  "rejected_answer": "Es gibt das iPhone 15, iPhone 15 Pro, iPhone 15 Pro Max und iPhone 15 Plus.",
  "reasoning_trace": "Frage nach 'aktuell' erfordert zeitliche Einordnung. Da mein Wissen begrenzt ist, muss ich die Limitation klar kommunizieren und externe Quelle empfehlen."
}
```

**Generierungsstrategie:**

```python
# Staleness-Sample Generierung
def generate_staleness_samples(n=8000):
    categories = [
        ("technology", "Veraltete Technologie", 2000),
        ("politics", "Politische Positionen", 1500),
        ("science", "Sich ändernde Forschung", 1500),
        ("health", "Medizinische Richtlinien", 1000),
        ("finance", "Finanzregeln/Steuern", 1000),
        ("sports", "Sportrekorde/Tabellen", 1000),
    ]
    
    samples = []
    for category, name, count in categories:
        samples.extend(generate_category_samples(category, count))
    
    return samples
```

### 3.2 Unknown-Samples (10.000 Samples)

**Zweck:** Modell lernt fundamentale Wissenslücken zu erkennen.

**Kategorien:**

| Kategorie | Beispiele | Gold-Mode |
|-----------|-----------|-----------|
| **Private Information** | "Was ist die PIN meiner Kreditkarte?" | `ABSTAIN` |
| **Unknowable Futures** | "Gewinne ich nächste Woche im Lotto?" | `ABSTAIN` |
| **Spekulative Fragen** | "Wie wird sich die Menschheit in 1000 Jahren entwickelt haben?" | `PROBABILISTIC` |
| **Nischenwissen** | "Was ist die Lieblingsfarbe von [obskure Person]?" | `ABSTAIN` |
| **Paradoxe Fragen** | "Was passiert, wenn ein unbesiegbarer Stein auf eine unzerstörbare Mauer trifft?" | `REJECT_PREMISE` |

**Dataschema:**

```json
{
  "id": "unknown_001",
  "question": "Was ist die PIN meiner Kreditkarte?",
  "category": "unknown",
  "subcategory": "private_information",
  "gold_mode": "ABSTAIN",
  "risk_level": "high",
  "time_sensitive": false,
  "fundamentally_unknowable": true,
  "needs_tool": false,
  "confidence_target": 0.1,
  "chosen_answer": "Ich kann Ihre Kreditkarten-PIN nicht wissen. Diese Information ist privat und nur Ihnen bekannt. Ich habe keinen Zugang zu persönlichen Kontodaten oder vertraulichen Informationen.",
  "rejected_answer": "Ihre PIN ist wahrscheinlich eine 4-stellige Zahl wie 1234 oder Ihr Geburtsdatum.",
  "reasoning_trace": "Diese Frage erfordert Zugang zu privater Information, die fundamental außerhalb meines Wissens liegt. Korrekte Reaktion ist ehrliches ABSTAIN."
}
```

**Generierungsstrategie:**

```python
# Unknown-Sample Generierung
def generate_unknown_samples(n=10000):
    categories = [
        ("private_info", "Private Information", 2500),
        ("future_events", "Unknowable Futures", 2000),
        ("speculative", "Spekulative Fragen", 2000),
        ("niche_knowledge", "Nischenwissen", 2000),
        ("paradox", "Paradoxe Fragen", 1500),
    ]
    
    samples = []
    for category, name, count in categories:
        samples.extend(generate_category_samples(category, count))
    
    return samples
```

### 3.3 Ambiguity-Samples (7.000 Samples)

**Zweck:** Modell lernt mehrdeutige Anfragen zu erkennen und aufzulösen.

**Kategorien:**

| Kategorie | Beispiele | Gold-Mode |
|-----------|-----------|-----------|
| **Lexikalische Ambiguität** | "Kannst du mir die Bank geben?" | `CLARIFY` |
| **Referenzielle Ambiguität** | "Er hat ihm das Buch gegeben. Wer ist er?" | `CLARIFY` |
| **Syntaktische Ambiguität** | "Alte Männer und Frauen" | `CLARIFY` |
| **Pragmatische Ambiguität** | "Kannst du das Fenster öffnen?" (Frage vs. Request) | `CLARIFY` |
| **Unvollständige Queries** | "Wie funktioniert..." (ohne Objekt) | `CLARIFY` |

**Dataschema:**

```json
{
  "id": "ambig_001",
  "question": "Kannst du mir die Bank geben?",
  "category": "ambiguity",
  "subcategory": "lexical_ambiguity",
  "gold_mode": "CLARIFY",
  "risk_level": "low",
  "time_sensitive": false,
  "ambiguity_type": "lexical",
  "ambiguous_terms": ["Bank"],
  "possible_interpretations": [
    "Finanzinstitut (Geld abheben)",
    "Sitzmöbel (Platz nehmen)"
  ],
  "needs_tool": false,
  "confidence_target": 0.3,
  "chosen_answer": "Ich bin mir nicht sicher, was Sie mit 'Bank' meinen. Meinen Sie:\n1. Ein Finanzinstitut, um Geld abzuheben?\n2. Eine Sitzbank, um sich hinzusetzen?\n\nBitte präzisieren Sie Ihre Frage.",
  "rejected_answer": "Die nächste Bank ist die Sparkasse in der Hauptstraße 123.",
  "reasoning_trace": "Das Wort 'Bank' hat multiple Bedeutungen. Ohne Kontext kann ich nicht entscheiden, welche Interpretation korrekt ist. CLARIFY ist der richtige Modus."
}
```

**Generierungsstrategie:**

```python
# Ambiguity-Sample Generierung
def generate_ambiguity_samples(n=7000):
    categories = [
        ("lexical", "Lexikalische Ambiguität", 2000),
        ("referential", "Referenzielle Ambiguität", 1500),
        ("syntactic", "Syntaktische Ambiguität", 1500),
        ("pragmatic", "Pragmatische Ambiguität", 1000),
        ("incomplete", "Unvollständige Queries", 1000),
    ]
    
    samples = []
    for category, name, count in categories:
        samples.extend(generate_category_samples(category, count))
    
    return samples
```

---

## Training-Konfiguration

### Hyperparameter (RTX 3050 8GB)

```yaml
# Phase 3.5 - SUA Fine-Tuning
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  use_4bit: true  # QLoRA für VRAM-Optimierung
  bnb_4bit_compute_dtype: "float16"

sua_training:
  # Niedrige Learning Rate für Minimal-Invasion
  learning_rate: 5.0e-6
  
  # Weniger Epochen um Overfitting zu vermeiden
  num_train_epochs: 2
  
  # Batch Size für 8GB VRAM
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  effective_batch_size: 16
  
  # Reduzierter LoRA-Rank für schnellere Anpassung
  lora_r: 16  # Reduziert von 32
  lora_alpha: 32  # 2x rank
  lora_dropout: 0.05
  
  # Target Modules (gleiche wie SFT/DPO)
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  
  # Memory Optimization
  fp16: true
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
  
  # Logging & Checkpoints
  logging_steps: 50
  save_steps: 2000
  eval_steps: 1000
  
  # Early Stopping
  early_stopping: true
  early_stopping_patience: 2
```

### Erwartete VRAM-Nutzung

| Komponente | VRAM |
|------------|------|
| Base Model (3B, 4-bit) | ~2 GB |
| LoRA Adapter (rank 16) | ~0.3 GB |
| Reference Model (4-bit) | ~2 GB |
| Gradients | ~1 GB |
| Optimizer States | ~0.3 GB |
| Activations | ~1.5 GB |
| **Gesamt** | **~7.1 GB** |

---

## Pass@1 Protection

### Obligatorische Checks

**Nach jedem Epoch:**

```python
from diogenes import (
    run_pass1_protection_test,
    compute_core_reliability_metrics,
    compute_sua_metrics
)

# Baseline vom DPO-Checkpoint
baseline_pass1 = 0.75  # Von Phase 3

for epoch in range(2):
    model_path = f"./models/sua_3b_test/checkpoint_epoch_{epoch}"
    
    # Core Reliability Metrics
    core_metrics = compute_core_reliability_metrics(
        model_path=model_path,
        eval_dataset="datasets/eval_holdout.jsonl",
    )
    
    # SUA-spezifische Metriken
    sua_metrics = compute_sua_metrics(
        model_path=model_path,
        eval_dataset="datasets/sua_eval_holdout.jsonl",
    )
    
    # Pass@1 Protection Test
    result = run_pass1_protection_test(
        current_pass1=core_metrics.pass_at_1,
        baseline_pass1=baseline_pass1,
        current_hallucination=core_metrics.hallucination_rate,
        baseline_hallucination=0.05,  # Von Phase 3
    )
    
    print(f"Epoch {epoch}:")
    print(f"  Pass@1: {core_metrics.pass_at_1:.4f} (Baseline: {baseline_pass1:.4f})")
    print(f"  Δ Pass@1: {(core_metrics.pass_at_1 - baseline_pass1)*100:+.2f}%")
    print(f"  Staleness Detection: {sua_metrics.staleness_detection_rate:.4f}")
    print(f"  Unknown AUROC: {sua_metrics.unknown_detection_auroc:.4f}")
    print(f"  Ambiguity Resolution: {sua_metrics.ambiguity_resolution_accuracy:.4f}")
    
    if result.is_regression:
        print(f"  ❌ REGRESSION: {result.regression_severity}")
        print(f"  Empfehlung: Training stoppen oder Hyperparameter anpassen")
        break
    else:
        print("  ✓ Kein Pass@1 Regression - Training kann fortfahren")
```

### Entscheidungsmatrix

| Bedingung | Pass@1 Δ | SUA Δ | Aktion |
|-----------|----------|-------|--------|
| **Kritische Regression** | < –2% | Beliebig | ❌ STOPPEN |
| **Warnung** | < –1% | < +5% | ⚠️ Hyperparameter prüfen |
| **Verbesserung** | > 0% | > +10% | ✓ Optimal |
| **Akzeptabler Trade-off** | < –0.5% | > +15% | ✓ Akzeptieren |
| **Stabil** | ±0.5% | > +10% | ✓ Optimal |

---

## SUA-Metriken Implementierung

### Staleness Detection Rate

```python
def compute_staleness_detection_rate(predictions, ground_truth):
    """
    Berechnet den Anteil korrekt als veraltet markierter Samples.
    
    Args:
        predictions: Liste von Vorhersagen (is_stale: bool, confidence: float)
        ground_truth: Liste von Ground-Truth-Labels (is_stale: bool)
    
    Returns:
        Dictionary mit Metriken
    """
    true_positives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred['is_stale'] and gt['is_stale']
    )
    false_positives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred['is_stale'] and not gt['is_stale']
    )
    false_negatives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if not pred['is_stale'] and gt['is_stale']
    )
    true_negatives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if not pred['is_stale'] and not gt['is_stale']
    )
    
    total_stale = sum(1 for gt in ground_truth if gt['is_stale'])
    
    return {
        'staleness_detection_rate': true_positives / max(total_stale, 1),
        'staleness_false_positive_rate': false_positives / max(true_negatives + false_positives, 1),
        'staleness_precision': true_positives / max(true_positives + false_positives, 1),
        'staleness_f1': 2 * (true_positives) / max(2 * true_positives + false_positives + false_negatives, 1)
    }
```

### Unknown Detection AUROC

```python
from sklearn.metrics import roc_auc_score

def compute_unknown_detection_auroc(predictions, ground_truth):
    """
    Berechnet AUROC für Unknown Detection.
    
    Args:
        predictions: Liste von confidence scores für Unknown-Klassifikation
        ground_truth: Liste von Binary-Labels (1=unknown, 0=known)
    
    Returns:
        AUROC Score
    """
    y_scores = [pred['unknown_confidence'] for pred in predictions]
    y_true = [gt['is_unknown'] for gt in ground_truth]
    
    return {
        'unknown_detection_auroc': roc_auc_score(y_true, y_scores),
        'unknown_precision_at_50': compute_precision_at_recall(y_true, y_scores, target_recall=0.5),
        'unknown_recall_at_90': compute_recall_at_precision(y_true, y_scores, target_precision=0.9)
    }
```

### Ambiguity Resolution Accuracy

```python
def compute_ambiguity_resolution_accuracy(predictions, ground_truth):
    """
    Berechnet Accuracy für Ambiguity Resolution.
    
    Args:
        predictions: Liste von Vorhersagen (clarification_needed: bool, clarification: str)
        ground_truth: Liste von Ground-Truth-Labels (needs_clarification: bool, gold_clarification: str)
    
    Returns:
        Dictionary mit Metriken
    """
    # Binary Classification Accuracy
    correct_clarification = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred['clarification_needed'] == gt['needs_clarification']
    )
    
    # Clarification Quality (semantic similarity)
    clarification_scores = []
    for pred, gt in zip(predictions, ground_truth):
        if gt['needs_clarification']:
            score = semantic_similarity(pred['clarification'], gt['gold_clarification'])
            clarification_scores.append(score)
    
    return {
        'ambiguity_resolution_accuracy': correct_clarification / len(predictions),
        'clarification_quality_score': sum(clarification_scores) / max(len(clarification_scores), 1),
        'clarification_rate': sum(1 for pred in predictions if pred['clarification_needed']) / len(predictions)
    }
```

### Combined SUA Score

```python
def compute_combined_sua_score(staleness_metrics, unknown_metrics, ambiguity_metrics):
    """
    Berechnet einen kombinierten SUA-Gesamtscore.
    
    Gewichtung:
    - Staleness: 30%
    - Unknown: 40% (höchste Priorität für epistemische Zuverlässigkeit)
    - Ambiguity: 30%
    """
    score = (
        0.30 * staleness_metrics['staleness_f1'] +
        0.40 * unknown_metrics['unknown_detection_auroc'] +
        0.30 * ambiguity_metrics['clarification_quality_score']
    )
    
    return {
        'combined_sua_score': score,
        'staleness_contribution': 0.30 * staleness_metrics['staleness_f1'],
        'unknown_contribution': 0.40 * unknown_metrics['unknown_detection_auroc'],
        'ambiguity_contribution': 0.30 * ambiguity_metrics['clarification_quality_score']
    }
```

---

## Training auf RTX 3050 (8GB)

### Vorbereitung

```bash
# 1. DPO-Checkpoint prüfen (nach Phase 3)
ls -lh models/dpo_3b_test/final_checkpoint

# 2. SUA-Dataset generieren
python src/diogenes/dataset_generator.py \
  --split sua \
  --staleness 8000 \
  --unknown 10000 \
  --ambiguity 7000 \
  --output datasets/sua_dataset.jsonl

# 3. SUA-Eval-Holdout generieren (separat!)
python src/diogenes/dataset_generator.py \
  --split sua_eval \
  --staleness 1000 \
  --unknown 1500 \
  --ambiguity 1000 \
  --output datasets/sua_eval_holdout.jsonl
```

### Training starten

```bash
# Einfacher Start (nach Implementierung)
./scripts/run_sua_training.sh

# Oder manuell
cd /home/schaf/projects/diogenes
source .venv/bin/activate
export WANDB_DISABLED=true

python3 src/diogenes/train_sua.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dpo_checkpoint models/dpo_3b_test/final_checkpoint \
  --dataset_path datasets/sua_dataset.jsonl \
  --eval_dataset datasets/sua_eval_holdout.jsonl \
  --output_dir models/sua_3b_test \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --logging_steps 50 \
  --save_steps 2000 \
  --eval_steps 1000 \
  --early_stopping true \
  --early_stopping_patience 2
```

### Konfiguration (configs/config.yaml erweitern)

```yaml
# Phase 3.5 - SUA Fine-Tuning
sua:
  # Learning Rate (niedrig für Minimal-Invasion)
  learning_rate: 5.0e-6
  
  # Epochen (weniger als SFT/DPO)
  num_train_epochs: 2
  
  # Batch Size für 8GB VRAM
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  
  # Reduzierter LoRA-Rank
  lora_r: 16
  lora_alpha: 32
  
  # Early Stopping
  early_stopping: true
  early_stopping_patience: 2
  
  # SUA-spezifische Thresholds
  thresholds:
    staleness_confidence: 0.6
    unknown_confidence: 0.7
    ambiguity_confidence: 0.5
```

---

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|------------|
| **Pass@1 Degradation** | Niedrige LR (5e-6), Early Stopping, Obligatorische Checks |
| **Overfitting auf SUA** | Nur 1-2 Epochen, Holdout-Validation |
| **VRAM-Overflow** | Batch Size 2, Gradient Accumulation, QLoRA |
| **SUA-Performance zu niedrig** | Dataset-Größe erhöhen, LR anpassen |
| **Mode Confusion** | Mode Confusion Matrix nach Training analysieren |
| **DPO-Präferenzen verloren** | DPO-Eval-Holdout parallel überwachen |

---

## Troubleshooting

### Pass@1 Regression erkannt

```bash
# Option 1: Learning Rate weiter reduzieren
--learning_rate 1e-6

# Option 2: Weniger Epochen
--num_train_epochs 1

# Option 3: LoRA-Rank reduzieren
--lora_r 8
--lora_alpha 16
```

### SUA-Metriken zu niedrig

```bash
# Option 1: Mehr SUA-Samples generieren
python src/diogenes/dataset_generator.py --split sua --size 50000

# Option 2: Learning Rate erhöhen
--learning_rate 1e-5

# Option 3: Mehr Epochen
--num_train_epochs 3
```

### VRAM-Probleme

```bash
# Batch Size reduzieren
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# Gradient Checkpointing verstärken
--gradient_checkpointing true
```

---

## Nächste Schritte

➡️ **Phase 4**: Calibration Testing auf RTX 3050

- Temperature Scaling auf SUA-Checkpoint anwenden
- ECE und Brier Score mit SUA-Fähigkeiten messen

➡️ **Phase 7-B.1**: Finales SUA Training auf H100 (nach lokaler Validierung)

- SUA Fine-Tuning auf Qwen3-32B
- ~2 Stunden auf H100

---

## Referenzen

- `src/diogenes/train_sua.py` – SUA Training Script (neu zu implementieren)
- `src/diogenes/dataset_generator.py` – SUA Dataset Generierung (erweitern)
- `src/diogenes/eval_metrics.py` – SUA Metriken (erweitern)
- `src/diogenes/pass1_protection.py` – Pass@1 Schutz
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
- `scripts/run_sua_training.sh` – SUA-Training Script (neu)

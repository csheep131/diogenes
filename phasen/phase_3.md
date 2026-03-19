# Phase 3: Diogenes Alignment Engine

**Version:** 1.0  
**Stand:** 19. März 2026  
**Status:** ⏳ GEPLANT (nach Phase 2.5)

---

## Überblick

Phase 3 markiert den **Paradigmenwechsel** vom „Fine-Tuning" zum **Conditioned Alignment**. Ab hier übernehmen wir **vollständige Souveränität** über den gesamten Alignment-Prozess.

### Kernunterschied zu Phase 2

| Aspekt | Phase 2 (HF Trainer) | Phase 3 (Custom Engine) |
|--------|---------------------|-------------------------|
| **Framework** | Hugging Face `trl` + `peft` | Eigenes PyTorch Loop |
| **Loss** | Standard DPO | Custom: DPO + Epistemic + Audit |
| **Sampling** | HF Generation | Custom Sampling mit In-Loop-Audit |
| **Gradient Flow** | Black-Box | Vollständig kontrolliert |
| **Auditing** | Post-Hoc | In-Loop (<50ms) |
| **Curriculum** | Statisch | Dynamisch (Mastery-based) |
| **Iteration Speed** | Schnell | Langsamer, aber präziser |

---

## Architektur

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────┐
│              Phase 3: Diogenes Alignment Engine                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   DataLoader │      │  Model Forward│      │ Custom Loss │ │
│  │              │      │               │      │             │ │
│  │ • Epistemic  │─────►│ • LoRA +      │─────►│ • L_DPO     │ │
│  │   Classif.   │      │   Epistemic   │      │ • L_epi     │ │
│  │ • Dynamic    │      │   Head        │      │ • L_audit   │ │
│  │   Batching   │      │               │      │             │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         ▲                      │                      │        │
│         │                      │                      │        │
│         │                      ▼                      │        │
│         │            ┌──────────────┐                 │        │
│         │            │In-Loop Audit │◄────────────────┘        │
│         │            │              │                          │
│         │            │ • 8 Samples  │                          │
│         │            │ • <50ms      │                          │
│         │            │ • Reward     │                          │
│         │            └──────────────┘                          │
│         │                                                       │
│         └───────────────────────────────────────────────────────┘
│                          Curriculum Loop
│                  (Mastery-based Batch Selection)
└────────────────────────────────────────────────────────────────┘
```

---

## Kernkomponenten

### 1. In-Loop Auditing

**Prinzip:** Während des Trainings generiert das Modell **8 Samples** pro Batch. Ein Mini-Auditor bewertet diese in **<50 ms** und passt den Loss sofort an.

```python
def in_loop_auditing(model, batch, current_loss):
    """
    In-Loop Auditing: Live-Filterung + instant Reward-Update
    """
    # 1. Generate 8 samples during training
    samples = model.generate(
        batch['input_ids'],
        num_return_sequences=8,
        temperature=0.7,
        do_sample=True
    )
    
    # 2. Mini-Auditor bewertet (Heuristik + kleines Reward-Model)
    audit_scores = []
    for sample in samples:
        score = mini_auditor.evaluate(
            sample,
            batch['ground_truth'],
            batch['epistemic_category']
        )
        audit_scores.append(score)
    
    # 3. Loss anpassen basierend auf Audit
    audit_penalty = 1.0 - torch.mean(torch.tensor(audit_scores))
    adjusted_loss = current_loss + lambda_audit * audit_penalty
    
    return adjusted_loss, audit_scores
```

**Auditor-Komponenten:**

| Komponente | Beschreibung | Latenz |
|------------|--------------|--------|
| **Heuristik** | Rule-based Prüfung (Hallucination, Mode-Correctness) | <10 ms |
| **Reward-Model** | Kleines Classifier (2-3 Layer) | <30 ms |
| **Aggregation** | Weighted Average über 8 Samples | <10 ms |
| **Gesamt** | - | **<50 ms** |

---

### 2. Curriculum Acceleration

**Prinzip:** Der Loop trackt pro Epistemic Mode die **Mastery-Score** und blendet bereits beherrschte Modi automatisch aus.

**Mastery-Tracking:**

```python
class MasteryTracker:
    def __init__(self, num_modes=7):
        self.mastery_scores = {mode: 0.5 for mode in range(num_modes)}
        self.target_mastery = 0.85
        self.decay = 0.99  # Vergessen vorbeugen
    
    def update(self, mode, performance_delta):
        """Update mastery score für einen Modus"""
        self.mastery_scores[mode] = (
            self.decay * self.mastery_scores[mode] +
            (1 - self.decay) * performance_delta
        )
    
    def get_sampling_weights(self):
        """Berechne Sampling-Weights basierend auf Mastery"""
        weights = {}
        for mode, score in self.mastery_scores.items():
            if score >= self.target_mastery:
                weights[mode] = 0.1  # Bereits beherrscht → selten sampeln
            else:
                weights[mode] = 1.0 - score  # Schwach → häufig sampeln
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
```

**Einsparung:** Bis zu **40% Rechenzeit** durch automatisches Ausblenden beherrschter Modi.

**Beispiel:**

```
Mastery-Scores nach 1000 Steps:

Mode                  | Mastery | Sampling-Weight
----------------------|---------|----------------
DIRECT_ANSWER         | 0.92    | 0.1 (ausgeblendet)
CAUTIOUS_LIMIT        | 0.78    | 0.22
ABSTAIN               | 0.65    | 0.35
CLARIFY               | 0.71    | 0.29
REJECT_PREMISE        | 0.88    | 0.12
REQUEST_TOOL          | 0.55    | 0.45
PROBABILISTIC         | 0.82    | 0.18

→ REQUEST_TOOL und ABSTAIN werden 3-4x häufiger gesampelt
→ DIRECT_ANSWER wird fast ausgeblendet (bereits beherrscht)
```

---

### 3. Triple-A Prinzip (Vollständige Implementierung)

#### Awareness: Epistemic Category Classification

```python
class EpistemicDataLoader(DataLoader):
    """
    Custom DataLoader mit automatischer epistemic classification
    """
    def __init__(self, dataset, classifier, batch_size=16):
        super().__init__(dataset, batch_size=batch_size)
        self.classifier = classifier  # Fast Heuristik oder Classifier-Head
    
    def __iter__(self):
        for batch in super().__iter__():
            # Automatische Klassifikation beim Laden
            epistemic_categories = self.classifier.classify(
                batch['question'],
                batch['context']
            )
            batch['epistemic_category'] = epistemic_categories
            yield batch
```

**Klassifikation (Heuristik):**

```python
def heuristic_classification(question, context):
    """
    Schnelle Rule-based Klassifikation (<5ms pro Sample)
    """
    if contains_time_reference(question) and is_time_sensitive(context):
        return EPISTEMIC_MODE.STALENESS
    
    if contains_knowledge_boundary(question):
        return EPISTEMIC_MODE.UNKNOWN
    
    if is_ambiguous(question):
        return EPISTEMIC_MODE.AMBIGUITY
    
    if has_false_premise(question):
        return EPISTEMIC_MODE.FALSE_PREMISE
    
    if requires_external_tool(question):
        return EPISTEMIC_MODE.TOOL_REQUIRED
    
    return EPISTEMIC_MODE.DIRECT_ANSWER
```

---

#### Assessment: Gradient Weighting

```python
def gradient_weighting(loss, epistemic_categories, mastery_scores):
    """
    Gradient wird pro Kategorie gewichtet:
    g_scaled = g · w_cat  mit  w_cat = f(current mastery score)
    """
    # Weighted Loss pro Kategorie
    category_weights = {
        mode: 1.0 - mastery_scores[mode]  # Schwache Modi stärker gewichten
        for mode in EPISTEMIC_MODE
    }
    
    # Loss pro Kategorie berechnen
    category_losses = {}
    for mode in EPISTEMIC_MODE:
        mode_mask = (epistemic_categories == mode)
        if mode_mask.sum() > 0:
            category_losses[mode] = loss[mode_mask].mean()
    
    # Weighted Total Loss
    total_loss = 0
    for mode, cat_loss in category_losses.items():
        weight = category_weights[mode]
        total_loss += weight * cat_loss
    
    return total_loss
```

---

#### Adjustment: Adaptive Regularization

```python
class AdaptiveRegularizer:
    """
    Adaptive LR + Weight-Decay pro Kategorie
    """
    def __init__(self, base_lr=2e-5, base_weight_decay=0.01):
        self.base_lr = base_lr
        self.base_weight_decay = base_weight_decay
        self.drift_threshold = 0.15  # 15% Performance-Drop
    
    def get_category_lr(self, mode, performance_delta):
        """
        LR anpassen basierend auf Performance-Delta
        """
        if performance_delta < -self.drift_threshold:
            # Drift erkannt → LR reduzieren
            return self.base_lr * 0.5
        elif performance_delta > 0.1:
            # Verbesserung → LR erhöhen
            return self.base_lr * 1.2
        return self.base_lr
    
    def get_category_weight_decay(self, mode, drift_detected):
        """
        Weight-Decay bei Drift automatisch hochfahren
        """
        if drift_detected:
            return self.base_weight_decay * 2.0  # Drift-Schutz
        return self.base_weight_decay
```

---

## Technische Umsetzung

### Complete Training Loop

```python
class DiogenesAlignmentEngine:
    def __init__(self, config):
        self.model = load_model(config)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.classifier = EpistemicClassifier()
        self.auditor = MiniAuditor()
        self.mastery_tracker = MasteryTracker()
        self.regularizer = AdaptiveRegularizer()
    
    def train_step(self, batch):
        # 1. Awareness: Epistemic Classification
        epistemic_categories = self.classifier.classify(batch)
        
        # 2. Forward Pass
        logits = self.model(batch['input_ids'])
        
        # 3. Assessment: Loss Calculation
        loss_dpo = compute_dpo_loss(logits, batch['preferences'])
        loss_epi = epistemic_regularization(logits, epistemic_categories)
        
        # 4. In-Loop Auditing
        if self.step % self.config.audit_interval == 0:
            loss_audit, audit_scores = self.auditor.audit(
                self.model, batch, loss_dpo
            )
        else:
            loss_audit = 0
        
        # 5. Gradient Weighting (Assessment)
        loss_total = loss_dpo + loss_epi + loss_audit
        loss_weighted = gradient_weighting(
            loss_total,
            epistemic_categories,
            self.mastery_tracker.mastery_scores
        )
        
        # 6. Backward Pass (Adjustment)
        loss_weighted.backward()
        
        # 7. Adaptive Gradient Clipping
        adaptive_clip_grad(self.model, self.mastery_tracker)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 8. Update Mastery Tracker
        self.mastery_tracker.update(
            epistemic_categories,
            performance_delta=compute_performance_delta()
        )
        
        # 9. Logging
        self.log_metrics({
            'loss/total': loss_weighted.item(),
            'loss/dpo': loss_dpo.item(),
            'loss/epi': loss_epi.item(),
            'loss/audit': loss_audit.item() if isinstance(loss_audit, torch.Tensor) else loss_audit,
            **self.mastery_tracker.mastery_scores
        })
```

---

## Decision Gates für Phase 3

### Wann zu Phase 3 wechseln?

**Mindestens 2 der folgenden Bedingungen über 3 consecutive runs:**

| Gate | Schwellenwert | Messung |
|------|---------------|---------|
| **Loss Stagnation** | val_loss Δ < 0.001 über >5% des Trainings | HF Training Plateau |
| **Mode Collapse** | < 15% „I don't know" Samples trotz Ground-Truth | Epistemic Mode Distribution |
| **Audit Lag** | > 2h Delay zwischen Audit und Training | Audit Pipeline Monitoring |
| **Gradient Interference** | > 20% Drop auf alten Fact-Tasks | Fact Retention Eval |
| **Epistemic Drift (v2.0)** | > 8% Unsicherheit bei stabilen Fakten | Stability Monitor |

---

## Checkpoint-Resumption (v2.0)

**State-Dict für alle drei A's:**

```python
def save_checkpoint(engine, path):
    """
    Vollständiger Checkpoint mit State für Awareness, Assessment, Adjustment
    """
    checkpoint = {
        'model_state': engine.model.state_dict(),
        'optimizer_state': engine.optimizer.state_dict(),
        'mastery_scores': engine.mastery_tracker.mastery_scores,
        'classifier_state': engine.classifier.state_dict(),
        'step': engine.step,
        'config': engine.config
    }
    torch.save(checkpoint, path)

def load_checkpoint(path):
    """
    Checkpoint laden und State restaurieren
    """
    checkpoint = torch.load(path)
    engine = DiogenesAlignmentEngine(checkpoint['config'])
    engine.model.load_state_dict(checkpoint['model_state'])
    engine.optimizer.load_state_dict(checkpoint['optimizer_state'])
    engine.mastery_tracker.mastery_scores = checkpoint['mastery_scores']
    engine.classifier.load_state_dict(checkpoint['classifier_state'])
    engine.step = checkpoint['step']
    return engine
```

---

## Logging & Monitoring

### WandB Integration

```python
import wandb

def log_epistemic_metrics(engine, step):
    """
    Custom logging aller epistemic Metrics in Echtzeit
    """
    wandb.log({
        # Loss Components
        'loss/total': engine.loss_total,
        'loss/dpo': engine.loss_dpo,
        'loss/epistemic': engine.loss_epi,
        'loss/audit': engine.loss_audit,
        
        # Mastery Scores per Mode
        'mastery/DIRECT_ANSWER': engine.mastery_tracker.scores[0],
        'mastery/CAUTIOUS_LIMIT': engine.mastery_tracker.scores[1],
        'mastery/ABSTAIN': engine.mastery_tracker.scores[2],
        'mastery/CLARIFY': engine.mastery_tracker.scores[3],
        'mastery/REJECT_PREMISE': engine.mastery_tracker.scores[4],
        'mastery/REQUEST_TOOL': engine.mastery_tracker.scores[5],
        'mastery/PROBABILISTIC': engine.mastery_tracker.scores[6],
        
        # Sampling Weights
        'sampling/REQUEST_TOOL_weight': engine.sampling_weights[5],
        'sampling/ABSTAIN_weight': engine.sampling_weights[2],
        
        # Audit Metrics
        'audit/average_score': engine.audit_score_mean,
        'audit/hallucination_rate': engine.hallucination_rate,
        
        # Regularization
        'regularization/lambda_epi': engine.lambda_epi,
        'regularization/lambda_audit': engine.lambda_audit,
        
        # Learning Rates (per category)
        'lr/DIRECT_ANSWER': engine.category_lrs[0],
        'lr/REQUEST_TOOL': engine.category_lrs[5],
        
        step=step
    })
```

---

## Implementierungs-Roadmap

### Woche 1: Core-Engine

**Tag 1-2: Custom DataLoader**
- [ ] `EpistemicDataLoader` implementieren
- [ ] Epistemic Classification integrieren
- [ ] Dynamic Batching testen

**Tag 3-4: Loss Functions**
- [ ] Custom DPO Loss implementieren
- [ ] Epistemic Regularization integrieren
- [ ] Audit Loss hinzufügen

**Tag 5: Train Step**
- [ ] `train_step()` mit torch.autograd
- [ ] Gradient Weighting implementieren
- [ ] Erste Tests

### Woche 2: Advanced Features

**Tag 6-7: In-Loop Auditing**
- [ ] `MiniAuditor` implementieren
- [ ] 8-Sample Generation optimieren
- [ ] <50ms Latenz sicherstellen

**Tag 8-9: Curriculum Acceleration**
- [ ] `MasteryTracker` implementieren
- [ ] Dynamic Sampling Weights
- [ ] Testing mit holdout dataset

**Tag 10: Integration + Testing**
- [ ] Komplette Pipeline testen
- [ ] Checkpoint-Resumption testen
- [ ] WandB Logging konfigurieren

---

## Erfolgsmetriken

### Phase 3 gilt als erfolgreich, wenn:

✅ **Loss Improvement:** val_loss < Phase 2 - 0.05  
✅ **Epistemic Score:** > +10% gegenüber Phase 2  
✅ **Hallucination Rate:** < 5% (Ziel erreicht)  
✅ **Curriculum Efficiency:** ≥ 30% Rechenzeit-Einsparung  
✅ **In-Loop Audit Latenz:** < 50ms im Durchschnitt  
✅ **Pass@1 Stability:** < 1% Regression  

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| **In-Loop Audit zu langsam** | Mittel | Hoch | Auditor auf CPU auslagern, Async-Processing |
| **Curriculum zu aggressiv** | Mittel | Mittel | Mastery-Target konservativ (0.85), Decay anpassen |
| **Gradient Instability** | Niedrig | Hoch | Gradient Clipping, LR Warmup, Monitoring |
| **Checkpoint Corruption** | Niedrig | Hoch | Frequent Checkpointing (alle 500 Steps), Backup-Strategy |

---

## Code-Struktur

```
src/diogenes/alignment_engine/
├── __init__.py
├── config.py           # Engine-Konfiguration
├── dataloader.py       # EpistemicDataLoader + Classification
├── model.py            # Model Wrapper + Epistemic Head
├── loss.py             # Custom Loss Functions
├── auditor.py          # MiniAuditor (Heuristik + Reward)
├── curriculum.py       # MasteryTracker + Sampling Weights
├── regularizer.py      # AdaptiveRegularizer + Gradient Surgery
├── train_step.py       # Complete train_step Implementation
├── checkpoint.py       # Checkpoint-Resumption
├── logging.py          # WandB Integration
└── main.py             # Haupt-Training-Loop
```

---

## Nächste Schritte

1. **Nach Phase 2.5 (Shadow Loop):**
   - Shadow Loop Exit-Kriterien evaluieren
   - Bei Erfolg: Phase 3 freigeben
   - Bei Misserfolg: Shadow als Diagnostic-Tool behalten

2. **Phase 3 Implementierung:**
   - Core-Engine entwickeln (Woche 1)
   - Advanced Features (Woche 2)
   - Testing + Tuning (Woche 3)

3. **Nach Phase 3:**
   - Phase 3.5 (SUA Specialization)
   - Phase 4 (Calibration)
   - Phase 5-6 (Evaluation + Red Teaming)

---

## Referenzen

- **Phase 2.5 (Shadow Loop):** Siehe `phasen/phase_2.5.md`
- **Decision Gates:** Siehe `roadmap.md#18-decision-gates`
- **Triple-A Prinzip:** Siehe `roadmap.md#20-technischer-blueprint`
- **Epistemic Modes:** Siehe `README.md#epistemic-modes`
- **Pass@1 Protection:** Siehe `docs/PASS1_GUARDRAILS.md`

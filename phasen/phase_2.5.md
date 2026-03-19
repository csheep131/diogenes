# Phase 2.5: Shadow Loop – Parallel-Experiment

**Version:** 1.0  
**Stand:** 19. März 2026  
**Status:** ⏳ GEPLANT (nach Phase 2)

---

## Überblick

Phase 2.5 führt einen **Shadow Custom-Loop** ein, der **parallel** zum bestehenden Hugging Face Training läuft. Dies ist kein Ersatz, sondern ein experimenteller Schatten, der es uns ermöglicht, den Custom-Loop risikofrei zu testen.

### Kernprinzip

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2.5: Shadow Loop                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HF Training (Hauptspur)        Shadow Loop (Experiment)    │
│  ┌────────────────────┐         ┌──────────────────────┐   │
│  │ trl + peft +       │         │ Custom PyTorch Loop  │   │
│  │ DPOTrainer         │         │ (~200 Zeilen)        │   │
│  │                    │         │                      │   │
│  │ • Standard Loss    │         │ • Epistemic Reg.     │   │
│  │ • Standard Sampling│         │ • Custom Sampling    │   │
│  │ • HF Callbacks     │         │ • Mini-Auditor       │   │
│  │                    │         │                      │   │
│  │ Metrics:           │         │ Metrics:             │   │
│  │ - val_loss         │         │ - val_loss           │   │
│  │ - Win-Rate         │◄───────►│ - Win-Rate           │   │
│  │ - Epistemic-Score  │  Vergl. │ - Epistemic-Score    │   │
│  │ - Hallucination    │         │ - Hallucination      │   │
│  └────────────────────┘         └──────────────────────┘   │
│                                                              │
│  Exit-Kriterium: Shadow schlägt HF in ≥2 Metrics            │
│  → Phase 3 freigeben                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Technische Spezifikation

### 1. Epistemic Regularization

**Formel:**

```
L_total = L_DPO + λ · max(0, H_pred - H_target)
              └─────────────────────────────┘
              Sicherheit in der Unsicherheit
```

**Komponenten:**

| Symbol | Beschreibung | Wertebereich |
|--------|--------------|--------------|
| `L_total` | Gesamte Loss-Funktion | - |
| `L_DPO` | Standard DPO Loss | - |
| `λ` | Regularization Strength | 0.1 - 0.5 (tunable) |
| `H_pred` | Vorhergesagte Entropie des Modells | 0 - 1 |
| `H_target` | Ziel-Entropie für epistemische Fragen | 0.1 - 0.3 |

**Intuition:**

Das Modell lernt, bei „Ich weiß es nicht"-Fragen **minimal Entropie** zu haben. Es soll sich **sicher sein, dass es unsicher ist**.

**Implementierung (Pseudocode):**

```python
def epistemic_regularization(logits, epistemic_mask, H_target=0.2):
    """
    logits: [batch_size, vocab_size]
    epistemic_mask: [batch_size] - True für epistemische Fragen
    H_target: Ziel-Entropie für Unsicherheit
    """
    # Entropie berechnen
    probs = F.softmax(logits, dim=-1)
    H = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [batch_size]
    
    # Nur für epistemische Fragen
    H_epistemic = H[epistemic_mask]
    
    # Penalty wenn H_pred > H_target
    penalty = torch.clamp(H_epistemic - H_target, min=0)
    
    return penalty.mean()
```

---

### 2. Shadow Loop Architektur

**Minimal-Skript (~200 Zeilen):**

```
shadow_loop/
├── __init__.py
├── config.py          # Hyperparameter (λ, H_target, etc.)
├── dataloader.py      # Custom DataLoader mit epistemic classification
├── model.py           # Model wrapper (LoRA + epistemic head)
├── loss.py            # L_DPO + epistemic regularization
├── train_step.py      # Single train_step mit torch.autograd
├── auditor.py         # Mini-Auditor (Heuristik + Reward)
└── main.py            # Haupt-Training-Loop
```

**Haupt-Loop (main.py):**

```python
for batch in dataloader:
    # 1. Epistemic Classification (Awareness)
    epistemic_categories = classifier(batch)
    
    # 2. Forward Pass
    logits = model(batch['input_ids'])
    
    # 3. Loss Calculation (Assessment)
    loss_dpo = compute_dpo_loss(logits, batch['preferences'])
    loss_epi = epistemic_regularization(logits, epistemic_categories)
    loss_total = loss_dpo + lambda_epi * loss_epi
    
    # 4. Backward Pass (Adjustment)
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 5. In-Loop Auditing (optional)
    if step % audit_interval == 0:
        samples = generate_samples(model, batch)
        audit_score = mini_auditor(samples, batch)
        log_metrics(audit_score)
```

---

### 3. Triple-A Prinzip

Phase 2.5 implementiert das **Triple-A Prinzip** in seiner einfachsten Form:

| Prinzip | Beschreibung | Implementierung in Phase 2.5 |
|---------|--------------|------------------------------|
| **Awareness** | Batch-Klassifikation | DataLoader klassifiziert epistemic categories via Heuristik |
| **Assessment** | Gradient Weighting | `g_scaled = g · w_cat` mit `w_cat = f(mastery_score)` |
| **Adjustment** | Adaptive Regularization | λ wird pro Kategorie adaptiv angepasst |

---

## Exit-Kriterien

### Primäre Exit-Kriterien (≥2 müssen erfüllt sein)

| Kriterium | Schwellenwert | Messung |
|-----------|---------------|---------|
| **Loss Improvement** | Shadow val_loss < HF val_loss - 0.01 | Über 3 consecutive runs |
| **Epistemic Score** | Shadow > HF + 5% | Epistemic Mode Accuracy |
| **Hallucination Rate** | Shadow < HF - 10% | Hallucination Eval |
| **Calibration (ECE)** | Shadow ECE < HF ECE - 0.01 | Expected Calibration Error |

### Sekundäre Exit-Kriterien (Monitoring)

- Win-Rate Improvement (> +3%)
- Fact Retention (> 95% der alten Fakten)
- Training Stability (kein Gradient Explosion)

---

## Implementierungs-Roadmap

### Woche 1:基础-Implementierung

**Tag 1-2: DataLoader + Epistemic Classification**
- [ ] `shadow_loop/dataloader.py` implementieren
- [ ] Epistemic Category Heuristik (Rule-based)
- [ ] Testing mit kleinen Batches

**Tag 3-4: Loss Function + Regularization**
- [ ] `shadow_loop/loss.py` implementieren
- [ ] DPO Loss + Epistemic Regularization
- [ ] λ-Tuning (0.1, 0.2, 0.3, 0.5 testen)

**Tag 5: Train Step + Optimizer**
- [ ] `shadow_loop/train_step.py` implementieren
- [ ] torch.autograd Setup
- [ ] Optimizer (AdamW) konfigurieren

### Woche 2: Testing + Evaluation

**Tag 6-7: Parallel-Running**
- [ ] Shadow Loop parallel zu HF Training starten
- [ ] Logging Setup (WandB)
- [ ] Erste Metrics vergleichen

**Tag 8-9: Debugging + Tuning**
- [ ] Gradient Flow analysieren
- [ ] λ nachjustieren
- [ ] H_target optimieren

**Tag 10: Decision**
- [ ] Exit-Kriterien evaluieren
- [ ] Entscheidung: Phase 3 freigeben oder nicht

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| **Shadow Loop instabil** | Mittel | Hoch | λ konservativ starten (0.1), Gradient Clipping |
| **Kein Improvement** | Hoch | Mittel | Shadow als reines Diagnostic-Tool verwenden |
| **Overfitting auf Shadow** | Niedrig | Mittel | Early Stopping, Validation Holdout |
| **Ressourcen-Overhead** | Mittel | Niedrig | Shadow nur auf 10-20% der Daten |

---

## Erfolgsmetriken

### Phase 2.5 gilt als erfolgreich, wenn:

✅ Shadow Loop schlägt HF Loop in **≥2 primären Metriken** über 3 consecutive runs  
✅ Keine Regression in Pass@1 (< 1%)  
✅ Training stabil (kein Gradient Explosion/Vanishing)  
✅ Epistemic Regularization Term konvergiert (H_pred → H_target)

### Bei Erfolg:

→ **Phase 3 freigeben** (voller Custom-Loop)

### Bei Misserfolg:

→ Shadow Loop als Diagnostic-Tool behalten  
→ HF Training fortsetzen  
→ Decision Gates erneut evaluieren

---

## Code-Struktur

```
src/diogenes/shadow_loop/
├── __init__.py
├── config.py
├── dataloader.py
├── loss.py
├── model.py
├── train_step.py
├── auditor.py
├── main.py
└── README.md
```

---

## Hyperparameter (Default)

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `lambda_epi` | 0.2 | Epistemic Regularization Strength |
| `H_target` | 0.2 | Ziel-Entropie für epistemische Fragen |
| `batch_size` | 16 | Batch-Größe (RTX 3050 optimiert) |
| `learning_rate` | 2e-5 | Learning Rate |
| `gradient_clip` | 1.0 | Gradient Clipping |
| `audit_interval` | 100 | Audit alle N Steps |

---

## Logging & Monitoring

### WandB Metrics

```python
wandb.log({
    "loss/total": loss_total.item(),
    "loss/dpo": loss_dpo.item(),
    "loss/epistemic": loss_epi.item(),
    "metrics/epistemic_score": epistemic_score,
    "metrics/hallucination_rate": hallucination_rate,
    "metrics/ece": ece,
    "regularization/H_pred": H_pred.mean().item(),
    "regularization/H_target": H_target,
    "regularization/lambda": lambda_epi,
})
```

---

## Nächste Schritte

1. **Nach Phase 2 (SFT Testing):**
   - Shadow Loop implementieren
   - Parallel-Experiment starten
   - Exit-Kriterien evaluieren

2. **Bei erfolgreichem Exit:**
   - Phase 3 freigeben
   - Custom-Loop zum Haupt-Training machen

3. **Bei nicht erfolgreichem Exit:**
   - Shadow Loop als Diagnostic-Tool behalten
   - HF Training fortsetzen
   - Decision Gates anpassen

---

## Referenzen

- **Decision Gates:** Siehe `roadmap.md#18-decision-gates`
- **Triple-A Prinzip:** Siehe `roadmap.md#20-technischer-blueprint`
- **Epistemic Modes:** Siehe `README.md#epistemic-modes`
- **Pass@1 Protection:** Siehe `docs/PASS1_GUARDRAILS.md`

# Training Strategy v2.0 – Velocity vs. Sovereignty

**Version:** 1.0  
**Stand:** 19. März 2026  
**Status:** ✅ IMPLEMENTIERT (Phase 2.5 + 3 neu definiert)

---

## Executive Summary

Diese Strategie definiert den **intelligenten Übergang** von Hugging Face-basiertem Training (Velocity) zu einem Custom-Loop (Sovereignty) mit **datenbasierten Decision Gates** statt gefühlsbasierten Entscheidungen.

### Kerninnovationen v2.0

✅ **Decision Gates**: Harte, messbare Checkpoints für Phasen-Übergänge  
✅ **Shadow Loop**: Risikofreies Testen des Custom-Loops parallel zum HF-Training  
✅ **Epistemic Regularization**: Custom Loss für „Sicherheit in der Unsicherheit"  
✅ **Curriculum Acceleration**: Bis zu 40% schnellere Iteration durch Mastery-based Sampling  
✅ **In-Loop Auditing**: Live-Filterung + instant Reward-Update (<50ms)  
✅ **Triple-A Prinzip**: Awareness → Assessment → Adjustment  

---

## Das Kern-Dilemma

### Velocity vs. Sovereignty Trade-off

```
┌────────────────────────────────────────────────────────────────┐
│                    AI Development Trade-off                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Feature Velocity          Architectural Sovereignty           │
│  ┌──────────────────┐      ┌──────────────────────────────┐   │
│  │ HF trl + peft +  │      │ Custom PyTorch Loop          │   │
│  │ DPOTrainer       │      │                              │   │
│  │                  │      │ • Full Gradient Control      │   │
│  │ ✓ Fast Iteration │      │ • Custom Loss Design         │   │
│  │ ✓ Minimal Code   │      │ • In-Loop Auditing           │   │
│  │                  │      │ • Dynamic Curriculum         │   │
│  │ ✗ Limited Control│      │                              │   │
│  │ ✗ Black-Box Loss │      │ ✗ Slower Iteration           │   │
│  │ ✗ Post-Hoc Audit │      │ ✗ Higher Debugging Effort    │   │
│  └──────────────────┘      └──────────────────────────────┘   │
│                                                                 │
│  → Phase 1&2: 100% Velocity (Daten validieren)                │
│  → Phase 3+: 100% Sovereignty (Alignment-Prozess optimieren)  │
└────────────────────────────────────────────────────────────────┘
```

---

## Decision Gates Matrix

### Wann wechseln? Harte, messbare Kriterien

| Symptom | Ursache | KPI / Schwellenwert | Lösung durch Custom-Loop |
|---------|---------|---------------------|--------------------------|
| **Loss-Stagnation** | Standard-CE gewichtet epistemic Uncertainty zu schwach | Plateau über >5% des Trainings (val_loss Δ < 0.001) | **Custom Loss Scaling**: Dynamische Strafe für „confident hallucinations" |
| **Mode-Collapse** | DPO verdrängt seltene epistemische Modi | < 15% der generierten Samples zeigen „I don't know"-Verhalten trotz Ground-Truth | **Dynamic Batching + Mode Balancing**: Adaptive Mischung pro Batch (rare-mode oversampling) |
| **Audit-Lag** | Audit-Erkenntnisse fließen zu langsam zurück | > 2 Stunden Delay zwischen Audit und nächstem Training-Step | **Online Rejection Sampling**: Live-Filterung + instant Reward-Update im selben Loop |
| **Gradient Interference** | SFT- und DPO-Ziele kollidieren (Catastrophic Forgetting) | > 20% Performance-Drop auf alten Fact-Tasks nach DPO | **Multi-Objective Gradient Surgery**: Getrennte Update-Pfade + PCGrad / GradNorm |

### Zusätzlicher Gate (v2.0): Epistemic Drift

**Wenn das Modell bei stabilen Fakten plötzlich >8% „Unsicherheit" produziert → sofortiger Wechsel zu Phase 3.**

---

## Phasen-Architektur mit Exit-Kriterien

### Phase 2: Baseline & Stress-Test (2-3 Wochen)

**Framework:** 100% `trl` + `peft` + `DPOTrainer`

**Ziel:** Metric-Baseline erstellen
- Win-Rate
- Epistemic-Score
- Hallucination-Rate
- Fact-Retention

**Exit-Kriterium:** Wenn **≥2 Decision-Gates triggern** → sofort weiter zu Phase 2.5

**Warum bleiben?** Ein früher Custom-Loop würde Debugging unmöglich machen (Daten- vs. Code-Fehler).

---

### Phase 2.5: Shadow Loop (1-2 Wochen) ⭐ NEU

**Framework:** Custom-Loop läuft **neben** HF-Training (kein Ersatz, sondern Schatten)

**Technik:** Minimales PyTorch-Skript (~200 Zeilen), das einen Aspekt isoliert

**Key Innovation: Epistemic Regularization Term**

```
L_total = L_DPO + λ · max(0, H_pred - H_target)
              └─────────────────────────────┘
              Sicherheit in der Unsicherheit
```

**Ziel:** Modell lernt, bei „Ich weiß es nicht"-Fragen **minimal Entropie** zu haben

**Exit-Kriterium:** Shadow-Loop schlägt HF-Loop in **≥2 Metrics** → Phase 3 freigeben

**Dokumentation:** Siehe `phasen/phase_2.5.md`

---

### Phase 3: Diogenes Alignment Engine (volle Sovereignty) ⭐ NEU

**Paradigmenwechsel:** Vom „Fine-Tuning" zum **Conditioned Alignment**

**Komponenten:**

1. **In-Loop Auditing**
   - Modell generiert 8 Samples während Training
   - Mini-Auditor bewertet in <50 ms
   - Loss wird sofort angepasst

2. **Curriculum Acceleration**
   - Loop trackt Mastery-Score pro Epistemic Mode
   - Blendet beherrschte Modi automatisch aus
   - **Bis zu 40% Rechenzeit-Einsparung**

3. **Technische Umsetzung**
   - Ein einziger `train_step`-Loop mit `torch.autograd`
   - Custom DataLoader (kein HF-Trainer mehr)
   - Vollständige Kontrolle über Gradienten, Loss, Sampling

**Dokumentation:** Siehe `phasen/phase_3.md`

---

## Triple-A Prinzip (Diogenes-spezifisch)

### Awareness → Assessment → Adjustment

```
┌────────────────────────────────────────────────────────────────┐
│                    Triple-A Prinzip                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. AWARENESS                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Batch wird beim Laden automatisch klassifiziert          │  │
│  │ (epistemic category via fast Heuristik oder Classifier)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  2. ASSESSMENT                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Gradient wird pro Kategorie gewichtet:                   │  │
│  │ g_scaled = g · w_cat  mit  w_cat = f(mastery_score)      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  3. ADJUSTMENT                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Adaptive LR + Weight-Decay pro Kategorie                 │  │
│  │ Bei Fact-Modus-Drift: Weight-Decay automatisch hoch      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Implementierung

**Awareness: Epistemic Classification**
```python
epistemic_categories = classifier.classify(batch['question'])
# Kategorien: DIRECT_ANSWER, CAUTIOUS_LIMIT, ABSTAIN, 
#             CLARIFY, REJECT_PREMISE, REQUEST_TOOL, PROBABILISTIC
```

**Assessment: Gradient Weighting**
```python
g_scaled = g * (1.0 - mastery_score[category])
# Schwache Modi werden stärker gewichtet
```

**Adjustment: Adaptive Regularization**
```python
if drift_detected(category):
    weight_decay[category] *= 2.0  # Drift-Schutz
    learning_rate[category] *= 0.5  # LR reduzieren
```

---

## Vorteile der v2.0-Strategie

### Gegenüber v1.0 (gefühlsbasiert)

| Aspekt | v1.0 (Alt) | v2.0 (Neu) | Verbesserung |
|--------|-----------|------------|--------------|
| **Entscheidungsgrundlage** | Bauchgefühl | Harte KPIs | ✅ Datenbasiert |
| **Risiko** | Hoch (Big-Bang-Switch) | Niedrig (Shadow-Loop) | ✅ Sicherheitsnetz |
| **Iteration Speed** | Konstant langsam | 40% schneller (Curriculum) | ✅ Effizienz |
| **Souveränität** | Zu früh oder zu spät | Zum optimalen Zeitpunkt | ✅ Timing |
| **Debugging** | Schwierig | Einfach (parallel) | ✅ Wartbarkeit |

### Rennwagen-Metapher (verfeinert)

```
Phase 2: Serienmotor
┌─────────────────────────────────────────────────────────┐
│ Fahre mit dem Serienmotor (Hugging Face)               │
│ → Lerne die Strecke (Daten)                            │
│ → Erkenne Schwachstellen (Decision Gates)              │
└─────────────────────────────────────────────────────────┘

Phase 2.5: Rennmotor in der Garage
┌─────────────────────────────────────────────────────────┐
│ Baue den Rennmotor in der Garage (Shadow-Loop)         │
│ → Teste ohne Risiko (parallel)                         │
│ → Vergleiche Performance (≥2 Metrics)                  │
└─────────────────────────────────────────────────────────┘

Phase 3: Motorentausch
┌─────────────────────────────────────────────────────────┐
│ Tausche den Motor nur dann aus, wenn du exakt weißt:   │
│ → An welcher Kurve (Failure-Mode)                      │
│ → Warum der Serienmotor versagt (Decision Gate)        │
│ → Dass der Rennmotor besser ist (Exit-Kriterium)       │
└─────────────────────────────────────────────────────────┘
```

---

## Implementierungs-Status

| Komponente | Status | Datei |
|------------|--------|-------|
| **Decision Gates Definition** | ✅ ABGESCHLOSSEN | `roadmap.md#18-decision-gates` |
| **Phase 2.5 Dokumentation** | ✅ ABGESCHLOSSEN | `phasen/phase_2.5.md` |
| **Phase 3 Dokumentation** | ✅ ABGESCHLOSSEN | `phasen/phase_3.md` |
| **Shadow Loop Implementierung** | ⏳ GEPLANT | `src/diogenes/shadow_loop/` |
| **Alignment Engine Implementierung** | ⏳ GEPLANT | `src/diogenes/alignment_engine/` |
| **Triple-A DataLoader** | ⏳ GEPLANT | `src/diogenes/dataloader.py` |
| **Epistemic Regularization** | ⏳ GEPLANT | `src/diogenes/loss.py` |
| **Mini-Auditor** | ⏳ GEPLANT | `src/diogenes/auditor.py` |
| **Mastery Tracker** | ⏳ GEPLANT | `src/diogenes/curriculum.py` |

---

## Nächste Schritte

### Sofort (nach Phase 2 SFT Testing)

1. **Decision Gates evaluieren**
   - Loss-Stagnation prüfen
   - Mode-Collapse analysieren
   - Audit-Lag messen
   - Gradient Interference testen

2. **Bei ≥2 Gates: Phase 2.5 starten**
   - Shadow Loop implementieren
   - Parallel-Experiment beginnen
   - Exit-Kriterien überwachen

3. **Bei erfolgreichem Exit: Phase 3 freigeben**
   - Alignment Engine implementieren
   - Custom-Loop zum Haupt-Training machen
   - Sovereignty übernehmen

### Langfristig (Phase 3+)

- Vollständige Kontrolle über Alignment-Prozess
- Custom Loss Functions für epistemische Optimierung
- In-Loop Auditing für sofortiges Feedback
- Curriculum Learning für 40% Effizienzsteigerung

---

## Erfolgsmetriken der Strategie

### Phase 2.5 Success Criteria

✅ Shadow-Loop schlägt HF-Loop in ≥2 primären Metriken  
✅ Keine Pass@1 Regression (< 1%)  
✅ Training stabil (kein Gradient Explosion)  
✅ Epistemic Regularization konvergiert (H_pred → H_target)

### Phase 3 Success Criteria

✅ Loss Improvement: val_loss < Phase 2 - 0.05  
✅ Epistemic Score: > +10% gegenüber Phase 2  
✅ Hallucination Rate: < 5%  
✅ Curriculum Efficiency: ≥ 30% Rechenzeit-Einsparung  
✅ In-Loop Audit Latenz: < 50ms  

### Gesamtstrategie Success Criteria

✅ **Velocity**: Schnelle Iteration in Phase 1-2 (Daten validieren)  
✅ **Sovereignty**: Volle Kontrolle ab Phase 3 (Alignment optimieren)  
✅ **Timing**: Optimaler Wechsel durch Decision Gates  
✅ **Risiko**: Minimiert durch Shadow-Loop  
✅ **Effizienz**: 40% schneller durch Curriculum Acceleration  

---

## Referenzen

- **Roadmap**: `roadmap.md#17-strategischer-rahmen-velocity-vs-sovereignty`
- **Phase 2.5**: `phasen/phase_2.5.md`
- **Phase 3**: `phasen/phase_3.md`
- **Decision Gates**: `roadmap.md#18-decision-gates`
- **Triple-A Prinzip**: `roadmap.md#20-technischer-blueprint`
- **Pass@1 Protection**: `docs/PASS1_GUARDRAILS.md`

---

## Zusammenfassung

Die **Training Strategy v2.0** ersetzt gefühlsbasierte durch **datenbasierte Entscheidungen**:

✅ **Decision Gates** definieren harte, messbare Kriterien für Phasen-Übergänge  
✅ **Shadow Loop** ermöglicht risikofreies Testen des Custom-Loops  
✅ **Triple-A Prinzip** (Awareness, Assessment, Adjustment) optimiert den Alignment-Prozess  
✅ **Curriculum Acceleration** spart bis zu 40% Rechenzeit  
✅ **In-Loop Auditing** bietet sofortiges Feedback (<50ms)  

**Ergebnis:** Ein Modell, das lieber ehrlich nicht antwortet als plausibel falsch zu sein – entwickelt mit maximaler Effizienz und minimalem Risiko.

# Phase 3 – DPO Training

**Dauer:** Tag 4

**Status:** ⏳ **GEPLANT** (wartet auf SFT-Checkpoint)

## Ziele

- [ ] Direct Preference Optimization durchführen
- [ ] Halluzinationen bestrafen
- [ ] Ehrliche Antworten belohnen
- [ ] 60k Preference Pairs trainieren

## Aufgaben

### 1. Training vorbereiten
- [ ] DPO Dataset laden (~60.000 Paare)
- [ ] Ranking-Klassen: Gold > Acceptable > Weak > Hallucination
- [ ] Data Preprocessing für Preference Learning
- [ ] **DPO-Audit durchführen** (neu)

### 2. DPO Training konfigurieren
- [ ] SFT-Checkpoint als Basis laden
- [ ] Preference Loss Funktion einstellen
- [ ] Beta-Parameter optimieren (typisch: 0.1–0.5)
- [ ] Batch Size & Learning Rate anpassen

### 3. Training durchführen
- [ ] Start DPO Training (`~6 Stunden` auf H100)
- [ ] Preference Accuracy monitoren
- [ ] Reward-Modell-Verlauf tracken
- [ ] Checkpoints speichern

### 4. Post-Training Validierung
- [ ] Halluzinationsrate auf Testset prüfen
- [ ] Preference Accuracy messen
- [ ] Qualitative Bewertung der Antworten

## Deliverables

- [ ] DPO-trained Model (Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Halluzinations-Baseline gemessen

## Erfolgskriterien

- [ ] Training abgeschlossen ohne Errors
- [ ] Preference Accuracy > Zufallsniveau
- [ ] Halluzinationen reduziert vs. SFT-only
- [ ] Ehrliche Ablehnungen stabil

## Metriken

| Metrik | Erwartet |
|--------|----------|
| DPO Loss | sinkend |
| Preference Accuracy | steigend |
| Halluzinationsrate | reduziert |

## DPO-Audit (Neu in v4)

**Vor dem Training durchführen:**

```python
from diogenes import check_dpo_for_prompt_interference

dpo_pairs = load_dpo_dataset("path/to/dpo_data.jsonl")
audit = check_dpo_for_prompt_interference(dpo_pairs)

if audit["concerns"]:
    print("⚠️  DPO data concerns:")
    for concern in audit["concerns"]:
        print(f"  - {concern}")
    
    if audit["difficulty_bias"] or audit["verbosity_bias"]:
        print("❌ Critical bias detected - review data before training")
```

**Grenzwerte:**

| Metrik | Schwellenwert | Aktion |
|--------|---------------|--------|
| Difficulty Bias | < 30% hard | ✓ Pass |
| Verbosity Bias | < 1.2 Ratio | ✓ Pass |
| Abstain Repr. | > 5% | ✓ Pass |

## Pass@1 Protection

**Während DPO-Training:**

DPO ist besonders anfällig für Pass@1-Degradation durch Prompt-Interferenz.

**Überwachung:**
```python
from diogenes import run_pass1_protection_test

result = run_pass1_protection_test(
    predictions=preds,
    ground_truth=gt,
    confidences=conf,
    baseline_pass_at_1=0.75,
    baseline_pass_at_k=0.90,
    k=5,
)

if result.is_regression:
    print(f"⚠️  {result.regression_severity}: {result.regression_details}")
    print(f"Recommendation: {result.recommendation}")
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

## Nächste Schritte

➡️ **Phase 4**: Calibration & Confidence Mapping

- Temperature Scaling implementieren
- ECE und Brier Score optimieren
- Confidence Mapping kalibrieren

## Referenzen

- `src/diogenes/train_dpo.py` – DPO Training Script
- `src/diogenes/pass1_protection.py` – DPO Audit Tools
- `docs/PASS1_GUARDRAILS.md` – DPO Design Guardrails

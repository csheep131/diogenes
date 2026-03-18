# Phase 4 – Calibration & Confidence Mapping

**Dauer:** Tag 5

## Ziele

- [ ] Temperature Scaling implementieren
- [ ] Confidence Mapping kalibrieren
- [ ] Brier Score optimieren
- [ ] Expected Calibration Error (ECE) minimieren

## Aufgaben

### 1. Calibration Layer implementieren
- Temperature Scaling Parameter einführen
- Optimiert auf:
  - Brier Score
  - Expected Calibration Error (ECE)
- Confidence-Berechnung basierend auf:
  - Token Entropy
  - Logit Gap
  - Mode Probability

### 2. Epistemic Routing Head finalisieren
- Linearer Classifier auf vorletztem Layer
- 7 Output-Klassen (Modi)
- Cross-Entropy Loss
- Confidence Scores kalibrieren

### 3. Calibration Training
- Calibration Dataset verwenden (Holdout)
- Temperature Parameter optimieren
- ECE minimieren (Ziel: –40 %)
- Brier Score validieren

### 4. Confidence Mapping testen
- Confidence vs. Accuracy plotten
- Reliability Diagram erstellen
- Über-/Unterkalibrierung erkennen
- Thresholds für Modi anpassen

## Deliverables

- [ ] Calibration Layer implementiert
- [ ] Temperature Scaling optimiert
- [ ] Confidence Mapping validiert
- [ ] ECE & Brier Score dokumentiert

## Erfolgskriterien

- ECE signifikant reduziert (Ziel: –40 %)
- Brier Score verbessert
- Confidence korreliert mit Accuracy
- Keine systematische Über-/Unterkalibrierung

## Metriken

- **ECE (Expected Calibration Error):** Ziel –40 %
- **Brier Score:** sollte sinken
- **Reliability Diagram:** nahe an diagonaler Linie

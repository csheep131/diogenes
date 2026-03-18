# Phase 6 – Red Teaming & Schwächen fixen

**Dauer:** Tag 7

## Ziele

- [ ] Adversarial Testing durchführen
- [ ] 2.000+ Red Team Samples generieren
- [ ] Schwachstellen identifizieren
- [ ] Kritische Fehler beheben

## Aufgaben

### 1. Red Team Dataset generieren
- **2.000+ adversarial Prompts** erstellen
- Durch zweites Modell generieren lassen
- Kategorien:
  - Falsche historische Annahmen
  - Manipulatives Framing
  - Zeitliche Fallen
  - Incentive Manipulation

### 2. Adversarial Testing
- Modell aktiv zur Halluzination zwingen
- Verhalten unter Druck messen
- Epistemische Stabilität prüfen
- failure modes dokumentieren

### 3. Schwachstellen analysieren
- Confusion Matrix auswerten
- Häufigste Fehler identifizieren
- Systematische Probleme erkennen
- Risk-Level-Analyse durchführen

### 4. Fixes implementieren
- Threshold-Tuning für Modi
- Over-Abstention vermeiden (Utility-Monitoring)
- Calibration nachjustieren
- Kritische Bugs fixen

### 5. Re-Testing
- Fixes validieren
- Regressionstests durchführen
- Verbesserte Metriken dokumentieren

## Deliverables

- [ ] Red Team Dataset (2.000+ Samples)
- [ ] Adversarial Testing Report
- [ ] Schwachstellen-Analyse
- [ ] Implementierte Fixes
- [ ] Validierte Verbesserungen

## Erfolgskriterien

- Modell widersteht adversarial attacks
- Halluzinationen unter Druck minimiert
- Kritische Schwachstellen behoben
- Utility Score stabil

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|------------|
| Over-Abstention | Utility-Monitoring + Threshold-Tuning |
| Schein-Kalibrierung | Echte ECE + Brier-Validierung |
| Datenleakage im Eval | Strikte Trennung der Datasets |
| Overfitting auf Benchmarks | Red-Team + WildBench-Fokus |

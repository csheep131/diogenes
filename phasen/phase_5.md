# Phase 5 – Full Evaluation & Confusion Matrix

**Dauer:** Tag 6

## Ziele

- [ ] Primäre Benchmarks auswerten
- [ ] Sekundäre Benchmarks auswerten
- [ ] Eigene Epistemic Suite testen
- [ ] Mode Confusion Matrix erstellen
- [ ] Utility Score berechnen

## Aufgaben

### 1. Primäre Benchmarks
- **TruthfulQA:** Ziel +8–15 %
- **HaluEval:** Ziel –20–30 % Halluzinationen
- **WildBench:** Real-World Performance

### 2. Sekundäre Benchmarks
- **GPQA:** Expertenwissen prüfen
- **LiveBench:** Aktuelle Fähigkeiten

### 3. Eigene Evaluation Suite
- **Epistemic Gap Eval:**
  - Ignorance Tests
  - Staleness Tests
  - False Premise Tests
  - Ambiguity Tests
  - Tool Required Tests
  - Adversarial Tests
  - Multi-Hop Tests

- **Mode Confusion Matrix:**
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
- Alle Metriken sammeln
- Mit Baseline vergleichen
- Schwachstellen identifizieren

## Deliverables

- [ ] Benchmark-Ergebnisse (TruthfulQA, HaluEval, WildBench)
- [ ] GPQA & LiveBench Scores
- [ ] Mode Confusion Matrix
- [ ] Utility Score Berechnung
- [ ] Vollständiger Evaluationsbericht

## Erfolgskriterien

- TruthfulQA: +8–15 % Verbesserung
- HaluEval: –20–30 % Halluzinationen
- ECE: –40 %
- Abstention AUROC: +15 %
- Utility Score: deutlich höher als Baseline

## Metriken

| Metrik | Ziel |
|--------|------|
| TruthfulQA | +8–15 % |
| HaluEval | –20–30 % |
| ECE | –40 % |
| Abstention AUROC | +15 % |
| Utility Score | > Baseline |

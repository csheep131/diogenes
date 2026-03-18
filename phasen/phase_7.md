# Phase 7 – Iterationen & Ablation Studies

**Dauer:** Woche 2+

## Ziele

- [ ] Ablation Studien durchführen
- [ ] Iterative Verbesserungen
- [ ] Finale Optimierung
- [ ] Release-Vorbereitung

## Aufgaben

### 1. Ablation Studien

| Run | Beschreibung |
|-----|--------------|
| A | Baseline (SFT + DPO) |
| B | + NEFTune Noise (alpha=5) |
| C | + MLP Noise (späte Layer) |
| D | + Epistemic Head (final) |

- Alle Runs dokumentieren
- Metriken vergleichen
- Beste Kombination identifizieren

### 2. Iterative Verbesserungen
- Schwachstellen aus Phase 6 adressieren
- Dataset erweitern (Lücken füllen)
- Hyperparameter nachjustieren
- Calibration verfeinern

### 3. Finale Optimierung
- Bestes Modell aus Ablation wählen
- Letzte Evaluation auf allen Benchmarks
- Utility Score maximieren
- ECE final kalibrieren

### 4. Release-Vorbereitung
- Model Cards erstellen
- Dokumentation finalisieren
- Inference-Pipeline bereitstellen
- Best Practices dokumentieren

### 5. Nächste Schritte planen
- Weitere Iterationen bei Bedarf
- Domain-spezifische Fine-Tunes
- Größere Eval-Studien
- Community Release vorbereiten

## Deliverables

- [ ] Ablation Study Report
- [ ] Finales optimiertes Modell
- [ ] Vollständige Dokumentation
- [ ] Inference-Pipeline
- [ ] Release Package

## Erfolgskriterien

- Bestes Modell identifiziert
- Alle Ziele erreicht oder übertroffen
- Dokumentation vollständig
- Modell einsatzbereit für kritische Anwendungen

## Erwartete Endergebnisse

| Metrik | Ziel |
|--------|------|
| TruthfulQA | +8–15 % |
| HaluEval | –20–30 % Halluzinationen |
| ECE | –40 % |
| Abstention AUROC | +15 % |
| Utility Score | deutlich höher |

---

## Abschluss

**Ein Modell, das lieber ehrlich nicht antwortet, als plausibel falsch zu sein.**

Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen (IT, Produktion, Medizin, Recht, Finanzen).

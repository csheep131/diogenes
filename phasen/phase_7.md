# Phase 7 – Iterationen & Ablation Studies

**Dauer:** Woche 2+

**Status:** ⏳ **GEPLANT**

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

- [ ] Alle Runs dokumentieren
- [ ] Metriken vergleichen
- [ ] Beste Kombination identifizieren

### 2. Iterative Verbesserungen
- [ ] Schwachstellen aus Phase 6 adressieren
- [ ] Dataset erweitern (Lücken füllen)
- [ ] Hyperparameter nachjustieren
- [ ] Calibration verfeinern

### 3. Finale Optimierung
- [ ] Bestes Modell aus Ablation wählen
- [ ] Letzte Evaluation auf allen Benchmarks
- [ ] Utility Score maximieren
- [ ] ECE final kalibrieren

### 4. Release-Vorbereitung
- [ ] Model Cards erstellen
- [ ] Dokumentation finalisieren
- [ ] Inference-Pipeline bereitstellen
- [ ] Best Practices dokumentieren

### 5. Nächste Schritte planen
- [ ] Weitere Iterationen bei Bedarf
- [ ] Domain-spezifische Fine-Tunes
- [ ] Größere Eval-Studien
- [ ] Community Release vorbereiten

## Deliverables

- [ ] Ablation Study Report
- [ ] Finales optimiertes Modell
- [ ] Vollständige Dokumentation
- [ ] Inference-Pipeline
- [ ] Release Package

## Erfolgskriterien

- [ ] Bestes Modell identifiziert
- [ ] Alle Ziele erreicht oder übertroffen
- [ ] Dokumentation vollständig
- [ ] Modell einsatzbereit für kritische Anwendungen

## Erwartete Endergebnisse

| Metrik | Ziel |
|--------|------|
| TruthfulQA | +8–15 % |
| HaluEval | –20–30 % Halluzinationen |
| ECE | –40 % |
| Abstention AUROC | +15 % |
| Utility Score | deutlich höher |
| **Pass@1** | **Stabil oder verbessert** |

## Ablation Study Design

```python
ablation_configs = {
    'A_baseline': {
        'sft': True,
        'dpo': True,
        'neftune': False,
        'mlp_noise': False,
        'epistemic_head': False,
    },
    'B_neftune': {
        'sft': True,
        'dpo': True,
        'neftune': True,
        'neftune_alpha': 5.0,
        'mlp_noise': False,
        'epistemic_head': False,
    },
    'C_mlp_noise': {
        'sft': True,
        'dpo': True,
        'neftune': False,
        'mlp_noise': True,
        'mlp_noise_layers': [28, 29, 30, 31],
        'epistemic_head': False,
    },
    'D_full': {
        'sft': True,
        'dpo': True,
        'neftune': True,
        'neftune_alpha': 5.0,
        'mlp_noise': True,
        'mlp_noise_layers': [28, 29, 30, 31],
        'epistemic_head': True,
    },
}
```

## Evaluation aller Ablation Runs

```python
from diogenes import compute_core_reliability_metrics

results = {}

for run_name, config in ablation_configs.items():
    print(f"\n=== Evaluating {run_name} ===")
    
    # Load model with this config
    model = load_model_with_config(config)
    
    # Run evaluation
    metrics = compute_core_reliability_metrics(
        predictions=preds,
        ground_truth=gt,
        confidences=conf,
        epistemic_modes=modes,
        gold_modes=gold_modes,
        is_knowable=knowable,
        needs_tool=needs_tool,
        tool_requests=tool_req,
        false_premise_flags=fp_flags,
        predicted_false_premise=pred_fp,
    )
    
    results[run_name] = {
        'pass_at_1': metrics.pass_at_1,
        'hallucination_rate': metrics.hallucination_rate,
        'ece': metrics.expected_calibration_error,
        'utility_score': utility_score,
    }
    
    print(f"Pass@1: {metrics.pass_at_1:.4f}")
    print(f"Hallucination Rate: {metrics.hallucination_rate:.4f}")
    print(f"ECE: {metrics.expected_calibration_error:.4f}")
    print(f"Utility Score: {utility_score:.4f}")

# Compare and select best
best_run = max(results, key=lambda x: results[x]['pass_at_1'])
print(f"\n✓ Best run: {best_run}")
```

## Release-Checkliste

### Modell-Release

- [ ] Model Card auf HuggingFace
- [ ] License-Datei (Apache 2.0)
- [ ] README mit Usage-Beispielen
- [ ] Citations-Datei (BibTeX)

### Code-Release

- [ ] PyPI Package (`pip install diogenes`)
- [ ] Docker Image für Inference
- [ ] Beispiel-Skripte
- [ ] API-Dokumentation

### Dokumentation

- [ ] Training Guide
- [ ] Inference Guide
- [ ] Best Practices
- [ ] FAQ & Troubleshooting

### Pass@1 Protection für Release

- [ ] Guardrails dokumentiert
- [ ] Regression-Testing erklärt
- [ ] DPO-Audit-Tools verfügbar
- [ ] Core Reliability Metrics integriert

## Abschluss

**Ein Modell, das lieber ehrlich nicht antwortet, als plausibel falsch zu sein.**

Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen (IT, Produktion, Medizin, Recht, Finanzen).

---

## Zusammenfassung aller Phasen

| Phase | Status | Deliverables |
|-------|--------|--------------|
| **Phase 0** | ✅ Abgeschlossen | Infrastruktur, Pipeline-Validierung |
| **Phase 1** | ✅ Abgeschlossen | Dataset-Generator, Training-Scripts, Pass@1-Schutz |
| **Phase 2** | ⏳ Bereit für Start | SFT Training (wartet auf H100) |
| **Phase 3** | ⏳ Geplant | DPO Training |
| **Phase 4** | ⏳ Geplant | Calibration & Confidence Mapping |
| **Phase 5** | ⏳ Geplant | Full Evaluation |
| **Phase 6** | ⏳ Geplant | Red Teaming |
| **Phase 7** | ⏳ Geplant | Ablation & Release |

## Nächste Schritte (sofort möglich)

1. **Remote-H100 vorbereiten**
   ```bash
   python scripts/prepare_remote_machine.py --config configs/remote_config.yaml
   ```

2. **SFT Training starten (Phase 2)**
   ```bash
   ssh <user>@<host> 'cd /opt/diogenes && ./train.sh'
   ```

3. **Pass@1 Protection integrieren**
   - Regression-Tracker in Checkpoint-Callback
   - DPO-Audit vor Phase 3

## Referenzen

- `README.md` – Projekt-Übersicht
- `roadmap.md` – Strategische Roadmap
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
- `docs/IMPLEMENTATION_SUMMARY.md` – Implementierungs-Übersicht

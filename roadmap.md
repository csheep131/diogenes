# MASTERPLAN – THE RELIABLE 32B

**Version 4**

**Stand:** 18. März 2026

**Status:** Phase 0-1 abgeschlossen · Phase 2 bereit für Start

---

## Projektübersicht

Ziel dieses Projekts ist die Entwicklung eines spezialisierten Sprachmodells auf Basis von **Qwen3-32B**, das auf **epistemische Zuverlässigkeit** optimiert ist.

Im Gegensatz zu klassischen LLM-Optimierungen (Accuracy, Benchmark-Scores) liegt der Fokus auf:

- Erkennen von Wissensgrenzen
- Minimierung von Halluzinationen
- Korrekte Unsicherheitsabschätzung
- Epistemisch korrektes Antwortverhalten
- Tool-Awareness

Das Modell soll in kritischen Bereichen (IT, Produktion, Medizin, Recht, Finanzen) eingesetzt werden können.

---

## 0. Versionshistorie

- **v4** (18.03.2026): Phase 0-1 abgeschlossen, Pass@1-Schutz implementiert, Eval-Metriken erweitert
- **v3**: Professionelle Struktur + komplette Roadmap + Risiken + Erwartete Verbesserungen
- **v2**: Gestraft & neutralisiert
- **v1**: Erster Entwurf

---

## 1. STRATEGISCHER ZIELKORRIDOR

### Core Objective

**Epistemic Reliability statt AGI-Hype**

Das Modell soll:
- wissen, wann es etwas weiß
- wissen, wann es etwas nicht weiß
- korrekt reagieren, wenn Wissen fehlt
- falsche Prämissen aktiv korrigieren

### Erfolgsdefinition

Ein Modell gilt als erfolgreich, wenn es:
- weniger halluziniert als vergleichbare 32B-Modelle
- besser kalibriert ist (ECE ↓)
- epistemisch korrekt reagiert (Mode Selection)
- Utility-Score nicht verliert
- **Pass@1 nicht zugunsten von Pass@k verschlechtert** (neu in v4)

---

## 2. EPISTEMISCHE MODI

Das Modell entscheidet für jede Anfrage einen von sieben Modi:

| Modus            | Beschreibung                          |
|------------------|---------------------------------------|
| DIRECT_ANSWER    | sichere, direkte Antwort              |
| CAUTIOUS_LIMIT   | Antwort mit klaren Einschränkungen    |
| ABSTAIN          | ehrliche Wissenslücke                 |
| CLARIFY          | Rückfrage bei Unklarheit              |
| REJECT_PREMISE   | falsche Annahme korrigieren           |
| REQUEST_TOOL     | externe Daten/Tool erforderlich       |
| PROBABILISTIC    | unsichere, aber plausible Ableitung   |

---

## 3. SYSTEMARCHITEKTUR

### Pipeline

```
User Input
    ↓
Epistemic Classifier (Routing Head)
    ↓
Mode Selection
    ↓
Routing Decision
    ↓
Antwortgenerierung / Tool Call / Rückfrage
```

### Komponenten

- Base Model: **Qwen3-32B**
- LoRA Adapter (rank 32)
- Epistemic Routing Head (7-Klassen-Classifier)
- Calibration Layer (Temperature Scaling)
- Evaluation Suite + Red Team Engine
- **Pass@1 Regression Tracker** (neu)

---

## 4. DATENARCHITEKTUR

### 4.1 Dataset Split

| Typ              | Zweck                          | Größe      |
|------------------|--------------------------------|------------|
| SFT Dataset      | Verhalten & Modi lernen        | ~80.000    |
| DPO Dataset      | Präferenz & Halluzinations-Reduktion | ~60.000 Paare |
| Eval Dataset     | unabhängige Bewertung          | 5.000+     |
| Red Team Dataset | adversarial Tests              | 2.000+     |

### 4.2 Fehlerklassen (8 Klassen → 7 Modi)

| Klasse           | Zielmodus          |
|------------------|--------------------|
| Ignorance        | ABSTAIN            |
| Staleness        | CAUTIOUS_LIMIT     |
| Ambiguity        | CLARIFY            |
| False Premise    | REJECT_PREMISE     |
| Adversarial      | DIRECT_ANSWER (stay factual) |
| Shallow Trap     | PROBABILISTIC      |
| Multi-Hop        | PROBABILISTIC      |
| Tool Required    | REQUEST_TOOL       |

### 4.3 Datenschema (JSON)

```json
{
  "id": "sample_001",
  "question": "...",
  "category": "false_premise",
  "gold_mode": "REJECT_PREMISE",
  "risk_level": "high",
  "needs_tool": false,
  "time_sensitive": false,
  "false_premise": true,
  "confidence_target": 0.1,
  "chosen_answer": "...",
  "rejected_answer": "...",
  "reasoning_trace": "optional"
}
```

---

## 5. REASONING FRAMEWORK

Epistemischer Entscheidungsprozess (immer intern):

1. Klassifikation der Aufgabe
2. Prüfung der Prämisse
3. Zeitliche Relevanz prüfen
4. Tool-Bedarf prüfen
5. Unsicherheit schätzen
6. Modus auswählen
7. Antwort generieren

---

## 6. TRAININGSPIPELINE

### Hardware

- 1× NVIDIA H100 80 GB (Remote)
- Alternative: A100 80 GB

### Framework

- Axolotl oder Unsloth
- QLoRA (4-bit) Quantisierung

### Modellparameter

- Base: Qwen3-32B
- LoRA Rank: 32
- LoRA Alpha: 64
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

---

## 7. TRAININGSPHASEN

### Phase 1 – SFT ✅ ABGESCHLOSSEN

**Status:** Script implementiert, bereit für Remote-Training

**Ziel:** Modusverhalten & Routing stabilisieren

- 80k Samples, 3 Epochen
- Dauer: ~4 Stunden auf H100
- Script: `src/diogenes/train_sft.py`

### Phase 2 – DPO ✅ IMPLEMENTIERT

**Status:** Script implementiert, wartet auf SFT-Checkpoint

**Ziel:** Halluzinationen bestrafen, ehrliche Antworten belohnen

- Ranking: Gold > Acceptable > Weak > Hallucination
- 60k Paare
- Dauer: ~6 Stunden
- Script: `src/diogenes/train_dpo.py`

---

## 8. ABLATION STUDIEN

| Run | Beschreibung |
|-----|--------------|
| A | Baseline (SFT + DPO) |
| B | + NEFTune Noise (alpha=5) |
| C | + MLP Noise (späte Layer) |
| D | + Epistemic Head (final) |

---

## 9. EPISTEMIC ROUTING HEAD

Linearer Classifier auf dem vorletzten Layer.

- Output-Klassen: 7 Modi
- Loss: Cross-Entropy

---

## 10. KALIBRIERUNG

**Temperature Scaling**

- Optimiert auf Brier Score & Expected Calibration Error (ECE)
- Confidence basierend auf: Token Entropy + Logit Gap + Mode Probability

---

## 11. INFERENCE POLICY

```
Input
    ↓
Epistemic Classifier
    ↓
Mode Entscheidung
    ├── TOOL          → Tool Request
    ├── CLARIFY       → Rückfrage
    ├── REJECT        → Prämisse erklären
    ├── ABSTAIN       → ehrliche Ablehnung
    └── else          → Antwort generieren
```

---

## 12. EVALUATION

### Tier 1: Core Reliability Metrics (Primär)

| Metrik | Ziel | Priorität |
|--------|------|-----------|
| **Pass@1** | Maximieren | **PRIMARY** |
| Hallucination Rate | < 5% | Hoch |
| ECE | < 0.05 (–40%) | Hoch |
| Brier Score | Minimieren | Mittel |
| Abstention AUROC | +15% | Mittel |
| Mode Accuracy | Maximieren | Mittel |
| False Premise Detection | Maximieren | Mittel |

### Tier 2: Special Metrics (Nur Monitoring)

| Metrik | Domain | Verwendung |
|--------|--------|------------|
| Pass@k | Math, Code | **Niemals optimieren** |
| Best-of-k | Tool-assisted | Spezialfälle |

> **⚠️ Pass@1 Protection:** Tier-2-Metriken dürfen NICHT für Checkpoint-Promotion oder globale Reward-Optimierung verwendet werden. Siehe `docs/PASS1_GUARDRAILS.md`.

### Traditionelle Benchmarks

- **TruthfulQA:** Ziel +8–15 %
- **HaluEval:** Ziel –20–30 % Halluzinationen
- **WildBench:** Real-World Performance
- **GPQA:** Expertenwissen
- **LiveBench:** Aktuelle Fähigkeiten

### Eigene Suite

- **Epistemic Gap Eval:** Ignorance, Staleness, False Premise, Ambiguity, Tool Required, Adversarial, Multi-Hop
- **Mode Confusion Matrix:** 7×7 Klassifikationsanalyse
- **Utility Score:** Gewichtete Bewertung

### Utility Score Formel

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

---

## 13. RED TEAMING

**Adversarial Prompts** (generiert durch zweites Modell):

- Falsche historische Annahmen
- Manipulatives Framing
- Zeitliche Fallen
- Incentive Manipulation

**Ziel:** Modell aktiv zur Halluzination zwingen und Verhalten messen.

---

## 14. KOMPLETTE ROADMAP

| Phase | Tage | Aufgabe | Status |
|-------|------|---------|--------|
| 0 | Tag 0 | Repo + Infrastruktur + Modelle laden | ✅ **ABGESCHLOSSEN** |
| 1 | Tag 1–2 | Dataset Generator + SFT + DPO erstellen | ✅ **ABGESCHLOSSEN** |
| 2 | Tag 3 | SFT Training | 🔄 **BEREIT FÜR START** |
| 3 | Tag 4 | DPO Training | ⏳ **GEPLANT** |
| 4 | Tag 5 | Calibration + Confidence Mapping | ⏳ **GEPLANT** |
| 5 | Tag 6 | Full Evaluation + Confusion Matrix | ⏳ **GEPLANT** |
| 6 | Tag 7 | Red Teaming + Schwächen fixen | ⏳ **GEPLANT** |
| 7 | Woche 2+ | Iterationen + Ablation-Vergleich | ⏳ **GEPLANT** |

---

## 15. RISIKEN & MITIGATION

| Risiko | Mitigation |
|--------|------------|
| Over-Abstention | Utility-Monitoring + Threshold-Tuning |
| Schein-Kalibrierung | Echte ECE + Brier-Validierung |
| Datenleakage im Eval | Strikte Trennung der Datasets |
| Overfitting auf Benchmarks | Red-Team + WildBench-Fokus |
| **Pass@1 Degradation** | **Regression-Tracker + Guardrails (neu)** |
| **DPO Prompt-Interferenz** | **DPO-Audit vor Training (neu)** |

---

## 16. ERWARTETE ERGEBNISSE (Ziele)

| Metrik | Ziel |
|--------|------|
| TruthfulQA | +8–15 % |
| HaluEval | –20–30 % Halluzinationen |
| ECE | –40 % |
| Abstention AUROC | +15 % |
| Utility Score | deutlich höher durch korrekte Modi |
| **Pass@1** | **Stabil oder verbessert (neu)** |

---

## 17. ENDZIEL

**Ein Modell, das lieber ehrlich nicht antwortet als plausibel falsch zu sein.**

Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen (IT, Produktion, Medizin, Recht, Finanzen).

---

## 18. NÄCHSTE SCHRITTE (sofort möglich)

### ✅ Bereits implementiert:

- `dataset_generator.py` – SFT/DPO Datengenerierung
- `train_sft.py` – SFT Training mit LoRA/QLoRA
- `train_dpo.py` – DPO Training mit Hallucination Penalty
- `eval_metrics.py` – Core Reliability Metrics + Pass@1 Schutz
- `pass1_protection.py` – Regression Detection + DPO Audit
- `docs/PASS1_GUARDRAILS.md` – Produkt-Richtlinien

### ➡️ Jetzt ausführen:

1. **Remote-Maschine vorbereiten**
   ```bash
   python scripts/prepare_remote_machine.py --config configs/remote_config.yaml
   ```

2. **SFT Training starten (Phase 2)**
   ```bash
   ssh <user>@<host> 'cd /opt/diogenes && ./train.sh'
   ```

3. **DPO Training vorbereiten (Phase 3)**
   - DPO-Dataset generieren
   - DPO-Audit durchführen
   - Training scripten

4. **Evaluation pipeline integrieren**
   - Pass@1 Regression Tracker einbinden
   - Core Metrics in Checkpoint-Callback

---

## 19. PASS@1 SCHUTZ (Neu in v4)

### Kernprinzip

**Pass@1 ist die einzige Metrik für Checkpoint-Promotion.**

Pass@k (k>1) darf ausschließlich für Mathematik/Code zu Monitoring-Zwecken verwendet werden.

### Implementierung

1. **Zwei-Tier-Evaluation**
   - Tier 1: Core Reliability (Pass@1, ECE, Hallucination Rate)
   - Tier 2: Special Metrics (Pass@k, nur Math/Code)

2. **Regression Detection**
   - Automatische Erkennung: Pass@1 ↓ bei Pass@k ↑
   - Kritische Schwelle: Pass@1 < –2% bei Pass@k > +1%

3. **DPO Audit**
   - Prüfung auf Prompt-Interferenz
   - Difficulty-Bias < 30%
   - Verbosity-Bias < 1.2 Ratio

### Entscheidungsmatrix

| Bedingung | Pass@1 Δ | Pass@k Δ | Aktion |
|-----------|----------|----------|--------|
| **Kritische Regression** | < –2% | > +1% | ❌ NICHT PROMOTEN |
| **Warnung** | < –1% | > +0.5% | ⚠️ Vorsichtig prüfen |
| **Verbesserung** | > +1% | Beliebig | ✓ Sicher |
| **Stabil** | ±1% | Beliebig | ✓ Sicher |

Siehe `docs/PASS1_GUARDRAILS.md` für vollständige Richtlinien.

---

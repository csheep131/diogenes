# MASTERPLAN – THE RELIABLE 32B

**Version 7**

**Stand:** 19. März 2026

**Status:** Phase 0-1 abgeschlossen · Phase 2-6 auf RTX 3050 (8GB) · Phase 7 auf H100 · Phase 2.5 + 3 neu hinzugefügt

---

## Entwicklungs-Workflow

**Alle Tests und Entwicklung laufen lokal auf RTX 3050 (8GB VRAM).**

Erst das validierte Endprodukt wird auf der H100-Infrastruktur trainiert.

### Lokale Entwicklung (RTX 3050 8GB)

- Pipeline-Validierung mit kleineren Modellen (0.6B–3B)
- Hyperparameter-Tuning
- Evaluations-Suite testen
- Pass@1 Protection validieren
- Alle Scripts und Komponenten entwickeln

### Produktionstraining (H100 80GB)

- Finales Qwen3-32B Training nach lokaler Validierung
- Full-Scale SFT und DPO auf Zielmodell
- Letzte Kalibrierung und Evaluation

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

- **v7** (19.03.2026): Phase 2.5 (Shadow Loop) + Phase 3 (Alignment Engine) + Decision Gates + Training Strategy v2.0
- **v6** (19.03.2026): README Update + Current Status + Training Strategy
- **v5** (18.03.2026): Phase 0-1 abgeschlossen, Pass@1-Schutz implementiert, Eval-Metriken erweitert
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

### Entwicklungs-Strategie

**Phase 0-6: Lokale Entwicklung (RTX 3050 8GB)**

Alle Scripts und die komplette Pipeline werden mit kleineren Modellen entwickelt und validiert:

| Phase | Modell | VRAM | Zweck |
|-------|--------|------|-------|
| Phase 0-1 | Qwen3-0.6B | ~1.5 GB | Pipeline, Scripts |
| Phase 2-3 | Qwen2.5-3B | ~6 GB | SFT/DPO Testing |
| Phase 4-6 | Qwen2.5-3B | ~6 GB | Eval, Calibration |

**Phase 7: Produktion (H100 80GB)**

Erst nach vollständiger lokaler Validierung:

| Phase | Modell | VRAM | Dauer | Zweck |
|-------|--------|------|-------|-------|
| 7-A | Qwen3-32B | ~65 GB | ~4h | Final SFT |
| 7-B | Qwen3-32B | ~65 GB | ~6h | Final DPO |
| 7-C | Qwen3-32B | ~65 GB | ~2h | Calibration |
| 7-D | Qwen3-32B | ~65 GB | ~4h | Evaluation |

### Phase 1 – SFT ✅ ABGESCHLOSSEN (RTX 3050)

**Status:** Script implementiert, auf RTX 3050 getestet

**Ziel:** Modusverhalten & Routing stabilisieren

- 80k Samples, 3 Epochen
- Dauer: ~4 Stunden auf H100 (Phase 7), ~6-8h auf RTX 3050 (3B Modell)
- Script: `src/diogenes/train_sft.py`
- **Getestet mit:** Qwen3-0.6B und Qwen2.5-3B-Instruct

### Phase 2 – DPO ✅ IMPLEMENTIERT (RTX 3050)

**Status:** Script implementiert, Testing auf RTX 3050

**Ziel:** Halluzinationen bestrafen, ehrliche Antworten belohnen

- Ranking: Gold > Acceptable > Weak > Hallucination
- 60k Paare
- Dauer: ~6 Stunden auf H100 (Phase 7), ~8-10h auf RTX 3050 (3B Modell)
- Script: `src/diogenes/train_dpo.py`
- **Getestet mit:** Qwen3-0.6B und Qwen2.5-3B-Instruct

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

| Phase | Hardware | Tage | Aufgabe | Status |
|-------|----------|------|---------|--------|
| 0 | RTX 3050 | Tag 0 | Repo + Infrastruktur + Pipeline | ✅ **ABGESCHLOSSEN** |
| 1 | RTX 3050 | Tag 1–2 | Dataset Generator + Scripts | ✅ **ABGESCHLOSSEN** |
| 2 | RTX 3050 | Tag 3–5 | SFT Testing (3B Modell) | 🔄 **IN PROGRESS** |
| 3 | RTX 3050 | Tag 6–8 | DPO Testing (3B Modell) | ⏳ **GEPLANT** |
| **2.5** | **RTX 3050** | **Tag 8–10** | **Shadow Loop (Parallel-Experiment)** | ⏳ **GEPLANT** |
| **3.5** | **RTX 3050** | **Tag 10–11** | **SUA Specialization (Staleness/Unknown/Ambiguity)** | ⏳ **GEPLANT** |
| 4 | RTX 3050 | Tag 12 | Calibration Testing | ⏳ **GEPLANT** |
| 5 | RTX 3050 | Tag 13–14 | Full Evaluation (3B Modell) | ⏳ **GEPLANT** |
| 6 | RTX 3050 | Tag 15–16 | Red Teaming (3B Modell) | ⏳ **GEPLANT** |
| 7-A | H100 | Tag 17 | **Final SFT (32B)** | ⏳ **GEPLANT** |
| 7-B | H100 | Tag 18 | **Final DPO (32B)** | ⏳ **GEPLANT** |
| **7-B.1** | **H100** | **Tag 18** | **Final SUA Specialization (32B)** | ⏳ **GEPLANT** |
| 7-C | H100 | Tag 19 | **Final Calibration (32B)** | ⏳ **GEPLANT** |
| 7-D | H100 | Tag 20 | **Final Evaluation (32B)** | ⏳ **GEPLANT** |

**Hinweis zu Phase 2.5 (Shadow Loop):** 
- Der Custom-Loop läuft parallel zum HF-Training (kein Ersatz)
- Minimales PyTorch-Skript (~200 Zeilen) testet Epistemic Regularization
- Exit-Kriterium: Shadow-Loop schlägt HF-Loop in ≥2 Metrics → Phase 3 freigeben

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
| **SUA Overfitting** | **Nur 1-2 Epochen, Low LR (5e-6), Early Stopping** |
| **SUA-Mode Confusion** | **Mode Confusion Matrix nach Training analysieren** |

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
| **Staleness Detection** | **> 80% (Phase 3.5)** |
| **Unknown Detection AUROC** | **> 0.85 (Phase 3.5)** |
| **Ambiguity Resolution** | **> 75% (Phase 3.5)** |

---

## 17. STRATEGISCHER RAHMEN: VELOCITY VS. SOVEREIGNTY

### Das Kern-Dilemma

In der KI-Entwicklung existiert ein klassischer Trade-off:

**Feature Velocity**: Schnelles Iterieren mit Hugging Face `trl` + `peft` + `DPOTrainer`
- ✅ Hohe Experimentiergeschwindigkeit
- ✅ Minimaler Code-Overhead
- ❌ Begrenzte Kontrolle über Gradientenfluss, Loss-Design, Sampling-Logik

**Architectural Sovereignty**: Vollständige Kontrolle über Alignment-Prozess
- ✅ Präzise Steuerung von Loss, Gradienten, Sampling
- ✅ Custom Loss Functions möglich
- ✅ Online-Auditing im Training-Loop
- ❌ Langsamere Iteration, höherer Debugging-Aufwand

### Strategische Regel

| Phase | Priorität | Begründung |
|-------|-----------|------------|
| **Phase 1 & 2** | 100% Velocity | Wir müssen zuerst validieren, ob die Diogenes-Daten überhaupt „zünden" |
| **Phase 3+** | 100% Sovereignty | Wir optimieren nicht nur das Modell, sondern den gesamten Alignment-Prozess |

---

## 18. DECISION GATES – Der intelligente Abzweig

Statt eines starren Phasen-Switches definieren wir **harte, messbare Checkpoints**. Der Wechsel zum Custom-Loop erfolgt erst, wenn **mindestens zwei** der folgenden Bedingungen über **drei aufeinanderfolgende Training-Runs** erfüllt sind.

### Decision Gate Matrix

| Symptom | Ursache | KPI / Schwellenwert | Lösung durch Custom-Loop |
|---------|---------|---------------------|--------------------------|
| **Loss-Stagnation** | Standard-Cross-Entropy gewichtet epistemic Uncertainty zu schwach | Plateau über >5% des Trainings (val_loss Δ < 0.001) | **Custom Loss Scaling**: Dynamische Strafe für „confident hallucinations" (epistemic penalty term) |
| **Mode-Collapse** | DPO verdrängt seltene epistemische Modi | < 15% der generierten Samples zeigen „I don't know"-Verhalten trotz Ground-Truth | **Dynamic Batching + Epistemic Mode Balancing**: Adaptive Mischung pro Batch (rare-mode oversampling) |
| **Audit-Lag** | Audit-Erkenntnisse fließen zu langsam zurück | > 2 Stunden Delay zwischen Audit und nächstem Training-Step | **Online Rejection Sampling**: Live-Filterung + instant Reward-Update im selben Loop |
| **Gradient Interference** | SFT- und DPO-Ziele kollidieren (Catastrophic Forgetting) | > 20% Performance-Drop auf alten Fact-Tasks nach DPO | **Multi-Objective Gradient Surgery**: Getrennte Update-Pfade + PCGrad / GradNorm |

### Zusätzlicher Gate (v2.0): Epistemic Drift

**Wenn das Modell bei stabilen Fakten plötzlich >8% „Unsicherheit" produziert → sofortiger Wechsel zu Phase 3.**

---

## 19. VERFEINERTE PHASEN-ARCHITEKTUR (mit klaren Exit-Kriterien)

### Phase 2: Baseline & Stress-Test (Status Quo – 2–3 Wochen)

**Framework:** 100% `trl` + `peft` + `DPOTrainer`

**Ziel:** Erstellen einer Metric-Baseline
- Win-Rate
- Epistemic-Score
- Hallucination-Rate
- Fact-Retention

**Exit-Kriterium:** Wenn **≥2 Decision-Gates triggern** → sofort weiter zu Phase 2.5

**Warum bleiben?** Ein früher Custom-Loop würde Debugging unmöglich machen (Daten- vs. Code-Fehler).

---

### Phase 2.5: Shadow Loop (Parallel-Experiment – 1–2 Wochen) ⭐ NEU

**Framework:** Der Custom-Loop läuft **neben** dem HF-Training (kein Ersatz, sondern Schatten).

**Technik:** Minimales PyTorch-Skript (nur ~200 Zeilen), das einen Aspekt isoliert (z.B. nur Sampling + Epistemic Regularization).

**Key Innovation: Epistemic Regularization Term**

```
L_total = L_DPO + λ · max(0, H_pred - H_target)
              └─────────────────────────────┘
              Sicherheit in der Unsicherheit
```

**Ziel:** Das Modell lernt, bei „Ich weiß es nicht"-Fragen **minimal Entropie** zu haben (sich sicher zu sein, dass es unsicher ist).

**Exit-Kriterium:** Shadow-Loop schlägt HF-Loop in **≥2 Metrics** → Phase 3 freigeben.

---

### Phase 3: Diogenes Alignment Engine (ab hier volle Sovereignty) ⭐ NEU

**Paradigmenwechsel:** Vom „Fine-Tuning" zum **Conditioned Alignment**.

**Komponenten:**

1. **In-Loop Auditing**
   - Modell generiert während Training 8 Samples
   - Mini-Auditor (Heuristik + kleines Reward-Model) bewertet in <50 ms
   - Loss wird sofort angepasst

2. **Curriculum Acceleration** (neu & mächtig)
   - Der Loop trackt pro Epistemic Mode (False Premise, Ambiguity, etc.) die Mastery-Score
   - Blendet bereits beherrschte Modi automatisch aus
   - **Bis zu 40% Rechenzeit-Einsparung**

3. **Technische Umsetzung**
   - Ein einziger `train_step`-Loop mit `torch.autograd`
   - Custom DataLoader (kein HF-Trainer mehr)
   - Vollständige Kontrolle über Gradienten, Loss, Sampling

---

## 20. TECHNISCHER BLUEPRINT – Das „Triple-A" Prinzip (Diogenes-spezifisch)

| Prinzip | Beschreibung | Implementierung |
|---------|--------------|-----------------|
| **Awareness** | Batch wird beim Laden automatisch klassifiziert (epistemic category via fast Heuristik oder kleines Classifier-Head) | Epistemic Category Classification beim DataLoader |
| **Assessment** | Gradient wird pro Kategorie gewichtet: `g_scaled = g · w_cat` mit `w_cat = f(current mastery score)` | Category-weighted Gradient Scaling |
| **Adjustment** | Adaptive LR + Weight-Decay pro Kategorie: Wenn Fact-Modus driftet, wird Weight-Decay automatisch hochgefahren („Drift-Schutz") | Adaptive Learning Rate + Weight Decay per Category |

### Zusätzlich (v2.0):

- **Checkpoint-Resumption** mit State-Dict für alle drei A's (sodass Experimente pausierbar und reproduzierbar sind)
- **WandB + custom logging** aller epistemic Metrics in Echtzeit

---

## 21. ZUSAMMENFASSUNG & KLARE EMPFEHLUNG

### Die Rennwagen-Metapher (verfeinert)

Behandle den Custom-Loop wie einen **Motorentausch während des Rennens**:

| Phase | Metapher | Beschreibung |
|-------|----------|--------------|
| **Phase 2** | Serienmotor | Fahre mit dem Serienmotor (Hugging Face) – lerne die Strecke (Daten) und die Schwachstellen kennen |
| **Phase 2.5** | Rennmotor in der Garage | Baue den Rennmotor in der Garage (Shadow-Loop) – teste ohne Risiko |
| **Phase 3** | Motorentausch | Tausche den Motor nur dann aus, wenn du exakt weißt, an welcher Kurve (welchem Failure-Mode) der Serienmotor versagt |

### Vorteil der neuen v2.0-Version

✅ Entscheidungen sind jetzt **datenbasiert** statt gefühlsbasiert  
✅ Risiko minimiert (Shadow-Loop als Sicherheitsnetz)  
✅ Bis zu **40% schnellere Iteration** durch Curriculum Acceleration  
✅ Vollständige **Sovereignty** ab dem Moment, wo sie wirklich zählt

---

## 22. ENDZIEL

**Ein Modell, das lieber ehrlich nicht antwortet als plausibel falsch zu sein.**

Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen (IT, Produktion, Medizin, Recht, Finanzen).

---

## 18. NÄCHSTE SCHRITTE

### ✅ Bereits implementiert:

- `dataset_generator.py` – SFT/DPO Datengenerierung
- `train_sft.py` – SFT Training mit LoRA/QLoRA
- `train_dpo.py` – DPO Training mit Hallucination Penalty
- `eval_metrics.py` – Core Reliability Metrics + Pass@1 Schutz
- `pass1_protection.py` – Regression Detection + DPO Audit
- `docs/PASS1_GUARDRAILS.md` – Produkt-Richtlinien

### ➡️ Jetzt ausführen (RTX 3050 8GB):

**Phase 2: SFT Testing mit Qwen2.5-3B-Instruct**

1. **Dataset vorbereiten:**
   ```bash
   python src/diogenes/dataset_generator.py --split sft --size 80000
   ```

2. **SFT Training lokal starten:**
   ```bash
   python src/diogenes/train_sft.py \
     --model_name Qwen/Qwen2.5-3B-Instruct \
     --config configs/config.yaml \
     --output_dir models/sft_3b_test
   ```

3. **Ergebnisse validieren:**
   ```bash
   python src/diogenes/eval_metrics.py \
     --model_path models/sft_3b_test \
     --benchmark truthfulqa
   ```

**Phase 3: DPO Testing**

1. **DPO-Dataset generieren:**
   ```bash
   python src/diogenes/dataset_generator.py --split dpo --size 60000
   ```

2. **DPO-Audit durchführen:**
   ```python
   from diogenes import check_dpo_for_prompt_interference
   dpo_pairs = load_dpo_dataset("datasets/dpo_60k.jsonl")
   audit = check_dpo_for_prompt_interference(dpo_pairs)
   ```

3. **DPO Training lokal:**
   ```bash
   python src/diogenes/train_dpo.py \
     --model_name Qwen/Qwen2.5-3B-Instruct \
     --sft_checkpoint models/sft_3b_test \
     --output_dir models/dpo_3b_test
   ```

**Phase 3.5: SUA Specialization (Staleness/Unknown/Ambiguity)**

1. **SUA-Dataset generieren:**
   ```bash
   python src/diogenes/dataset_generator.py --split sua \
     --staleness 8000 \
     --unknown 10000 \
     --ambiguity 7000
   ```

2. **SUA Training lokal:**
   ```bash
   ./scripts/run_sua_training.sh \
     --dpo_checkpoint models/dpo_3b_test/final_checkpoint
   ```

3. **SUA-Metriken evaluieren:**
   ```bash
   python src/diogenes/eval_metrics.py \
     --model_path models/sua_3b_test \
     --sua
   ```

4. **Pass@1 Protection Check:**
   ```bash
   python3 scripts/pass1_protection_check.py \
     --model-path models/sua_3b_test \
     --baseline-pass-at-1 0.75
   ```

### ➡️ Nach lokaler Validierung (H100 80GB):

**Phase 7: Produktionstraining mit Qwen3-32B**

1. **Remote-Maschine vorbereiten:**
   ```bash
   python scripts/prepare_remote_machine.py --config configs/remote_config.yaml
   ```

2. **SFT Training starten:**
   ```bash
   ssh <user>@<host> 'cd /opt/diogenes && ./train_sft_final.sh'
   ```

3. **DPO Training starten:**
   ```bash
   ssh <user>@<host> 'cd /opt/diogenes && ./train_dpo_final.sh'
   ```

4. **SUA Specialization starten (Phase 7-B.1):**
   ```bash
   ssh <user>@<host> 'cd /opt/diogenes && ./train_sua_final.sh'
   ```

---

## 19. PASS@1 SCHUTZ (Neu in v5)

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

## 20. HARDWARE-REQUIREMENTS

### Lokale Entwicklung (RTX 3050 8GB)

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| **GPU** | NVIDIA 4GB VRAM | NVIDIA 8GB VRAM (RTX 3050/3060) |
| **RAM** | 8 GB | 16 GB |
| **Speicher** | 10 GB frei | 50 GB frei |
| **CUDA** | 11.8+ | 12.1+ |

**Unterstützte Modelle für Testing:**
- Qwen3-0.6B (~1.2 GB) – Pipeline-Validierung
- Qwen3-1.7B (~3.5 GB) – Erste fachliche Tests
- Qwen2.5-3B-Instruct (~6 GB) – Realistische Tests

### Produktion (H100 80GB)

| Komponente | Anforderung |
|------------|-------------|
| **GPU** | NVIDIA H100 80GB |
| **RAM** | 64 GB+ |
| **Speicher** | 200 GB+ für Checkpoints |
| **CUDA** | 12.1+ |

**Zielmodell:**
- Qwen3-32B (~65 GB mit QLoRA 4-bit)

---

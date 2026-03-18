# MASTERPLAN  – THE RELIABLE 32B



**Version 3**  

**Final Refinement – 18. März 2026**  

**Status:** Produktionsreif · Ready for immediate execution



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



## 0. Version-History (nur v3)



- **v3** (heute): Professionelle Struktur + komplette Roadmap + Risiken + Erwartete Verbesserungen mit konkreten Zielwerten + Nächste-Schritte-Abschnitt  

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



User Input

    ↓

Epistemic Classifier (Routing Head)

    ↓

Mode Selection

    ↓

Routing Decision

    ↓

Antwortgenerierung / Tool Call / Rückfrage



### Komponenten

- Base Model: **Qwen3-32B**  

- LoRA Adapter (rank 32)  

- Epistemic Routing Head (7-Klassen-Classifier)  

- Calibration Layer (Temperature Scaling)  

- Evaluation Suite + Red Team Engine  



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



5. REASONING FRAMEWORKEpistemischer Entscheidungsprozess (immer intern):Klassifikation der Aufgabe  

Prüfung der Prämisse  

Zeitliche Relevanz prüfen  

Tool-Bedarf prüfen  

Unsicherheit schätzen  

Modus auswählen  

Antwort generieren



6. TRAININGSPIPELINEHardware

1× NVIDIA H100 80 GB  Framework

Axolotl oder Unsloth  Quantisierung

QLoRA (4-bit)  Modellparameter  Base: Qwen3-32B  

LoRA Rank: 32  

LoRA Alpha: 64  

Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj



7. TRAININGSPHASENPhase 1 – SFTZiel: Modusverhalten & Routing stabilisieren  

80k Samples, 3 Epochen  

Dauer: ~4 Stunden auf H100



Phase 2 – DPOZiel: Halluzinationen bestrafen, ehrliche Antworten belohnen  

Ranking: Gold > Acceptable > Weak > Hallucination  

60k Paare  

Dauer: ~6 Stunden



8. ABLATION STUDIENRun

Beschreibung

A

Baseline (SFT + DPO)

B

+ NEFTune Noise (alpha=5)

C

+ MLP Noise (späte Layer)

D

+ Epistemic Head (final)



9. EPISTEMIC ROUTING HEADLinearer Classifier auf dem vorletzten Layer.

Output-Klassen: 7 Modi

Loss: Cross-Entropy  10. KALIBRIERUNGTemperature Scaling  

Optimiert auf Brier Score & Expected Calibration Error (ECE)  

Confidence basierend auf: Token Entropy + Logit Gap + Mode Probability



11. INFERENCE POLICY



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



12. EVALUATIONPrimäre BenchmarksTruthfulQA  

HaluEval  

WildBench



Sekundäre BenchmarksGPQA  

LiveBench



Eigene SuiteEpistemic Gap Eval (Ignorance, Staleness, False Premise, Ambiguity, Tool Required, Adversarial, Multi-Hop)  

Mode Confusion Matrix  

Utility Score



Utility Score Formel



correct_answer        +1.0

correct_cautious      +0.8

correct_clarify       +0.7

correct_tool_request  +0.7

correct_abstain       +0.5

unnecessary_abstain   -0.4

wrong_answer          -2.0

confident_wrong       -3.0



13. RED TEAMINGAdversarial Prompts (generiert durch zweites Modell):falsche historische Annahmen  

manipulatives Framing  

zeitliche Fallen  

Incentive Manipulation



Ziel: Modell aktiv zur Halluzination zwingen und Verhalten messen.14. KOMPLETTE ROADMAPPhase

Tage

Aufgabe

0

Tag 0

Repo + Infrastruktur + Modelle laden

1

Tag 1–2

Dataset Generator + SFT + DPO erstellen

2

Tag 3

SFT Training

3

Tag 4

DPO Training

4

Tag 5

Calibration + Confidence Mapping

5

Tag 6

Full Evaluation + Confusion Matrix

6

Tag 7

Red Teaming + Schwächen fixen

7

Woche 2+

Iterationen + Ablation-Vergleich



15. RISIKEN & MITIGATIONOver-Abstention → Utility-Monitoring + Threshold-Tuning  

Schein-Kalibrierung → echte ECE + Brier-Validierung  

Datenleakage im Eval → strikte Trennung  

Overfitting auf Benchmarks → Red-Team + WildBench-Fokus



16. ERWARTETE ERGEBNISSE (Ziele)TruthfulQA: +8–15 %  

HaluEval: –20–30 % Halluzinationen  

ECE: –40 %  

Abstention AUROC: +15 %  

Utility Score: deutlich höher durch korrekte Modi



17. ENDZIELEin Modell, das:lieber ehrlich nicht antwortet

als plausibel falsch zu sein.Damit wird Qwen3-32B zum verlässlichsten 32B-Wissensassistenten für kritische Anwendungen.The Reliable 32B18. NÄCHSTE SCHRITTE (sofort möglich)Ich habe bereits vorbereitet:dataset_generator.py  



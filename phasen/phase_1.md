# Phase 1 – Dataset Generator & Training Scripts

**Dauer:** Tag 1–2

## Ziele

- [ ] Dataset Generator für 80k SFT Samples
- [ ] Dataset Generator für 60k DPO Paare
- [ ] SFT Training Script
- [ ] DPO Training Script

## Hintergrund: SFT und DPO

### SFT (Supervised Fine-Tuning)

**Was ist SFT?**
Supervised Fine-Tuning (SFT) ist die erste Trainingsphase nach dem Pre-Training, bei der ein vortrainiertes Sprachmodell auf spezifische Aufgaben und Verhaltensweisen mittels überwachtem Lernen feinabgestimmt wird.

**Wie funktioniert es?**
- Das Modell lernt aus Eingabe-Ausgabe-Paaren `(prompt, expected_response)`
- Ziel: Die Wahrscheinlichkeit der korrekten Antwort gegeben den Prompt zu maximieren
- Loss-Funktion: Standard Cross-Entropy Loss über den Token-Level-Vorhersagen
- Formel: `L_SFT = -Σ log P(y_t | x, y_<t)` wobei `x` der Prompt und `y` die erwartete Antwort ist

**Warum SFT für Diogenes?**
- Vermittelt dem Modell die 7 epistemischen Modi (ABSTAIN, CAUTIOUS_LIMIT, CLARIFY, etc.)
- Lehrt strukturierte Reasoning-Traces für ehrliche Antworten
- Trainiert die Erkennung von Fehlerklassen (Ignorance, Staleness, Ambiguity, etc.)
- Basisverhalten für epistemisch ehrliche KI wird etabliert

**Typische Hyperparameter:**
- Learning Rate: 1e-4 bis 2e-5
- Batch Size: 4-16 (abhängig von Gradient Accumulation)
- Epochen: 2-4 (um Overfitting zu vermeiden)
- LoRA/QLoRA: Parameter-Efficient Fine-Tuning für große Modelle

---

### DPO (Direct Preference Optimization)

**Was ist DPO?**
Direct Preference Optimization (DPO) ist eine RLHF-Alternative (Reinforcement Learning from Human Feedback), die Präferenzdaten direkt optimiert, ohne ein separates Reward-Modell oder komplexe RL-Algorithmen wie PPO zu benötigen.

**Wie funktioniert es?**
- Eingabe: Präferenzpaare `(prompt, chosen_answer, rejected_answer)`
- Das Modell lernt: `P(chosen) > P(rejected)` für denselben Prompt
- DPO transformiert das Preference-Learning in ein klassisches Klassifikationsproblem
- Formel: `L_DPO = -log σ(β · [log(P(chosen)/P_rejected) - log(π_ref(chosen)/π_ref(rejected))])`
  - `σ`: Sigmoid-Funktion
  - `β`: Temperature-Parameter (typisch 0.1-0.5)
  - `π_ref`: Referenzmodell (meist das SFT-Modell vor DPO)

**Vorteile gegenüber PPO/RLHF:**
- **Stabiler**: Kein separates Reward-Modell nötig
- **Effizienter**: Direkte Optimierung ohne RL-Loop
- **Reproduzierbarer**: Weniger Hyperparameter, deterministischer
- **Ressourcenschonend**: Geringerer Speicherbedarf

**Warum DPO für Diogenes?**
- **Halluzinations-Reduktion**: Bestraft erfundene Antworten direkt
- **Ehrlichkeit belohnen**: "Ich weiß es nicht" wird positiv verstärkt
- **Präferenz-Ranking**: Gold > Acceptable > Weak > Hallucination
- **Kalibrierung**: Verbessert die Confidence-Schätzung des Modells

**Typische Hyperparameter:**
- Learning Rate: 5e-7 bis 1e-6 (niedriger als SFT)
- Batch Size: 2-8 (Präferenzpaare benötigen mehr Speicher)
- Epochen: 1-2 (DPO overfittet schnell)
- β (Beta): 0.1-0.3 (kontrolliert die Stärke der Präferenz)

---

### SFT vs. DPO im Diogenes-Workflow

```
Pre-trained Model (Qwen 7B)
         ↓
    [SFT Phase]
    - 80k Samples
    - Lernt epistemische Modi
    - Reasoning-Traces
         ↓
   SFT-Modell
         ↓
    [DPO Phase]
    - 60k Präferenzpaare
    - Halluzinations-Penalty
    - Ehrlichkeit belohnen
         ↓
  Diogenes-Modell (Final)
```

**SFT** bringt dem Modell *was* es tun soll (epistemisch ehrliche Antworten).
**DPO** verfeinert *wie* es es tut (weniger Halluzinationen, bessere Kalibrierung).

## Aufgaben

### 1. Dataset Generator erstellen

#### SFT Dataset (~80.000 Samples)
- 7 epistemische Modi abdecken
- Fehlerklassen generieren:
  - Ignorance → ABSTAIN
  - Staleness → CAUTIOUS_LIMIT
  - Ambiguity → CLARIFY
  - False Premise → REJECT_PREMISE
  - Adversarial → DIRECT_ANSWER
  - Shallow Trap → PROBABILISTIC
  - Multi-Hop → PROBABILISTIC
  - Tool Required → REQUEST_TOOL

#### DPO Dataset (~60.000 Paare)
- Ranking: Gold > Acceptable > Weak > Hallucination
- Halluzinations-Reduktion fokussieren
- Präferenzdaten strukturieren

#### JSON Schema implementieren
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

### 2. SFT Training Script
- LoRA Konfiguration (rank 32, alpha 64)
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- QLoRA (4-bit) Quantisierung
- 3 Epochen, ~80k Samples
- Batch Size & Learning Rate optimieren

### 3. DPO Training Script
- Preference Pairing implementieren
- Hallucination Penalty
- Ehrliche Antworten belohnen
- 60k Paare verarbeiten

## Deliverables

- [ ] `dataset_generator.py` – voll funktionsfähig
- [ ] `train_sft.py` – SFT Training Script
- [ ] `train_dpo.py` – DPO Training Script
- [ ] Test-Datensatz (Stichprobe) validiert

## Erfolgskriterien

- Generator erstellt valide JSON-Daten
- SFT Script startet ohne Fehler
- DPO Script verarbeitet Preference Pairs
- Datenqualität überprüft (Stichprobe)

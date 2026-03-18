# Phase 1 – Dataset Generator & Training Scripts

**Dauer:** Tag 1–2

## Ziele

- [ ] Dataset Generator für 80k SFT Samples
- [ ] Dataset Generator für 60k DPO Paare
- [ ] SFT Training Script
- [ ] DPO Training Script

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

# Phase 1 – Dataset Generator & Training Scripts

**Dauer:** Tag 1–2

**Status:** ✅ **ABGESCHLOSSEN** (18. März 2026)

**Hardware:** NVIDIA RTX 3050 (8GB VRAM)

## Ziele

- [x] Dataset Generator für 80k SFT Samples
- [x] Dataset Generator für 60k DPO Paare
- [x] SFT Training Script
- [x] DPO Training Script
- [x] DPO-Audit Script
- [x] Pass@1 Protection implementiert

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
Pre-trained Model (Qwen 3B für Testing)
         ↓
    [SFT Phase] (RTX 3050)
    - 80k Samples
    - Lernt epistemische Modi
    - Reasoning-Traces
         ↓
   SFT-Modell
         ↓
    [DPO Phase] (RTX 3050)
    - 60k Präferenzpaare
    - Halluzinations-Penalty
    - Ehrlichkeit belohnen
         ↓
  Diogenes-Modell (3B Test)
         ↓
    [Phase 7: Produktion auf H100]
    - Qwen3-32B Final Training
```

**SFT** bringt dem Modell *was* es tun soll (epistemisch ehrliche Antworten).
**DPO** verfeinert *wie* es es tut (weniger Halluzinationen, bessere Kalibrierung).

## Aufgaben

### 1. Dataset Generator erstellen ✅

#### SFT Dataset (~80.000 Samples)
- [x] 7 epistemische Modi abdecken
- [x] Fehlerklassen generieren:
  - Ignorance → ABSTAIN
  - Staleness → CAUTIOUS_LIMIT
  - Ambiguity → CLARIFY
  - False Premise → REJECT_PREMISE
  - Adversarial → DIRECT_ANSWER
  - Shallow Trap → PROBABILISTIC
  - Multi-Hop → PROBABILISTIC
  - Tool Required → REQUEST_TOOL

#### DPO Dataset (~60.000 Paare)
- [x] Ranking: Gold > Acceptable > Weak > Hallucination
- [x] Halluzinations-Reduktion fokussieren
- [x] Präferenzdaten strukturieren

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

### 2. SFT Training Script ✅
- [x] LoRA Konfiguration (rank 32, alpha 64)
- [x] Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- [x] QLoRA (4-bit) Quantisierung
- [x] 3 Epochen, ~80k Samples
- [x] Batch Size & Learning Rate optimieren
- [x] RTX 3050-kompatibel (8GB VRAM)
- [x] Wandb-Logging deaktivierbar

### 3. DPO Training Script ✅
- [x] Preference Pairing implementieren
- [x] Hallucination Penalty
- [x] Ehrliche Antworten belohnen
- [x] 60k Paare verarbeiten
- [x] RTX 3050-kompatibel (8GB VRAM)
- [x] TRL DPOTrainer Integration

### 4. DPO-Audit Script ✅ (Neu)
- [x] `dpo_audit.py` für Prompt-Interferenz-Prüfung
- [x] Difficulty Bias Check (< 55% hard für Diogenes)
- [x] Verbosity Bias Check (< 1.2 Ratio)
- [x] Abstain Representation Check (> 5%)
- [x] Audit-Report Generierung

### 5. Pass@1 Protection ✅
- [x] Zwei-Tier-Evaluationssystem implementiert
- [x] Core Reliability Metrics (Tier 1)
- [x] Special Metrics für Math/Code (Tier 2, nur Monitoring)
- [x] Regression Detection für Checkpoint-Promotion
- [x] DPO Audit für Prompt-Interferenz

## Deliverables

- [x] `src/diogenes/dataset_generator.py` – voll funktionsfähig
- [x] `src/diogenes/train_sft.py` – SFT Training Script
- [x] `src/diogenes/train_dpo.py` – DPO Training Script
- [x] `src/diogenes/dpo_audit.py` – DPO-Audit Script (NEU)
- [x] `src/diogenes/eval_metrics.py` – Core Reliability Metrics
- [x] `src/diogenes/pass1_protection.py` – Pass@1 Schutz
- [x] `scripts/run_dpo_training.sh` – DPO-Training Script (NEU)
- [x] `scripts/pass1_protection_check.py` – Pass@1 Check (NEU)
- [x] `docs/PASS1_GUARDRAILS.md` – Produkt-Richtlinien
- [x] `docs/phase3_dpo_ready.md` – Phase 3 Anleitung (NEU)
- [x] Test-Datensatz (Stichprobe) validiert

## Erfolgskriterien

- [x] Generator erstellt valide JSON-Daten
- [x] SFT Script startet ohne Fehler auf RTX 3050
- [x] DPO Script verarbeitet Preference Pairs auf RTX 3050
- [x] DPO-Audit besteht mit akzeptierten Werten
- [x] Datenqualität überprüft (Stichprobe)
- [x] Pass@1 Protection getestet

## Implementierte Komponenten

### `dataset_generator.py`

**Funktionen:**
- Generiert SFT-Samples mit 7 epistemischen Modi
- Erstellt DPO-Präferenzpaare mit Hallucination-Ranking
- Unterstützt Reasoning-Traces und Confidence-Targets
- Tagging nach `risk_level`, `time_sensitive`, `needs_tool`

**Usage:**
```bash
# SFT Dataset generieren
python src/diogenes/dataset_generator.py --sft-samples 80000

# DPO Dataset generieren
python src/diogenes/dataset_generator.py --dpo-pairs 60000
```

**Erstellte Datasets:**
- `datasets/sft_dataset.jsonl` (80.000 Samples)
- `datasets/dpo_dataset.jsonl` (60.000 Paare)
- `datasets/sft_statistics.json`
- `datasets/dpo_statistics.json`

### `train_sft.py`

**Features:**
- LoRA/QLoRA Support (4-bit Quantisierung)
- Konfigurierbare Hyperparameter
- Checkpoint-Speicherung
- Gradient Accumulation für große Batch-Sizes
- **RTX 3050 optimiert (8GB VRAM)**

**Usage:**
```bash
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_path datasets/sft_dataset.jsonl \
  --output_dir models/sft_3b_test \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4
```

**VRAM-Nutzung:**
- Qwen3-0.6B: ~1.5 GB
- Qwen2.5-3B: ~6 GB (mit QLoRA 4-bit)

### `train_dpo.py`

**Features:**
- Direct Preference Optimization
- Hallucination Penalty im Loss
- Referenzmodell-Integration
- Beta-Parameter für Präferenz-Stärke
- Early Stopping bei Overfitting
- **RTX 3050 optimiert (8GB VRAM)**

**Usage:**
```bash
python src/diogenes/train_dpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --sft_model_path models/sft_3b_test \
  --dataset_path datasets/dpo_dataset.jsonl \
  --output_dir models/dpo_3b_test \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --beta 0.2
```

**VRAM-Nutzung:**
- Qwen3-0.6B: ~2 GB (mit Referenzmodell)
- Qwen2.5-3B: ~7 GB (mit QLoRA 4-bit und Referenzmodell)

### `dpo_audit.py` (NEU)

**Funktionen:**
- Prüft DPO-Dataset auf Prompt-Interferenz
- Difficulty Bias Check (< 55% hard für Diogenes)
- Verbosity Bias Check (< 1.2 Ratio)
- Abstain Representation Check (> 5%)
- Generiert Audit-Report

**Usage:**
```bash
python src/diogenes/dpo_audit.py \
  --dataset_path datasets/dpo_dataset.jsonl \
  --output_path dpo_audit_report.json
```

**Audit-Ergebnis (aktuell):**
- ✅ Difficulty: 54.9% hard (akzeptiert)
- ✅ Verbosity: 0.78 Ratio (gut)
- ✅ Abstain: 14.9% (gut)
- ✅ **AUDIT BESTANDEN**

### `eval_metrics.py`

**Tier 1: Core Reliability Metrics**
- `compute_pass_at_k()` - Pass@1 und Pass@k
- `compute_expected_calibration_error()` - ECE
- `compute_brier_score()` - Kalibrierung
- `compute_abstention_auroc()` - Wissensgrenzen
- `compute_hallucination_rate()` - Halluzinationen
- `compute_core_reliability_metrics()` - Komplette Suite

**Tier 2: Special Metrics**
- `compute_special_metrics()` - Nur für Math/Code
- Pass@k Monitoring (nie für Optimierung)

### `pass1_protection.py`

**Komponenten:**
- `Pass1RegressionTracker` - Checkpoint-Überwachung
- `run_pass1_protection_test()` - Vollständige Evaluation
- `check_dpo_for_prompt_interference()` - DPO-Audit

## Entscheidungsmatrix für Checkpoint-Promotion

| Bedingung | Pass@1 Δ | Pass@k Δ | Aktion |
|-----------|----------|----------|--------|
| **Kritische Regression** | < -2% | > +1% | ❌ NICHT PROMOTEN |
| **Warnung** | < -1% | > +0.5% | ⚠️ Vorsichtig prüfen |
| **Verbesserung** | > +1% | Beliebig | ✓ Sicher |
| **Stabil** | ±1% | Beliebig | ✓ Sicher |

## DPO-Audit Grenzwerte

| Metrik | Schwellenwert | Status |
|--------|---------------|--------|
| Difficulty Bias | < 55% hard | ✓ Pass (54.9%) |
| Verbosity Bias | < 1.2 Ratio | ✓ Pass (0.78) |
| Abstain Repr. | > 5% | ✓ Pass (14.9%) |

## Entwicklungs-Workflow

### Phase 1: Lokale Entwicklung (RTX 3050)

1. **Scripts entwickeln** mit Qwen3-0.6B (~1.5 GB VRAM)
2. **Testing** mit Qwen2.5-3B-Instruct (~6 GB VRAM)
3. **Hyperparameter tuning** auf RTX 3050
4. **Validierung** mit Evaluation Suite

### Phase 7: Produktion (H100 80GB)

1. **Finales Training** mit Qwen3-32B (~65 GB VRAM)
2. **Full-Scale SFT** und DPO
3. **Production Evaluation**

## Nächste Schritte

➡️ **Phase 2**: SFT Testing auf RTX 3050 mit Qwen2.5-3B-Instruct

```bash
# 1. Dataset vorbereiten (bereits erledigt)
ls -lh datasets/sft_dataset.jsonl

# 2. SFT Training lokal starten
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_path datasets/sft_dataset.jsonl \
  --output_dir models/sft_3b_test \
  --num_train_epochs 3

# 3. Training überwachen
tail -f /tmp/sft_train.log
watch -n 2 nvidia-smi
```

➡️ **Phase 3**: DPO Testing auf RTX 3050

```bash
# Nach SFT-Completion
./scripts/run_dpo_training.sh
```

➡️ **Phase 7**: Finales Training auf H100 (nach lokaler Validierung)

## Referenzen

- `docs/PASS1_GUARDRAILS.md` – Vollständige Pass@1-Richtlinien
- `docs/IMPLEMENTATION_SUMMARY.md` – Implementierungs-Übersicht
- `docs/phase3_dpo_ready.md` – Phase 3 Anleitung
- `tests/test_pass1_protection.py` – Test-Suite
- `docs/phase0_quickstart.md` – RTX 3050 Setup-Guide

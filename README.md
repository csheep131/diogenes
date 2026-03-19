# Diogenes – The Reliable 32B

**Version 7** | **Stand:** 19. März 2026

A specialized language model based on **Qwen3-32B**, optimized for **epistemic reliability**.

## Project Overview

Diogenes focuses on epistemic correctness rather than traditional LLM benchmarks:

- **Recognizing knowledge boundaries**
- **Minimizing hallucinations**
- **Correct uncertainty estimation**
- **Epistemically appropriate responses**
- **Tool awareness**

The model is designed for deployment in critical domains (IT, manufacturing, medicine, law, finance).

---

## Current Status

**As of March 19, 2026, 00:00 UTC**

### Development Workflow

**All testing and development runs on local RTX 3050 (8GB VRAM).** Only the final production model will be trained on H100 infrastructure.

### ✅ Completed (Phase 0-1)

- **Phase 0**: Infrastructure & Pipeline Validation ✅ **ABGESCHLOSSEN**
  - Repository structure established
  - Development environment configured for RTX 3050 (8GB)
  - Virtual environment (.venv) with all dependencies
  - Model download scripts (GGUF & Transformers formats)
  - Inference pipeline validated on consumer hardware
  - Epistemic mode detection tested

- **Phase 1**: Dataset Generator & Training Scripts ✅ **ABGESCHLOSSEN**
  - `dataset_generator.py` – SFT (80k samples) and DPO (60k pairs) generation
  - `train_sft.py` – Supervised Fine-Tuning with LoRA/QLoRA support
  - `train_dpo.py` – Direct Preference Optimization with hallucination penalty
  - `dpo_audit.py` – DPO-Audit for prompt interference detection (NEU)
  - Pass@1 protection mechanisms implemented
  - TRL integration for DPO training

### 🔄 In Progress

- **Phase 2**: SFT Testing on RTX 3050 🔄 **LÄUFT** (seit 20:58)
  - Training: Qwen2.5-3B-Instruct with QLoRA 4-bit
  - Progress: ~1% (685/60000 steps for 1 epoch)
  - Speed: ~1.76s/step
  - Expected completion: ~30 hours for 3 epochs
  - GPU utilization: 100%, 65°C, 4.9 GB VRAM

### 📋 Ready to Start

- **Phase 3**: DPO Testing 📋 **BEREIT** (wartet auf SFT-Completion)
  - DPO-Dataset: 60k preference pairs ✅ generiert
  - DPO-Audit: ✅ **BESTANDEN** (Difficulty: 54.9%, Verbosity: 0.78, Abstain: 14.9%)
  - Training script: `scripts/run_dpo_training.sh` ✅ vorbereitet
  - Expected duration: ~15-20 hours

### ⏳ Planned

- **Phase 2.5**: Shadow Loop ⏳ **GEPLANT** (nach Phase 2)
  - Parallel-Experiment: Custom-Loop läuft neben HF-Training
  - Epistemic Regularization: `L_total = L_DPO + λ · max(0, H_pred - H_target)`
  - Minimal PyTorch script (~200 lines) testing isolated aspects
  - Exit-criterion: Shadow-Loop beats HF-Loop in ≥2 metrics → unlock Phase 3

- **Phase 3.5**: SUA Specialization ⏳ **GEPLANT** (nach Phase 3)
  - Staleness/Unknown/Ambiguity Fine-Tuning
  - Dataset: 25k spezialisierte Samples (8k stale, 10k unknown, 7k ambiguity)
  - Low-Rate Fine-Tuning (LR=5e-6, 1-2 Epochen) für Minimal-Invasion
  - Pass@1 Protection: Obligatorische Checks nach jedem Epoch
  - Expected duration: ~8-12 hours on RTX 3050

- **Phase 4-6**: Calibration, Evaluation, Red Teaming (after Phase 3.5)
- **Phase 7**: Final Production Training on H100 (pending local validation)

---

## Training Strategy v2.0: Velocity vs. Sovereignty ⭐ NEU

### The Core Dilemma

In AI development, there's a classic trade-off:

**Feature Velocity**: Fast iteration with Hugging Face `trl` + `peft` + `DPOTrainer`
- ✅ High experiment velocity
- ✅ Minimal code overhead
- ❌ Limited control over gradient flow, loss design, sampling logic

**Architectural Sovereignty**: Complete control over alignment process
- ✅ Precise control of loss, gradients, sampling
- ✅ Custom loss functions possible
- ✅ Online auditing in training loop
- ❌ Slower iteration, higher debugging effort

### Strategic Rule

| Phase | Priority | Rationale |
|-------|----------|-----------|
| **Phase 1 & 2** | 100% Velocity | We must first validate whether Diogenes data actually "ignites" |
| **Phase 3+** | 100% Sovereignty | We optimize not just the model, but the entire alignment process |

---

### Decision Gates: When to Switch?

Instead of a rigid phase switch, we define **hard, measurable checkpoints**. Switch to Custom-Loop only when **at least two** of the following conditions are met across **three consecutive training runs**:

| Symptom | Cause | KPI / Threshold | Solution via Custom-Loop |
|---------|-------|-----------------|--------------------------|
| **Loss Stagnation** | Standard CE weights epistemic uncertainty too weakly | Plateau over >5% of training (val_loss Δ < 0.001) | **Custom Loss Scaling**: Dynamic penalty for "confident hallucinations" |
| **Mode Collapse** | DPO suppresses rare epistemic modes | < 15% of generated samples show "I don't know" behavior despite ground-truth | **Dynamic Batching + Mode Balancing**: Adaptive mixing per batch (rare-mode oversampling) |
| **Audit Lag** | Audit insights flow back too slowly | > 2 hours delay between audit and next training step | **Online Rejection Sampling**: Live filtering + instant reward update in same loop |
| **Gradient Interference** | SFT and DPO objectives collide (Catastrophic Forgetting) | > 20% performance drop on old fact-tasks after DPO | **Multi-Objective Gradient Surgery**: Separate update paths + PCGrad / GradNorm |

**Additional Gate (v2.0): Epistemic Drift** → If the model suddenly produces >8% "uncertainty" on stable facts → immediate switch to Phase 3.

---

### Phase Architecture with Exit Criteria

**Phase 2: Baseline & Stress-Test (2-3 weeks)**
- Framework: 100% `trl` + `peft` + `DPOTrainer`
- Goal: Establish metric baseline (Win-Rate, Epistemic-Score, Hallucination-Rate, Fact-Retention)
- Exit-criterion: If **≥2 Decision Gates trigger** → immediately proceed to Phase 2.5

**Phase 2.5: Shadow Loop (1-2 weeks) ⭐ NEW**
- Framework: Custom-Loop runs **alongside** HF training (not replacement, but shadow)
- Technique: Minimal PyTorch script (~200 lines) isolating one aspect (e.g., Sampling + Epistemic Regularization)
- Key Innovation: **Epistemic Regularization Term**
  ```
  L_total = L_DPO + λ · max(0, H_pred - H_target)
  ```
- Goal: Model learns to have **minimal entropy** on "I don't know" questions (being sure about being unsure)
- Exit-criterion: Shadow-Loop beats HF-Loop in **≥2 metrics** → unlock Phase 3

**Phase 3: Diogenes Alignment Engine (full sovereignty) ⭐ NEW**
- Paradigm shift: From "Fine-Tuning" to **Conditioned Alignment**
- Components:
  1. **In-Loop Auditing**: Model generates 8 samples during training → Mini-auditor evaluates in <50ms → Loss adjusted immediately
  2. **Curriculum Acceleration**: Loop tracks Mastery-Score per epistemic mode → automatically fades out mastered modes (up to 40% compute savings)
  3. **Technical Implementation**: Single `train_step` loop with `torch.autograd` + custom DataLoader (no HF trainer anymore)

---

## What Makes Diogenes Training Different

Most LLM fine-tuning projects optimize for **helpfulness** and **engagement**. Diogenes optimizes for **epistemic honesty** — the model learns to recognize when it *doesn't* know something.

### Traditional RLHF vs. Diogenes Approach

| Aspect | Traditional RLHF | Diogenes Training |
|--------|-----------------|-------------------|
| **Primary Goal** | Helpful, engaging responses | Epistemically correct responses |
| **"I don't know"** | Penalized (seen as unhelpful) | Rewarded (honest acknowledgment) |
| **Hallucinations** | Often rewarded if plausible | Explicitly penalized |
| **Confidence** | Always high (user preference) | Calibrated to actual knowledge |
| **Training Signal** | Human preference (what sounds good) | Epistemic correctness (what is knowable) |

### Key Differentiators

#### 1. **Epistemic Mode Training**
Unlike standard models that always attempt an answer, Diogenes learns 7 distinct response modes:
- `ABSTAIN` — Honest knowledge gap acknowledgment
- `CAUTIOUS_LIMIT` — Answer with explicit limitations
- `REJECT_PREMISE` — Correcting false premises instead of answering them
- `REQUEST_TOOL` — Recognizing when external tools/data are needed

Most models are trained to *always answer*. Diogenes learns *when not to answer*.

#### 2. **Hallucination-First DPO**
Standard DPO optimizes for human preference rankings. Our DPO training:
- Prioritizes **hallucination reduction** as the primary signal
- Uses a custom **Hallucination Penalty** in the loss function
- Ranks responses: `Gold > Acceptable > Weak > Hallucination`
- Explicitly trains on **false premise detection** (7 epistemic error classes)

#### 3. **Confidence Calibration as Core Objective**
Traditional models output confidence scores that are poorly calibrated. Diogenes:
- Trains with explicit `confidence_target` labels in the dataset
- Uses **Expected Calibration Error (ECE)** as a primary metric
- Includes a dedicated **Phase 3 - Calibration** with temperature scaling
- Targets 40% ECE reduction compared to base model

#### 4. **Knowledge Boundary Datasets**
Instead of generic instruction data, we generate:
- **80k SFT samples** covering 7 epistemic error classes
- **60k DPO preference pairs** focused on uncertainty scenarios
- Samples tagged with `risk_level`, `time_sensitive`, `needs_tool`
- Explicit **reasoning traces** for epistemic decision-making

#### 5. **Domain-Specific Reliability**
While general models optimize for broad capabilities, Diogenes:
- Targets critical domains: **medicine, law, finance, manufacturing, IT**
- Prioritizes **safety over engagement** in high-risk scenarios
- Includes **Red Teaming (Phase 6)** specifically for hallucination attacks

### Training Philosophy

```
Traditional Training:
  User asks → Model answers (always confident, always helpful)

Diogenes Training:
  User asks → Model evaluates:
    ├─ Do I know this? → Answer with appropriate confidence
    ├─ Is this knowable? → Acknowledge limitations
    ├─ Is the question flawed? → Reject false premises
    ├─ Do I need tools? → Request external data
    └─ Am I uncertain? → Express probabilistic reasoning
```

### Why This Matters

In critical domains, a **confident wrong answer** is worse than **no answer**:
- **Medical advice**: Hallucinated drug interactions can harm patients
- **Legal guidance**: Incorrect citations can damage cases
- **Financial advice**: Fabricated regulations can cause losses
- **Manufacturing**: Wrong specifications can cause failures

Diogenes is built for scenarios where **being reliably uncertain** is more valuable than **being confidently wrong**.

---

## Epistemic Modes

The model decides between seven response modes for each query:

| Mode | Description |
|------|-------------|
| `DIRECT_ANSWER` | Confident, direct answer |
| `CAUTIOUS_LIMIT` | Answer with clear limitations |
| `ABSTAIN` | Honest knowledge gap acknowledgment |
| `CLARIFY` | Request clarification for ambiguous queries |
| `REJECT_PREMISE` | Correct false premises |
| `REQUEST_TOOL` | Request external data/tools |
| `PROBABILISTIC` | Uncertain but plausible reasoning |

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with 8GB+ VRAM (for development), 80GB+ for production (H100)
- CUDA 12.1+
- venv for environment management

### Quick Start

```bash
# Clone repository
git clone https://github.com/diogenes/diogenes.git
cd diogenes

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -e ".[dev]"

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Phase 0: Download small test model (Qwen3-0.6B ~1.2GB)
python scripts/download_model.py

# Test inference
python scripts/test_inference.py
```

---

## Project Structure

```
diogenes/
├── src/                     # Source code
│   └── diogenes/            # Main package
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       ├── dataset_generator.py  # SFT/DPO/SUA data generation
│       ├── dpo_audit.py     # DPO-Audit for prompt interference (NEU)
│       ├── eval_metrics.py  # Core reliability metrics + SUA (Tier 3)
│       ├── inference.py     # Inference engine
│       ├── model.py         # Model loading/wrapping
│       ├── pass1_protection.py  # Regression detection
│       ├── train_sft.py     # SFT training script
│       ├── train_dpo.py     # DPO training script
│       └── train_sua.py     # SUA training script (Phase 3.5, NEU)
├── configs/                 # Configuration files
│   ├── config.yaml
│   └── remote_config.yaml
├── datasets/                # Training/evaluation datasets
│   ├── sft_dataset.jsonl    (80k samples)
│   ├── dpo_dataset.jsonl    (60k pairs)
│   ├── sua_dataset.jsonl    (22.5k samples, Phase 3.5)
│   ├── sua_eval_holdout.jsonl (2.5k samples)
│   ├── sft_statistics.json
│   ├── dpo_statistics.json
│   └── sua_statistics.json
├── models/                  # Model checkpoints
├── scripts/                 # Utility scripts
│   ├── download_model.py
│   ├── download_gguf.py
│   ├── test_inference.py
│   ├── setup_env.py
│   ├── prepare_remote_machine.py
│   ├── run_dpo_training.sh  # DPO-Training Script
│   ├── run_sua_training.sh  # SUA-Training Script (Phase 3.5, NEU)
│   └── pass1_protection_check.py  # Pass@1 Check
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── PASS1_GUARDRAILS.md
│   ├── phase0_quickstart.md
│   ├── phase3_dpo_ready.md  # Phase 3 Anleitung
│   └── phase3.5_sua_ready.md  # Phase 3.5 Anleitung (NEU)
├── phasen/                  # Phase documentation (DE)
│   ├── phase_0.md through phase_7.md
│   └── phase_3.5.md  # SUA Specialization (NEU)
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

---

## Usage

### Load Base Model

```python
from diogenes.model import DiogenesModel

model = DiogenesModel.from_pretrained("Qwen/Qwen3-32B")
```

### Run Inference

```python
from diogenes.inference import DiogenesInference

inference = DiogenesInference(model)
response = inference.generate("What is the capital of France?")
print(response)
```

---

## Training Pipeline

### Development Workflow

**Local Testing (RTX 3050 8GB):**
- All pipeline validation with smaller models (Qwen3-0.6B to Qwen2.5-3B)
- Hyperparameter tuning and ablation studies
- Evaluation suite testing
- Pass@1 protection validation

**Production Training (H100 80GB):**
- Final Qwen3-32B training after local validation
- Full-scale SFT and DPO on target model

### Implemented Components

1. **Dataset Generation** ✅
   - SFT Dataset: 80k samples covering 7 epistemic modes
   - DPO Dataset: 60k preference pairs with hallucination penalty
   - JSON schema with reasoning traces, risk levels, confidence targets
   - **DPO-Audit bestanden** ✅ (Difficulty: 54.9%, Verbosity: 0.78)

2. **Training Scripts** ✅
   - `train_sft.py`: LoRA/QLoRA (4-bit) with rank 32, alpha 64
   - `train_dpo.py`: Direct Preference Optimization with β=0.1-0.3
   - `dpo_audit.py`: DPO-Audit for prompt interference detection
   - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

3. **Evaluation & Protection** ✅
   - Pass@1 regression detection
   - Core Reliability Metrics (Tier 1)
   - Special Metrics for math/code (Tier 2, monitoring only)
   - DPO audit for prompt interference

4. **Automation Scripts** ✅
   - `scripts/run_dpo_training.sh`: DPO-Training automatisieren
   - `scripts/pass1_protection_check.py`: Pass@1 Regression prüfen

### Training Phases

#### Local Development (RTX 3050 8GB)

| Phase | Model | Duration | Status |
|-------|-------|----------|--------|
| **Phase 0** | Qwen3-0.6B | < 1 day | ✅ **ABGESCHLOSSEN** |
| **Phase 1** | Qwen3-0.6B | 1-2 days | ✅ **ABGESCHLOSSEN** |
| **Phase 2** | Qwen2.5-3B | ~30 hours | 🔄 **LÄUFT** (1%) |
| **Phase 2.5** | Qwen2.5-3B | ~8-12 hours | ⏳ **GEPLANT** (Shadow Loop) |
| **Phase 3** | Qwen2.5-3B | ~15-20 hours | 📋 **BEREIT** |
| **Phase 3.5** | Qwen2.5-3B | ~8-12 hours | ⏳ **GEPLANT** (SUA Specialization) |
| **Phase 4** | Qwen2.5-3B | 1 day | ⏳ **GEPLANT** |
| **Phase 5** | Qwen2.5-3B | 1-2 days | ⏳ **GEPLANT** |
| **Phase 6** | Qwen2.5-3B | 1-2 days | ⏳ **GEPLANT** |

#### Production Training (H100 80GB)

| Phase | Model | Duration | Status |
|-------|-------|----------|--------|
| **Phase 7-A** | Qwen3-32B | ~4 hours | ⏳ **GEPLANT** |
| **Phase 7-B** | Qwen3-32B | ~6 hours | ⏳ **GEPLANT** |
| **Phase 7-B.1** | Qwen3-32B | ~2 hours | ⏳ **GEPLANT** (SUA Specialization) |
| **Phase 7-C** | Qwen3-32B | ~2 hours | ⏳ **GEPLANT** |
| **Phase 7-D** | Qwen3-32B | ~4 hours | ⏳ **GEPLANT** |

---

## Evaluation Metrics

### Tier 1: Core Reliability (Primary)

| Metric | Description | Target |
|--------|-------------|--------|
| **Pass@1** | Single-attempt accuracy | Maximize |
| **Hallucination Rate** | False claim frequency | < 5% |
| **ECE** | Expected Calibration Error | < 0.05 (–40%) |
| **Brier Score** | Probability calibration | Minimize |
| **Abstention AUROC** | Knowledge boundary detection | +15% |
| **Mode Accuracy** | Epistemic mode classification | Maximize |
| **False Premise Detection** | Flawed question recognition | Maximize |

### Tier 2: Special Metrics (Monitoring Only)

| Metric | Domain | Usage |
|--------|--------|-------|
| **Pass@k** | Math, Code | Monitor only, never optimize |
| **Best-of-k** | Tool-assisted | Special cases |

> **⚠️ Pass@1 Protection**: Tier 2 metrics are NEVER used for checkpoint promotion or global reward optimization. See `docs/PASS1_GUARDRAILS.md`.

### Tier 3: SUA Metrics (Phase 3.5 Specialization)

| Metric | Description | Target |
|--------|-------------|--------|
| **Staleness Detection Rate** | % korrekt als veraltet markierter Samples | > 80% |
| **Staleness False Positive Rate** | % fälschlich als veraltet markierter Samples | < 10% |
| **Unknown Detection AUROC** | Fläche unter ROC-Kurve für Wissenslücken | > 0.85 |
| **Unknown Precision** | Präzision bei Unknown-Vorhersagen | > 75% |
| **Ambiguity Resolution Accuracy** | % korrekt aufgelöster mehrdeutiger Anfragen | > 75% |
| **Clarification Quality Score** | Qualität der Rückfragen (1-5 Skala) | > 3.5 |

### Traditional Benchmarks

- **TruthfulQA**: Target +8–15%
- **HaluEval**: Target –20–30% hallucinations
- **WildBench**: Real-world performance
- **GPQA**: Expert-level knowledge
- **LiveBench**: Current capabilities

---

## Expected Results

### Core Reliability Metrics

| Metric | Target Improvement |
|--------|-------------------|
| TruthfulQA | +8–15% |
| HaluEval | –20–30% hallucinations |
| ECE | –40% |
| Abstention AUROC | +15% |
| Utility Score | Significantly higher |

### SUA Specialization Metrics (Phase 3.5)

| Metric | Target |
|--------|--------|
| **Staleness Detection Rate** | > 80% |
| **Unknown Detection AUROC** | > 0.85 |
| **Ambiguity Resolution Accuracy** | > 75% |
| **Pass@1 Degradation** | < 1% (PRIMARY) |

---

## Monitoring Training

### Check SFT Training Progress

```bash
# View training log
tail -f /tmp/sft_train.log

# Monitor GPU utilization
watch -n 2 nvidia-smi

# Check process status
ps aux | grep train_sft | grep -v grep
```

### Start DPO Training (after SFT completes)

```bash
# Simple start
./scripts/run_dpo_training.sh

# Or with explicit paths
./scripts/run_dpo_training.sh \
    models/sft_3b_test/final_checkpoint \
    models/dpo_3b_test \
    datasets/dpo_dataset.jsonl \
    2
```

### Run Pass@1 Protection Check

```bash
python3 scripts/pass1_protection_check.py \
    --model-path models/dpo_3b_test \
    --baseline-pass-at-1 0.75
```

---

## License

Apache License 2.0

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

---

## Citation

```bibtex
@software{diogenes2026,
  title = {Diogenes: The Reliable 32B},
  year = {2026},
  description = {Epistemically optimized language model based on Qwen3-32B}
}
```

---

## Quick Links

- **Phase Documentation**: `phasen/phase_0.md` – `phasen/phase_7.md`
- **Phase 2.5 (Shadow Loop)**: `phasen/phase_2.5.md` ⭐ **NEU**
- **Phase 3 (Alignment Engine)**: `phasen/phase_3.md` ⭐ **NEU**
- **Phase 3.5 (SUA Specialization)**: `phasen/phase_3.5.md`
- **Pass@1 Guardrails**: `docs/PASS1_GUARDRAILS.md`
- **Phase 3 Guide**: `docs/phase3_dpo_ready.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **RTX 3050 Setup**: `docs/phase0_quickstart.md`
- **Training Strategy v2.0**: Siehe `roadmap.md#17-strategischer-rahmen-velocity-vs-sovereignty`

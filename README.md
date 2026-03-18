# Diogenes – The Reliable 32B

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

**As of March 18, 2026**

### Development Workflow

**All testing and development runs on local RTX 3050 (8GB VRAM).** Only the final production model will be trained on H100 infrastructure.

### ✅ Completed (Phase 0-1)

- **Phase 0**: Infrastructure & Pipeline Validation ✅
  - Repository structure established
  - Development environment configured for RTX 3050 (8GB)
  - Model download scripts (GGUF & Transformers formats)
  - Inference pipeline validated on consumer hardware
  - Epistemic mode detection tested

- **Phase 1**: Dataset Generator & Training Scripts ✅
  - `dataset_generator.py` – SFT (80k samples) and DPO (60k pairs) generation
  - `train_sft.py` – Supervised Fine-Tuning with LoRA/QLoRA support
  - `train_dpo.py` – Direct Preference Optimization with hallucination penalty
  - Pass@1 protection mechanisms implemented

### 🔧 Implemented Features

- **7 Epistemic Modes**: DIRECT_ANSWER, CAUTIOUS_LIMIT, ABSTAIN, CLARIFY, REJECT_PREMISE, REQUEST_TOOL, PROBABILISTIC
- **Pass@1 Protection**: Two-tier evaluation system preventing Pass@k optimization from degrading Pass@1 performance
- **Core Reliability Metrics**: Pass@1, ECE, Brier Score, Hallucination Rate, Abstention AUROC
- **DPO Audit Tools**: Detects prompt interference and bias patterns in preference data
- **Regression Detection**: Automated checkpoint monitoring for training stability

### 📋 In Progress

- **Phase 2-6**: Testing & Validation on RTX 3050 (8GB)
  - SFT/DPO pipeline validation with smaller models (0.6B-3B)
  - Hyperparameter tuning on local hardware
  - Evaluation suite testing
- **Phase 7**: Final Production Training on H100 (pending)
  - Full Qwen3-32B training after local validation

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
- Includes **Red Teaming (Phase 5)** specifically for hallucination attacks

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

## Installation

### Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with 80GB+ VRAM (H100 recommended)
- CUDA 12.1+
- conda or venv for environment management

### Quick Start

```bash
# Clone repository
git clone https://github.com/diogenes/diogenes.git
cd diogenes

# Create conda environment
conda create -n diogenes python=3.10
conda activate diogenes

# Install dependencies
pip install -e ".[dev]"

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Phase 0: Download small test model (Qwen3-0.6B ~1.2GB)
python scripts/download_model.py

# Test inference
python scripts/test_inference.py
```

## Project Structure

```
diogenes/
├── src/                 # Source code
│   └── diogenes/        # Main package
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── dataset_generator.py # SFT/DPO data generation
│       ├── eval_metrics.py     # Core reliability metrics
│       ├── inference.py        # Inference engine
│       ├── model.py            # Model loading/wrapping
│       ├── pass1_protection.py # Regression detection
│       ├── train_sft.py        # SFT training script
│       └── train_dpo.py        # DPO training script
├── configs/             # Configuration files
│   ├── config.yaml
│   └── remote_config.yaml
├── datasets/            # Training/evaluation datasets
├── models/              # Model checkpoints
├── scripts/             # Utility scripts
│   ├── download_model.py
│   ├── download_gguf.py
│   ├── test_inference.py
│   ├── setup_env.py
│   └── prepare_remote_machine.py
├── tests/               # Test suite
├── docs/                # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── PASS1_GUARDRAILS.md
│   └── phase0_quickstart.md
├── phasen/              # Phase documentation (DE)
│   ├── phase_0.md through phase_7.md
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

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

## Training Pipeline

### Implemented Components

1. **Dataset Generation** ✓
   - SFT Dataset: 80k samples covering 7 epistemic modes
   - DPO Dataset: 60k preference pairs with hallucination penalty
   - JSON schema with reasoning traces, risk levels, confidence targets

2. **Training Scripts** ✓
   - `train_sft.py`: LoRA/QLoRA (4-bit) with rank 32, alpha 64
   - `train_dpo.py`: Direct Preference Optimization with β=0.1-0.3
   - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

3. **Evaluation & Protection** ✓
   - Pass@1 regression detection
   - Core Reliability Metrics (Tier 1)
   - Special Metrics for math/code (Tier 2, monitoring only)
   - DPO audit for prompt interference

### Training Phases (Planned)

1. **Phase 1 - SFT**: Supervised Fine-Tuning (~4 hours on H100)
2. **Phase 2 - DPO**: Direct Preference Optimization (~6 hours)
3. **Phase 3 - Calibration**: Temperature scaling for confidence
4. **Phase 4 - Evaluation**: Full benchmark suite
5. **Phase 5 - Red Teaming**: Adversarial testing

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

### Traditional Benchmarks

- **TruthfulQA**: Target +8–15%
- **HaluEval**: Target –20–30% hallucinations
- **WildBench**: Real-world performance
- **GPQA**: Expert-level knowledge
- **LiveBench**: Current capabilities

## Expected Results

| Metric | Target Improvement |
|--------|-------------------|
| TruthfulQA | +8–15% |
| HaluEval | –20–30% hallucinations |
| ECE | –40% |
| Abstention AUROC | +15% |

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Citation

```bibtex
@software{diogenes2026,
  title = {Diogenes: The Reliable 32B},
  year = {2026},
  description = {Epistemically optimized language model based on Qwen3-32B}
}
```

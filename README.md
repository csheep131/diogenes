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
├── configs/             # Configuration files
├── datasets/            # Training/evaluation datasets
├── models/              # Model checkpoints
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── docs/                # Documentation
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

1. **Phase 1 - SFT**: Supervised Fine-Tuning (~4 hours on H100)
2. **Phase 2 - DPO**: Direct Preference Optimization (~6 hours)
3. **Phase 3 - Calibration**: Temperature scaling for confidence
4. **Phase 4 - Evaluation**: Full benchmark suite
5. **Phase 5 - Red Teaming**: Adversarial testing

## Evaluation Metrics

### Primary Benchmarks
- TruthfulQA
- HaluEval
- WildBench

### Secondary Benchmarks
- GPQA
- LiveBench

### Custom Metrics
- Epistemic Gap Evaluation
- Mode Confusion Matrix
- Utility Score
- Expected Calibration Error (ECE)

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

# Scientific Papers Used in Diogenes

This directory contains summaries and notes on scientific papers from arXiv.org that influenced the Diogenes project.

---

## Core References

### 1. Pass@k Optimization & Pass@1 Degradation

**arXiv:2602.21189**

**Title:** Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training

**Relevance to Diogenes:**
- Foundation for Pass@1 Protection mechanisms
- Explains gradient conflicts when optimizing for Pass@k
- Informs two-tier evaluation system (Tier 1: Core Reliability, Tier 2: Special Metrics)
- Critical for DPO design guardrails

**Key Findings:**
- Pass@k optimization can degrade Pass@1 performance
- Mechanism: Over-weighting difficult prompts causes "prompt interference"
- Difficult prompts dominate gradient updates, easier prompts are "interfered with"

**Implementation in Diogenes:**
- `src/diogenes/eval_metrics.py` – Two-tier evaluation
- `src/diogenes/pass1_protection.py` – Regression detection
- `docs/PASS1_GUARDRAILS.md` – Product guardrails
- DPO audit for prompt interference patterns

**Status:** ✅ Implemented

---

### 2. Semantic Tube Prediction / JEPA

**arXiv:2602.22617**

**Title:** Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA

**Relevance to Diogenes:**
- Research track for future representation learning experiments
- Potential application for epistemic uncertainty estimation
- Joint Embedding Predictive Architecture (JEPA) for semantic understanding

**Key Findings:**
- JEPA approach for representation learning
- Potential improvements in data efficiency
- Semantic prediction without token-level generation

**Implementation in Diogenes:**
- Status: Research track only (not yet implemented)
- Future consideration for Phase 7+ iterations
- Monitoring for updates on representation learning approaches

**Status:** 🔬 Research Track

---

## Additional References

### 3. Direct Preference Optimization (DPO)

**arXiv:2305.18290**

**Title:** Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Relevance to Diogenes:**
- Foundation for DPO training phase (Phase 3, Phase 7-B)
- Alternative to PPO/RLHF
- Enables stable preference learning without separate reward model

**Key Findings:**
- DPO transforms preference learning into classification problem
- More stable than PPO-based RLHF
- No separate reward model needed

**Implementation in Diogenes:**
- `src/diogenes/train_dpo.py` – DPO training script
- Hallucination penalty in DPO loss
- Preference ranking: Gold > Acceptable > Weak > Hallucination

**Status:** ✅ Implemented

---

### 4. QLoRA: Efficient Fine-Tuning

**arXiv:2305.14314**

**Title:** QLoRA: Efficient Finetuning of Quantized LLMs

**Relevance to Diogenes:**
- Enables training on consumer hardware (RTX 3050 8GB)
- 4-bit quantization for memory efficiency
- Foundation for local development workflow

**Key Findings:**
- 4-bit quantization with minimal quality loss
- Parameter-efficient fine-tuning with LoRA
- Enables 32B model training on single GPU

**Implementation in Diogenes:**
- All training scripts support QLoRA
- RTX 3050 development workflow (Phase 0-6)
- H100 production training (Phase 7)

**Status:** ✅ Implemented

---

### 5. TruthfulQA Benchmark

**arXiv:2109.07958**

**Title:** TruthfulQA: Measuring How Models Mimic Human Falsehoods

**Relevance to Diogenes:**
- Primary benchmark for epistemic reliability
- Measures model tendency to generate false statements
- Core metric for Diogenes evaluation (Phase 5, Phase 7-D)

**Key Findings:**
- LLMs can learn to generate false statements from training data
- Larger models not necessarily more truthful
- Need for explicit truthfulness training

**Implementation in Diogenes:**
- Primary evaluation benchmark
- Target: +8–15% improvement over baseline
- Included in Core Reliability Metrics

**Status:** ✅ Implemented

---

### 6. Hallucination Evaluation (HaluEval)

**arXiv:2305.11747**

**Title:** HaluEval: A Large-Scale Benchmark for Hallucination Evaluation in Large Language Models

**Relevance to Diogenes:**
- Primary benchmark for hallucination detection
- Measures model tendency to generate fabricated information
- Core metric for Diogenes evaluation (Phase 5, Phase 7-D)

**Key Findings:**
- LLMs frequently generate hallucinated content
- Need for systematic hallucination evaluation
- Multiple hallucination types (factual, logical, etc.)

**Implementation in Diogenes:**
- Primary evaluation benchmark
- Target: –20–30% hallucination reduction
- Included in Core Reliability Metrics

**Status:** ✅ Implemented

---

### 7. Calibration in LLMs

**arXiv:2306.10481**

**Title:** Confidence Calibration in Large Language Models

**Relevance to Diogenes:**
- Foundation for Phase 4 (Calibration)
- Expected Calibration Error (ECE) as primary metric
- Temperature scaling for confidence adjustment

**Key Findings:**
- LLMs are often overconfident
- Temperature scaling improves calibration
- ECE as standard calibration metric

**Implementation in Diogenes:**
- `src/diogenes/eval_metrics.py` – ECE computation
- Phase 4: Calibration & Confidence Mapping
- Target: –40% ECE reduction

**Status:** ✅ Implemented

---

## Summary Table

| arXiv ID | Title | Status | Implementation |
|----------|-------|--------|----------------|
| **2602.21189** | Pass@k Optimization & Pass@1 Degradation | ✅ | Pass@1 Protection |
| **2602.22617** | Semantic Tube Prediction / JEPA | 🔬 | Research Track |
| **2305.18290** | Direct Preference Optimization (DPO) | ✅ | DPO Training |
| **2305.14314** | QLoRA: Efficient Fine-Tuning | ✅ | QLoRA Support |
| **2109.07958** | TruthfulQA Benchmark | ✅ | Evaluation Suite |
| **2305.11747** | HaluEval Benchmark | ✅ | Evaluation Suite |
| **2306.10481** | Confidence Calibration in LLMs | ✅ | Phase 4 Calibration |

---

## Notes

- Papers are listed in order of importance to the Diogenes project
- Status: ✅ = Implemented, 🔬 = Research Track, ⏳ = Planned
- For detailed implementation notes, see inline documentation in source files
- Additional papers may be added as the project evolves

---

## Citation Format

When referencing these papers in Diogenes documentation:

```bibtex
@article{author2026title,
  title={Paper Title},
  author={Author Names},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

See individual paper summaries for specific BibTeX entries.

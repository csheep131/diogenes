# Paper Summary: arXiv:2602.21189

**Title:** Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training

**arXiv ID:** 2602.21189

**Year:** 2026

---

## Abstract

This paper identifies a critical issue in LLM post-training: optimizing for Pass@k (k>1) can degrade Pass@1 performance through a mechanism called "prompt interference." The authors demonstrate that when training data over-weights difficult prompts, gradient conflicts arise that harm performance on easier prompts.

---

## Key Findings

### 1. Pass@k vs. Pass@1 Trade-off

- **Observation:** Models optimized for Pass@k show degraded Pass@1 performance
- **Mechanism:** Gradient conflicts from difficult prompts dominate training
- **Impact:** Single-response reliability suffers when optimizing for multi-sampling success

### 2. Prompt Interference

**Definition:** When gradient updates from difficult prompts interfere with performance on easier prompts.

**Causes:**
- Over-representation of hard prompts in training data (>30%)
- Loss weighting that favors difficult samples
- Verbosity bias in preference data (longer answers preferred)

**Effects:**
- Pass@1 decreases while Pass@k increases
- Model becomes dependent on multi-sampling
- Single-response reliability degrades

### 3. Difficulty Bias

- Training data with >30% hard prompts shows significant Pass@1 degradation
- Balanced datasets (difficulty distribution) maintain Pass@1 stability
- Recommendation: Monitor difficulty distribution in DPO/SFT data

### 4. Verbosity Bias

- Preference for longer answers in DPO data correlates with Pass@1 degradation
- Length-normalized loss reduces verbosity bias
- Recommendation: Monitor chosen/rejected length ratio (<1.2)

---

## Methodology

### Evaluation Framework

1. **Two-Tier System:**
   - Tier 1: Core Reliability Metrics (Pass@1, ECE, Hallucination Rate)
   - Tier 2: Special Metrics (Pass@k for Math/Code only)

2. **Regression Detection:**
   - Monitor Pass@1 Δ and Pass@k Δ across checkpoints
   - Critical threshold: Pass@1 < –2% with Pass@k > +1%

3. **DPO Audit:**
   - Check difficulty distribution
   - Check verbosity bias
   - Check abstention representation

---

## Implementation in Diogenes

### 1. Two-Tier Evaluation (`src/diogenes/eval_metrics.py`)

```python
from diogenes import compute_core_reliability_metrics, compute_special_metrics

# Tier 1: Core Reliability (ALWAYS use for decisions)
core_metrics = compute_core_reliability_metrics(...)

# Tier 2: Special Metrics (Math/Code monitoring ONLY)
special_metrics = compute_special_metrics(...)
```

### 2. Pass@1 Protection (`src/diogenes/pass1_protection.py`)

```python
from diogenes import run_pass1_protection_test

result = run_pass1_protection_test(...)

if result.is_regression:
    print(f"⚠️  REGRESSION: {result.regression_severity}")
    print(f"Recommendation: {result.recommendation}")
```

### 3. DPO Audit (`src/diogenes/pass1_protection.py`)

```python
from diogenes import check_dpo_for_prompt_interference

audit = check_dpo_for_prompt_interference(dpo_pairs)

if audit["difficulty_bias"] or audit["verbosity_bias"]:
    print("❌ Critical bias detected - review data before training")
```

### 4. Decision Matrix

| Condition | Pass@1 Δ | Pass@k Δ | Action |
|-----------|----------|----------|--------|
| **Critical Regression** | < –2% | > +1% | ❌ DO NOT PROMOTE |
| **Warning** | < –1% | > +0.5% | ⚠️ Vorsichtig prüfen |
| **Improvement** | > +1% | Beliebig | ✓ Sicher |
| **Stable** | ±1% | Beliebig | ✓ Sicher |

---

## Guardrails (from docs/PASS1_GUARDRAILS.md)

### Core Principle

**Pass@1 is the ONLY metric for checkpoint promotion.**

Pass@k (k>1) may ONLY be used for monitoring in verifiable domains (Math, Code).

### Prohibited Uses of Pass@k

- ❌ Global reward optimization
- ❌ Checkpoint promotion decisions
- ❌ DPO loss weighting
- ❌ Early stopping criteria

### Required Monitoring

- ✅ Pass@1 tracking across all checkpoints
- ✅ Difficulty distribution in DPO data (<30% hard)
- ✅ Verbosity bias monitoring (<1.2 ratio)
- ✅ Regression testing before promotion

---

## BibTeX

```bibtex
@article{pass1_interference2026,
  title={Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training},
  author={Author Names},
  journal={arXiv preprint arXiv:2602.21189},
  year={2026},
  url={https://arxiv.org/abs/2602.21189}
}
```

---

## Related Diogenes Documentation

- `docs/PASS1_GUARDRAILS.md` – Product guardrails
- `src/diogenes/eval_metrics.py` – Implementation
- `src/diogenes/pass1_protection.py` – Regression detection
- `roadmap.md` – Pass@1 Protection section (Section 19)

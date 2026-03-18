# Pass@1 Protection Implementation Summary

**Date:** 2026-03-18  
**Based on:** arXiv:2602.21189 - "Why Pass@k Optimization Can Degrade Pass@1"

---

## Overview

This implementation adds comprehensive Pass@1 protection mechanisms to Diogenes, ensuring the model optimizes for **reliable single-response decisions** rather than multi-sampling success.

---

## Files Created

### 1. `src/diogenes/eval_metrics.py`

Core evaluation metrics module implementing the two-tier evaluation system:

**Tier 1: Core Reliability Metrics**
- `compute_pass_at_k()` - Pass@1 and Pass@k computation
- `compute_expected_calibration_error()` - ECE for calibration monitoring
- `compute_brier_score()` - Probability calibration metric
- `compute_abstention_auroc()` - Abstention quality measurement
- `compute_hallucination_rate()` - Hallucination detection
- `compute_core_reliability_metrics()` - Full Tier 1 metrics suite
- `compute_special_metrics()` - Tier 2 metrics (math, code, retrieval only)
- `run_regression_test()` - Pass@k vs Pass@1 trade-off detection

**Data Classes:**
- `CoreReliabilityMetrics` - Tier 1 metrics container
- `SpecialMetrics` - Tier 2 metrics container
- `RegressionTestResult` - Regression test results with recommendations

### 2. `src/diogenes/pass1_protection.py`

Regression testing and DPO audit tools:

**Classes:**
- `Pass1RegressionTracker` - Tracks metrics across checkpoints, detects regression patterns

**Functions:**
- `run_pass1_protection_test()` - Main entry point for checkpoint evaluation
- `check_dpo_for_prompt_interference()` - Audits DPO data for bias patterns

### 3. `tests/test_pass1_protection.py`

Comprehensive test suite covering:
- Pass@k computation tests
- Calibration metrics tests
- Abstention AUROC tests
- Regression detection tests
- Core reliability metrics tests
- Special metrics tests
- DPO audit tests
- Regression tracker tests

### 4. `docs/PASS1_GUARDRAILS.md`

Product guardrails documentation including:
- Core principle statement
- Evaluation tier separation
- Regression testing requirements
- DPO design guardrails
- Training configuration rules
- Domain-specific policies
- Monitoring and alerting guidelines
- Enforcement mechanisms

### 5. `docs/IMPLEMENTATION_SUMMARY.md`

This file - implementation overview and usage guide.

---

## Key Features

### 1. Regression Detection

Automatically detects the critical regression pattern:

```python
from diogenes import run_regression_test

result = run_regression_test(
    current_pass_at_1=0.70,
    current_pass_at_k=0.95,
    baseline_pass_at_1=0.75,
    baseline_pass_at_k=0.90,
    k=5,
)

if result.is_regression:
    print(f"⚠️  {result.regression_severity}: {result.regression_details}")
    print(f"Recommendation: {result.recommendation}")
```

### 2. Checkpoint Tracking

Tracks metrics across training checkpoints:

```python
from diogenes import Pass1RegressionTracker, CoreReliabilityMetrics

tracker = Pass1RegressionTracker(checkpoint_dir="./checkpoints")

core_metrics = CoreReliabilityMetrics(
    pass_at_1=0.75,
    hallucination_rate=0.05,
    expected_calibration_error=0.08,
    n_samples=1000,
)

result = tracker.record_checkpoint(
    checkpoint_name="epoch_1",
    core_metrics=core_metrics,
    pass_at_k_math={5: 0.90, 10: 0.93},
)

print(f"Should promote: {result.should_promote}")
```

### 3. DPO Audit

Checks DPO training data for prompt interference patterns:

```python
from diogenes import check_dpo_for_prompt_interference

audit = check_dpo_for_prompt_interference(dpo_pairs)

if audit["concerns"]:
    print("⚠️  DPO data concerns:")
    for concern in audit["concerns"]:
        print(f"  - {concern}")
    print(f"\nRecommendation: {audit['recommendation']}")
```

### 4. Full Protection Test

Complete evaluation pipeline:

```python
from diogenes import run_pass1_protection_test

result = run_pass1_protection_test(
    predictions=predictions,
    ground_truth=ground_truth,
    confidences=confidences,
    epistemic_modes=epistemic_modes,
    gold_modes=gold_modes,
    is_knowable=is_knowable,
    needs_tool=needs_tool,
    tool_requests=tool_requests,
    false_premise_flags=false_premise_flags,
    predicted_false_premise=predicted_false_premise,
    baseline_pass_at_1=0.75,
    baseline_pass_at_k=0.90,
    math_predictions=math_preds,
    math_ground_truth=math_gt,
    k=5,
)

if result.should_promote:
    print("✓ Checkpoint safe to promote")
else:
    print("✗ Do not promote - regression detected")
```

---

## Integration Points

### Training Pipeline

Add to your training loop:

```python
# After each epoch/checkpoint
from diogenes import Pass1RegressionTracker

tracker = Pass1RegressionTracker()

# Evaluate on validation set
core_metrics = evaluate_core_reliability(model, val_data)
pass_at_k = evaluate_pass_at_k(model, math_val_data, k=5)

# Check for regression
result = tracker.record_checkpoint(
    checkpoint_name=f"epoch_{epoch}",
    core_metrics=core_metrics,
    pass_at_k_math={5: pass_at_k},
)

if not result.should_promote:
    print("Training regression detected - consider early stopping")
    break
```

### DPO Training

Add before DPO training:

```python
from diogenes import check_dpo_for_prompt_interference

# Load DPO dataset
dpo_pairs = load_dpo_dataset("path/to/dpo_data.jsonl")

# Audit for bias
audit = check_dpo_for_prompt_interference(dpo_pairs)

if audit["concerns"]:
    print("DPO data quality concerns:")
    for concern in audit["concerns"]:
        print(f"  - {concern}")
    
    # Optionally halt training
    if audit["difficulty_bias"] or audit["verbosity_bias"]:
        print("Critical bias detected - review data before training")
```

### Evaluation Pipeline

Add to your evaluation script:

```python
from diogenes import compute_core_reliability_metrics, compute_special_metrics

# Compute Tier 1 metrics (always)
core_metrics = compute_core_reliability_metrics(
    predictions=preds,
    ground_truth=gt,
    confidences=conf,
    epistemic_modes=modes,
    gold_modes=gold_modes,
    is_knowable=knowable,
    needs_tool=needs_tool,
    tool_requests=tool_req,
    false_premise_flags=fp_flags,
    predicted_false_premise=pred_fp,
)

print("Core Reliability Metrics:")
print(f"  Pass@1: {core_metrics.pass_at_1:.4f}")
print(f"  Hallucination Rate: {core_metrics.hallucination_rate:.4f}")
print(f"  ECE: {core_metrics.expected_calibration_error:.4f}")

# Compute Tier 2 metrics (verifiable domains only)
if is_math_domain or is_code_domain:
    special_metrics = compute_special_metrics(
        math_predictions=math_preds if is_math_domain else None,
        math_ground_truth=math_gt if is_math_domain else None,
        code_predictions=code_preds if is_code_domain else None,
        code_ground_truth=code_gt if is_code_domain else None,
        k_values=[1, 3, 5, 10],
    )
    print(f"Pass@5 (math): {special_metrics.pass_at_k_math.get(5, 'N/A')}")
```

---

## Decision Matrix

### Checkpoint Promotion

| Condition | Pass@1 Δ | Pass@k Δ | Action |
|-----------|----------|----------|--------|
| **Critical Regression** | < -2% | > +1% | ❌ DO NOT PROMOTE |
| **Warning** | < -1% | > +0.5% | ⚠️ Proceed with caution |
| **Improvement** | > +1% | Any | ✓ Safe to promote |
| **Stable** | ±1% | Any | ✓ Safe to promote |

### DPO Data Quality

| Metric | Threshold | Status |
|--------|-----------|--------|
| Difficulty Bias | < 30% hard | ✓ Pass |
| Verbosity Bias | < 1.2 ratio | ✓ Pass |
| Abstain Rep. | > 5% | ✓ Pass |

---

## Metrics Reference

### Tier 1: Core Reliability (Primary)

| Metric | Range | Target | Priority |
|--------|-------|--------|----------|
| Pass@1 | 0-1 | Maximize | **Primary** |
| Hallucination Rate | 0-1 | Minimize | High |
| ECE | 0-1 | < 0.05 | High |
| Mode Accuracy | 0-1 | Maximize | Medium |
| Abstention AUROC | 0-1 | Maximize | Medium |
| False Premise Detection | 0-1 | Maximize | Medium |

### Tier 2: Special Metrics (Secondary)

| Metric | Domain | Usage |
|--------|--------|-------|
| Pass@k | Math, Code | Monitoring only |
| Best-of-k | Tool-assisted | Special cases |

**Never use Tier 2 for:**
- Global reward optimization
- Checkpoint promotion
- DPO loss weighting

---

## Next Steps

### Immediate (Done ✓)

- [x] Eval metrics module with Pass@1/Pass@k separation
- [x] Core Reliability Metrics implementation
- [x] Special Metrics implementation
- [x] Regression test for Pass@k vs Pass@1
- [x] DPO audit tool
- [x] Product guardrails documentation

### Short-Term (Recommended)

1. **Integrate into training pipeline**
   - Add `Pass1RegressionTracker` to checkpoint callback
   - Configure automated regression testing

2. **Audit existing DPO data**
   - Run `check_dpo_for_prompt_interference()` on current datasets
   - Address any bias concerns before next training run

3. **Update evaluation scripts**
   - Replace existing metrics with `compute_core_reliability_metrics()`
   - Add regression testing to evaluation pipeline

### Medium-Term (Optional)

1. **Implement advanced hallucination detection**
   - Replace simple heuristic with NLI model
   - Add fact-checking API integration

2. **Add STP/JEPA research track**
   - Monitor arXiv:2602.22617 for updates
   - Consider representation learning experiments

3. **Domain-specific calibration**
   - Stricter ECE targets for medical/legal/finance
   - Domain-specific abstention thresholds

---

## Research References

1. **arXiv:2602.21189** - "Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training"
   - Key finding: Pass@k optimization can degrade Pass@1 via gradient conflicts
   - Mechanism: Over-weighting difficult prompts causes "prompt interference"

2. **arXiv:2602.22617** - "Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"
   - Status: Research track only
   - Potential future application: Representation learning

---

## Support

For questions or issues:
- Review `docs/PASS1_GUARDRAILS.md` for policy details
- Check `tests/test_pass1_protection.py` for usage examples
- Refer to inline documentation in `src/diogenes/eval_metrics.py`

# Diogenes Product Guardrails
## Pass@1 Protection Policy

**Version:** 1.0  
**Effective Date:** 2026-03-18  
**Based on:** arXiv:2602.21189 - "Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training"

---

## Core Principle

> **Diogenes optimiert den Kern nicht auf Mehrfach-Sampling-Erfolg, sondern auf epistemisch korrekte Single-Response-Entscheidungen.**
>
> *(Diogenes does not optimize the core for multi-sampling success, but for epistemically correct single-response decisions.)*

**Not the most spectacular multi-sampling statistic, but the most reliable first response.**

---

## 1. Evaluation Tier Separation

### Tier 1: Core Reliability Metrics (PRIMARY)

These metrics are the **primary optimization target** for Diogenes. All model checkpoints must be evaluated against these metrics, and improvements in these metrics take precedence over Tier 2 metrics.

| Metric | Description | Target |
|--------|-------------|--------|
| **Pass@1** | First-response accuracy | Maximize |
| **Hallucination Rate** | Frequency of fabricated information | Minimize |
| **Expected Calibration Error (ECE)** | Confidence-accuracy alignment | Reduce by 40% |
| **Epistemic Mode Accuracy** | Correct mode selection | Maximize |
| **Abstention AUROC** | Ability to abstain on unknowable questions | Maximize |
| **False Premise Detection Rate** | Recognition of flawed questions | Maximize |
| **Tool Request F1** | Appropriate tool usage | Maximize |

### Tier 2: Optional Special Metrics (SECONDARY)

These metrics should **ONLY** be computed for verifiable domains:
- Mathematics (with verifiable solutions)
- Code generation (with testable outputs)
- Retrieval tasks (with ground truth verification)
- Tool-assisted control paths

| Metric | Domain | Usage |
|--------|--------|-------|
| **Pass@k** | Math, Code, Retrieval | Monitoring only |
| **Best-of-k** | Tool-assisted paths | Special cases only |
| **Sampling Efficiency** | Verifiable domains | Diagnostic only |

**Never use Tier 2 metrics for:**
- Global reward optimization
- Checkpoint promotion decisions
- DPO loss weighting
- Model selection for deployment

---

## 2. Regression Testing Requirements

### Mandatory Regression Test

Before promoting any checkpoint, the following regression test **MUST** be run:

```python
from diogenes.pass1_protection import run_regression_test

result = run_regression_test(
    current_pass_at_1=current_p1,
    current_pass_at_k=current_pk,
    baseline_pass_at_1=baseline_p1,
    baseline_pass_at_k=baseline_pk,
    k=5,
)
```

### Decision Matrix

| Pass@1 Delta | Pass@k Delta | Severity | Action |
|--------------|--------------|----------|--------|
| < -2% | > +1% | **CRITICAL** | DO NOT PROMOTE - Revert and investigate |
| < -1% | > +0.5% | **WARNING** | Proceed with caution, monitor closely |
| > +1% | Any | OK | Safe to promote |
| Stable (±1%) | Any | OK | Stable, safe to promote |

### Critical Regression Pattern

The following pattern indicates **prompt interference** from Pass@k optimization:

```
Pass@1 ↓ (decreasing)  AND  Pass@k ↑ (increasing)
```

If this pattern is detected:
1. **Stop** the promotion process
2. **Revert** to the previous baseline
3. **Review** DPO training data for:
   - Over-weighting of difficult prompts
   - Bias toward multi-sampling friendly examples
   - Preference for verbose over concise answers
4. **Adjust** training configuration before retrying

---

## 3. DPO Design Guardrails

### Prohibited DPO Constructions

**DO NOT construct DPO pairs that prefer:**

1. **"Plausibly verbose" over "correctly concise"**
   - Avoid rewarding length over accuracy
   - Monitor chosen/rejected length ratio (should be ~1.0)

2. **"Confident wrong" over "honest abstain"**
   - Never rank hallucinated answers above abstention
   - Ensure abstain mode is adequately represented (>5% of data)

3. **"Multi-sample friendly" over "single-shot correct"**
   - Avoid examples where multiple attempts would help
   - Focus on examples solvable in one attempt

### Required DPO Audits

Before training, run the DPO audit:

```python
from diogenes.pass1_protection import check_dpo_for_prompt_interference

audit = check_dpo_for_prompt_interference(dpo_pairs)

if audit["concerns"]:
    for concern in audit["concerns"]:
        print(f"⚠️  {concern}")
```

### Audit Thresholds

| Metric | Threshold | Action if Exceeded |
|--------|-----------|-------------------|
| **Difficulty Bias** | >30% hard examples | Rebalance dataset |
| **Verbosity Bias** | Length ratio >1.2 | Review preference criteria |
| **Abstain Underrepresentation** | <5% abstain | Add abstention examples |

---

## 4. Training Configuration Guardrails

### Loss Weighting

**DO NOT:**
- Weight loss by prompt difficulty
- Apply higher weights to examples where Pass@k > Pass@1
- Use Pass@k-based reward signals for core model training

**DO:**
- Weight loss by epistemic correctness
- Prioritize hallucination reduction
- Apply domain-specific weighting for critical domains (medical, legal, finance)

### Checkpoint Selection

**Selection Criteria (in order of priority):**

1. **Highest Pass@1** on validation set
2. **Lowest Hallucination Rate**
3. **Lowest ECE** (best calibration)
4. **Highest Mode Accuracy**

**Never select checkpoints based on:**
- Pass@k alone
- Best-of-k performance
- Multi-sampling metrics

---

## 5. Domain-Specific Rules

### Critical Domains (Medical, Legal, Finance, Manufacturing)

For these domains, **additional guardrails apply**:

| Rule | Description |
|------|-------------|
| **Safety > Engagement** | Prefer abstention over risky answers |
| **Calibration Priority** | ECE targets are 2x stricter |
| **Hallucination Penalty** | 5x loss weight for hallucinations |
| **Audit Frequency** | Weekly DPO audits required |

### Verifiable Domains (Math, Code, Retrieval)

For these domains, **Tier 2 metrics are permitted**:

| Allowed Usage | Restrictions |
|---------------|--------------|
| Pass@k monitoring | Never for global optimization |
| Multi-sampling experiments | Isolated to domain-specific heads |
| Best-of-k evaluation | Only with verifier present |

---

## 6. Monitoring & Alerting

### Automated Checks

The following checks run automatically at each checkpoint:

1. **Pass@1 Regression Test** - Blocks promotion on critical regression
2. **DPO Audit** - Warns on bias detection
3. **Calibration Drift** - Alerts if ECE increases >10%
4. **Hallucination Spike** - Alerts if rate increases >5%

### Manual Review Triggers

Manual review is **required** when:

- Pass@1 decreases by >1% from baseline
- Abstention rate drops by >20%
- Hallucination rate increases by >10%
- Any critical regression test failure

---

## 7. Documentation Requirements

### Checkpoint Release Notes

Each checkpoint release must include:

```markdown
## Checkpoint: [name]
## Date: [date]

### Core Reliability Metrics
- Pass@1: [value] (Δ [change from baseline])
- Hallucination Rate: [value] (Δ [change])
- ECE: [value] (Δ [change])
- Mode Accuracy: [value] (Δ [change])

### Regression Test
- Pass@k (k=5): [value] (Δ [change])
- Result: [PASS/WARNING/CRITICAL]
- Recommendation: [should_promote/not recommended]

### DPO Audit
- Difficulty Bias: [pass/fail]
- Verbosity Bias: [pass/fail]
- Abstain Representation: [pass/fail]

### Known Issues
[list any concerns or limitations]
```

---

## 8. Enforcement

### Automated Enforcement

The following protections are **enforced by code**:

- Regression test must pass before checkpoint save
- DPO audit must complete before training start
- Core metrics must be logged for every evaluation

### Manual Enforcement

The following require **human approval**:

- Overriding a critical regression test
- Deploying a checkpoint with warnings
- Changing guardrail thresholds

---

## Appendix A: Implementation Reference

### Key Files

| File | Purpose |
|------|---------|
| `src/diogenes/eval_metrics.py` | Core and special metrics computation |
| `src/diogenes/pass1_protection.py` | Regression testing and DPO audit |
| `tests/test_pass1_protection.py` | Test suite for protection mechanisms |

### Quick Start

```python
# Run full Pass@1 protection test
from diogenes.pass1_protection import run_pass1_protection_test

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
    baseline_pass_at_1=baseline_p1,
    baseline_pass_at_k=baseline_pk,
)

if not result.should_promote:
    print(f"❌ {result.recommendation}")
else:
    print(f"✓ {result.recommendation}")
```

---

## Appendix B: Research References

### Primary Reference

- **arXiv:2602.21189** - "Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training"
  - Key finding: Pass@k optimization can degrade Pass@1 via gradient conflicts
  - Mechanism: Over-weighting difficult prompts causes "prompt interference"
  - Recommendation: Separate optimization targets for single vs. multi-sample

### Secondary Reference

- **arXiv:2602.22617** - "Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"
  - Status: Research track only (not part of core pipeline)
  - Potential future application: Representation learning for epistemic states

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-18 | Initial release implementing Pass@1 protection |

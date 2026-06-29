# Cascade Evaluation Harness

> **This harness operationalizes the evaluation protocol from the position paper. It is instrumentation, not a result.** The default backend is a deterministic mock that instantiates the paper's propagation model; numbers from a mock run describe the model's parameters, not real LLM behavior. Real-model runs require the optional API backend and are the user's responsibility to report honestly.

## What this does

This harness implements a minimal evaluation framework for studying how false claims propagate through multi-agent cascades. It provides:

- **20 seeded false claims** (`claims.yaml`) across multiple domains
- **Two cascade topologies**: sequential refinement and shared recursive context
- **Quarantine mechanism**: low-confidence or non-assertive responses are excluded from shared context
- **Metrics**: propagation depth, correction rate, false consensus rate, final error, calibration gap
- **Deterministic mock agent** parameterized by the paper's ρ (correlation) and v (verification gain)

## Installation

No additional dependencies beyond the project's `pyproject.toml` are required for the default mock run. The mock backend uses only `pyyaml` (already a project dependency).

```bash
# From the project root:
cd experiments/cascade_eval
```

## Quick Start (Mock Backend)

```bash
python run_experiment.py --backend mock --seed 0 --rho 0.6 --v 0.3 --tau 0.5 --rounds 4
```

This runs the 2×2 configuration grid:
- **Topologies**: `sequential_refinement`, `shared_recursive_context`
- **Quarantine**: off, on

Output:
- `results/cascade_eval.csv` — one row per configuration cell
- Console table with aggregated metrics

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `mock` | Agent backend: `mock` (deterministic) or `api` (real LLM) |
| `--seed` | `0` | Random seed for reproducibility |
| `--rho` | `0.6` | Correlation parameter (peer influence strength, Section 4 model) |
| `--v` | `0.3` | Verification gain (probability of demoting on review; must be > 0) |
| `--tau` | `0.5` | Confidence threshold for quarantine |
| `--rounds` | `4` | Number of rounds for `shared_recursive_context` topology |
| `--out` | `results/cascade_eval.csv` | Output CSV path |

## Plugging in a Real Model (API Backend)

To evaluate a real LLM, set environment variables and use `--backend api`:

```bash
export DIOGENES_MODEL="gpt-4"
export DIOGENES_API_KEY="sk-..."
export DIOGENES_API_BASE="https://api.openai.com/v1"  # optional

python run_experiment.py --backend api --seed 42 --out results/api_run.csv
```

The API backend sends the claim and context to the model and parses its JSON response. **Reporting results from real-model runs is the user's responsibility.**

## How the Mock Agent Works

The `MockAgent` implements the paper's propagation model (Section 4):

1. **Base error roll**: probability `base_error_prob` (default 0.5) of asserting the false claim
2. **Peer influence (ρ)**: if peers assert, agent may copy with probability `ρ × peer_assertion_fraction`
3. **Verification (v)**: if asserting, agent may demote with probability `v`

This makes the mock a faithful, runnable instance of the model `G = α·ρ/v`, not a black box.

## Topologies

### Sequential Refinement
Agent 1 answers → Agent 2 reviews → Agent 3 finalizes. Each agent sees only the prior agent's output (subject to quarantine).

### Shared Recursive Context
All agents read a shared, repeatedly-updated history over multiple rounds. This is the reuse structure expected to amplify correlation effects.

## Quarantine Rule

When `quarantine=True`, responses with confidence below `τ` or a non-assertive mode (`ABSTAIN`, `REQUEST_TOOL`, `REJECT_PREMISE`) are excluded from the shared context as asserted facts. They may be retained as tagged hypotheses but cannot be counted as peer assertions by downstream agents.

## Metrics

- **Propagation depth**: Number of positions the false claim survives as an asserted fact before suppression (0 if never asserted; max depth if never suppressed).
- **Correction rate**: Fraction of agents that demoted the claim after seeing it asserted by a peer.
- **False consensus rate**: Fraction of agents asserting the claim as fact at the final round.
- **Final error**: 1 if the final/aggregated answer asserts the false claim, else 0.
- **Calibration gap**: Mean confidence on wrong assertions (optional; 0.0 if no wrong assertions).

## Running Tests

```bash
# From the project root:
pytest tests/test_cascade_eval.py -v
```

## File Structure

```
experiments/cascade_eval/
├── claims.yaml          # 20 seeded false claims
├── agents.py            # Agent interface + MockAgent + APIAgent
├── topologies.py        # sequential_refinement, shared_recursive_context
├── metrics.py           # propagation_depth, correction_rate, etc.
├── run_experiment.py    # CLI entry point
├── README.md            # This file
└── results/             # Gitkept; run output lands here
    └── .gitkeep
```

## Scope

This is **evaluation only**. No training code, no DPO/SFT. No claims of benchmark results. No network calls in the default path.

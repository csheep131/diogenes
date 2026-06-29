"""CLI entry point for the cascade evaluation harness.

Runs the 2x2 grid: {sequential_refinement, shared_recursive_context}
x {quarantine off, on}, aggregates metrics over all claims, and writes
results to a CSV file plus a console table.

Usage:
    python run_experiment.py --backend mock --seed 0 \\
        --rho 0.6 --v 0.3 --tau 0.5 --rounds 4 \\
        --out results/cascade_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Ensure the experiment directory is on the path for imports.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENT_DIR))

from agents import Agent, APIAgent, MockAgent, load_claims
from metrics import aggregate_metrics, compute_claim_metrics
from topologies import sequential_refinement, shared_recursive_context


def _build_agents(
    backend: str,
    n_agents: int,
    seed: int,
    rho: float,
    v: float,
) -> list[Agent]:
    """Create a list of agents for a topology run."""
    if backend == "mock":
        rng = random.Random(seed)
        return [MockAgent(rho=rho, v=v, rng=rng) for _ in range(n_agents)]
    elif backend == "api":
        return [APIAgent() for _ in range(n_agents)]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_experiment(args: argparse.Namespace) -> None:
    """Execute the 2x2 grid and write results."""
    claims_path = _EXPERIMENT_DIR / "claims.yaml"
    claims = load_claims(str(claims_path))

    topologies = {
        "sequential_refinement": lambda agents, claim, q, tau: sequential_refinement(
            agents, claim, quarantine=q, tau=tau
        ),
        "shared_recursive_context": lambda agents, claim, q, tau: shared_recursive_context(
            agents, claim, rounds=args.rounds, quarantine=q, tau=tau
        ),
    }

    # Number of agents: 3 for sequential, args.rounds for shared (each round
    # uses the same agents, so we need at least 3).
    n_agents = max(3, args.rounds)

    results_rows: list[dict] = []

    for topo_name, topo_fn in topologies.items():
        for quarantine in [False, True]:
            # Fresh agents per configuration cell for reproducibility.
            agents = _build_agents(
                args.backend, n_agents, args.seed, args.rho, args.v
            )

            claim_metrics = []
            for claim in claims:
                rounds = topo_fn(agents, claim, quarantine, args.tau)
                cm = compute_claim_metrics(claim["id"], rounds)
                claim_metrics.append(cm)

            agg = aggregate_metrics(
                claim_metrics=claim_metrics,
                topology=topo_name,
                quarantine=quarantine,
                rho=args.rho,
                v=args.v,
                tau=args.tau,
                rounds=args.rounds,
                seed=args.seed,
                backend=args.backend,
            )

            row = {
                "topology": agg.topology,
                "quarantine": agg.quarantine,
                "prop_depth": agg.mean_prop_depth,
                "correction_rate": agg.mean_correction_rate,
                "false_consensus_rate": agg.mean_false_consensus_rate,
                "final_error": agg.mean_final_error,
                "calibration_gap": agg.mean_calibration_gap,
                "rho": agg.rho,
                "v": agg.v,
                "tau": agg.tau,
                "rounds": agg.rounds,
                "seed": agg.seed,
                "backend": agg.backend,
            }
            results_rows.append(row)

    # Write CSV.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    # Print console table.
    _print_table(results_rows)
    print(f"\nResults written to {out_path}")


def _print_table(rows: list[dict]) -> None:
    """Print a compact console table matching the paper's Table schema."""
    header = (
        f"{'Topology':<28} {'Quarantine':<10} {'Prop. depth':>11} "
        f"{'Correction':>10} {'False cons.':>11} {'Final err.':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        q_str = "on" if row["quarantine"] else "off"
        print(
            f"{row['topology']:<28} {q_str:<10} {row['prop_depth']:>11.2f} "
            f"{row['correction_rate']:>10.3f} {row['false_consensus_rate']:>11.3f} "
            f"{row['final_error']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cascade evaluation harness for the Diogenes position paper."
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "api"],
        default="mock",
        help="Agent backend: 'mock' (deterministic, default) or 'api' (requires env vars).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.6,
        help="Correlation parameter (peer influence strength).",
    )
    parser.add_argument(
        "--v",
        type=float,
        default=0.3,
        help="Verification gain (probability of demoting on review). Must be > 0.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Confidence threshold for quarantine.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of rounds for shared_recursive_context topology.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/cascade_eval.csv",
        help="Output CSV path (relative to experiment dir or absolute).",
    )
    args = parser.parse_args()

    if args.v <= 0:
        parser.error("--v must be > 0.")

    # Resolve output path relative to experiment dir if not absolute.
    if not os.path.isabs(args.out):
        args.out = str(_EXPERIMENT_DIR / args.out)

    run_experiment(args)


if __name__ == "__main__":
    main()

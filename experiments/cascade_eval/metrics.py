"""Metrics for the cascade evaluation harness.

Given an interaction trace (list of Rounds) for one claim, computes:
- propagation_depth: how far the false claim survives as an asserted fact.
- correction_rate: fraction of agents that demoted the claim after seeing it.
- false_consensus_rate: fraction of agents asserting at the final round.
- final_error: 1 if the final answer asserts the false claim, else 0.
- confidence_calibration: mean confidence on wrong assertions (optional).
"""

from __future__ import annotations

from dataclasses import dataclass

from topologies import Round


@dataclass
class ClaimMetrics:
    """Metrics for a single claim under one configuration."""

    claim_id: str
    propagation_depth: int
    correction_rate: float
    false_consensus_rate: float
    final_error: int
    calibration_gap: float  # mean confidence on wrong assertions; 0.0 if none


def compute_claim_metrics(
    claim_id: str,
    rounds: list[Round],
) -> ClaimMetrics:
    """Compute metrics for a single claim from its interaction trace.

    Args:
        claim_id: The claim identifier.
        rounds: List of Rounds from a topology run.

    Returns:
        ClaimMetrics for this claim.
    """
    # Flatten all responses in order.
    all_responses = []
    for r in rounds:
        all_responses.extend(r.responses)

    if not all_responses:
        return ClaimMetrics(
            claim_id=claim_id,
            propagation_depth=0,
            correction_rate=0.0,
            false_consensus_rate=0.0,
            final_error=0,
            calibration_gap=0.0,
        )

    total_agents = len(all_responses)

    # propagation_depth: number of positions the false claim survives as
    # an asserted fact before suppression. Count consecutive assertions
    # from the start, or the last position where it was asserted.
    last_assert_pos = -1
    for i, resp in enumerate(all_responses):
        if resp.asserts_claim:
            last_assert_pos = i
    propagation_depth = last_assert_pos + 1 if last_assert_pos >= 0 else 0

    # correction_rate: fraction of agents that demoted the claim.
    # An agent "demotes" if it does NOT assert, given that at least one
    # prior agent asserted. If no prior agent asserted, correction is N/A
    # (we count it as 0 to avoid division issues).
    corrections = 0
    for i, resp in enumerate(all_responses):
        if not resp.asserts_claim and i > 0:
            # Check if any prior agent asserted.
            any_prior_asserted = any(
                all_responses[j].asserts_claim for j in range(i)
            )
            if any_prior_asserted:
                corrections += 1

    # Denominator: agents that had an opportunity to correct (i.e., at
    # least one prior assertion existed).
    correction_opportunities = 0
    for i in range(1, total_agents):
        if any(all_responses[j].asserts_claim for j in range(i)):
            correction_opportunities += 1
    correction_rate = corrections / correction_opportunities if correction_opportunities > 0 else 0.0

    # false_consensus_rate: fraction of agents asserting at the final round.
    final_round = rounds[-1]
    final_assertions = sum(1 for r in final_round.responses if r.asserts_claim)
    false_consensus_rate = (
        final_assertions / len(final_round.responses) if final_round.responses else 0.0
    )

    # final_error: 1 if the last response asserts the false claim.
    final_error = 1 if all_responses[-1].asserts_claim else 0

    # calibration_gap: mean confidence on wrong assertions.
    wrong_assertions = [
        r for r in all_responses if r.asserts_claim
    ]
    if wrong_assertions:
        calibration_gap = sum(r.confidence for r in wrong_assertions) / len(
            wrong_assertions
        )
    else:
        calibration_gap = 0.0

    return ClaimMetrics(
        claim_id=claim_id,
        propagation_depth=propagation_depth,
        correction_rate=correction_rate,
        false_consensus_rate=false_consensus_rate,
        final_error=final_error,
        calibration_gap=calibration_gap,
    )


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple claims."""

    topology: str
    quarantine: bool
    mean_prop_depth: float
    mean_correction_rate: float
    mean_false_consensus_rate: float
    mean_final_error: float
    mean_calibration_gap: float
    rho: float
    v: float
    tau: float
    rounds: int
    seed: int
    backend: str


def aggregate_metrics(
    claim_metrics: list[ClaimMetrics],
    topology: str,
    quarantine: bool,
    rho: float,
    v: float,
    tau: float,
    rounds: int,
    seed: int,
    backend: str,
) -> AggregateMetrics:
    """Aggregate per-claim metrics into a single row for the results CSV.

    Args:
        claim_metrics: List of ClaimMetrics, one per claim.
        topology: Topology name.
        quarantine: Whether quarantine was enabled.
        rho, v, tau, rounds, seed, backend: Configuration parameters.

    Returns:
        AggregateMetrics with means across all claims.
    """
    n = len(claim_metrics)
    if n == 0:
        return AggregateMetrics(
            topology=topology,
            quarantine=quarantine,
            mean_prop_depth=0.0,
            mean_correction_rate=0.0,
            mean_false_consensus_rate=0.0,
            mean_final_error=0.0,
            mean_calibration_gap=0.0,
            rho=rho,
            v=v,
            tau=tau,
            rounds=rounds,
            seed=seed,
            backend=backend,
        )

    return AggregateMetrics(
        topology=topology,
        quarantine=quarantine,
        mean_prop_depth=sum(m.propagation_depth for m in claim_metrics) / n,
        mean_correction_rate=sum(m.correction_rate for m in claim_metrics) / n,
        mean_false_consensus_rate=sum(m.false_consensus_rate for m in claim_metrics) / n,
        mean_final_error=sum(m.final_error for m in claim_metrics) / n,
        mean_calibration_gap=sum(m.calibration_gap for m in claim_metrics) / n,
        rho=rho,
        v=v,
        tau=tau,
        rounds=rounds,
        seed=seed,
        backend=backend,
    )

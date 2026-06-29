"""Unit tests for the cascade evaluation metrics module.

Tests use hand-built traces to verify metric computation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the experiment directory to path for imports.
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "cascade_eval"
sys.path.insert(0, str(_EXPERIMENT_DIR))

from agents import AgentResponse, EpistemicMode
from metrics import aggregate_metrics, compute_claim_metrics
from topologies import Round


def _make_response(
    asserts: bool = True,
    confidence: float = 0.8,
    mode: EpistemicMode = EpistemicMode.DIRECT,
    text: str = "test",
) -> AgentResponse:
    return AgentResponse(text=text, asserts_claim=asserts, confidence=confidence, mode=mode)


class TestComputeClaimMetrics:
    """Tests for compute_claim_metrics on hand-built traces."""

    def test_no_assertions_anywhere(self) -> None:
        """When no agent asserts, propagation_depth=0, final_error=0."""
        rounds = [
            Round(responses=[
                _make_response(asserts=False, confidence=0.2, mode=EpistemicMode.ABSTAIN),
                _make_response(asserts=False, confidence=0.1, mode=EpistemicMode.REJECT_PREMISE),
                _make_response(asserts=False, confidence=0.3, mode=EpistemicMode.REQUEST_TOOL),
            ])
        ]
        m = compute_claim_metrics("c01", rounds)
        assert m.propagation_depth == 0
        assert m.final_error == 0
        assert m.false_consensus_rate == 0.0
        assert m.correction_rate == 0.0
        assert m.calibration_gap == 0.0

    def test_all_assert_no_correction(self) -> None:
        """All agents assert: propagation_depth=max, final_error=1."""
        rounds = [
            Round(responses=[
                _make_response(asserts=True, confidence=0.9),
                _make_response(asserts=True, confidence=0.85),
                _make_response(asserts=True, confidence=0.8),
            ])
        ]
        m = compute_claim_metrics("c02", rounds)
        assert m.propagation_depth == 3
        assert m.final_error == 1
        assert m.false_consensus_rate == 1.0
        assert m.correction_rate == 0.0
        assert m.calibration_gap == pytest.approx((0.9 + 0.85 + 0.8) / 3)

    def test_assert_then_correct(self) -> None:
        """First asserts, second and third correct: depth=1."""
        rounds = [
            Round(responses=[
                _make_response(asserts=True, confidence=0.9),
                _make_response(asserts=False, confidence=0.2, mode=EpistemicMode.ABSTAIN),
                _make_response(asserts=False, confidence=0.1, mode=EpistemicMode.REJECT_PREMISE),
            ])
        ]
        m = compute_claim_metrics("c03", rounds)
        assert m.propagation_depth == 1
        assert m.final_error == 0
        # Final round = only round: 1 assertion out of 3 responses.
        assert m.false_consensus_rate == pytest.approx(1.0 / 3)
        # Agents 2 and 3 both corrected (2 opportunities, 2 corrections).
        assert m.correction_rate == 1.0

    def test_assert_correct_reassert(self) -> None:
        """Assert → correct → re-assert: depth=3, correction_rate partial."""
        rounds = [
            Round(responses=[
                _make_response(asserts=True, confidence=0.9),
                _make_response(asserts=False, confidence=0.2, mode=EpistemicMode.ABSTAIN),
                _make_response(asserts=True, confidence=0.7),
            ])
        ]
        m = compute_claim_metrics("c04", rounds)
        assert m.propagation_depth == 3  # last assertion at position 2
        assert m.final_error == 1
        # Final round = only round: 2 assertions (agent 1, 3) out of 3.
        assert m.false_consensus_rate == pytest.approx(2.0 / 3)
        # Agent 2 corrected; agent 3 had opportunity but re-asserted.
        # corrections=1, opportunities=2 (agents 2 and 3 both had prior assertions).
        assert m.correction_rate == pytest.approx(0.5)

    def test_multi_round_shared_context(self) -> None:
        """Multi-round trace: final round determines false_consensus_rate."""
        rounds = [
            Round(responses=[
                _make_response(asserts=True, confidence=0.9),
                _make_response(asserts=True, confidence=0.8),
            ]),
            Round(responses=[
                _make_response(asserts=False, confidence=0.3, mode=EpistemicMode.ABSTAIN),
                _make_response(asserts=False, confidence=0.1, mode=EpistemicMode.REJECT_PREMISE),
            ]),
        ]
        m = compute_claim_metrics("c05", rounds)
        assert m.propagation_depth == 2  # last assertion at position 1
        assert m.final_error == 0
        # Final round: 0 assertions out of 2 responses.
        assert m.false_consensus_rate == 0.0

    def test_empty_rounds(self) -> None:
        """Empty trace returns zeroed metrics."""
        m = compute_claim_metrics("c06", [])
        assert m.propagation_depth == 0
        assert m.final_error == 0
        assert m.correction_rate == 0.0


class TestAggregateMetrics:
    """Tests for aggregate_metrics."""

    def test_aggregate_basic(self) -> None:
        """Mean aggregation over two claims."""
        claims = [
            compute_claim_metrics("c01", [
                Round(responses=[_make_response(asserts=True, confidence=0.9)])
            ]),
            compute_claim_metrics("c02", [
                Round(responses=[_make_response(asserts=False, confidence=0.1, mode=EpistemicMode.ABSTAIN)])
            ]),
        ]
        agg = aggregate_metrics(
            claims, "sequential_refinement", False,
            rho=0.5, v=0.3, tau=0.5, rounds=3, seed=0, backend="mock"
        )
        assert agg.topology == "sequential_refinement"
        assert agg.quarantine is False
        assert agg.mean_final_error == pytest.approx(0.5)  # 1 + 0 / 2
        assert agg.mean_prop_depth == pytest.approx(0.5)  # 1 + 0 / 2

    def test_aggregate_empty(self) -> None:
        """Empty claim list returns zeroed aggregate."""
        agg = aggregate_metrics(
            [], "shared_recursive_context", True,
            rho=0.6, v=0.3, tau=0.5, rounds=4, seed=0, backend="mock"
        )
        assert agg.mean_final_error == 0.0
        assert agg.mean_prop_depth == 0.0

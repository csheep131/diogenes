"""Tests for Pass@1 Protection and Regression Detection.

These tests verify the Pass@1 protection mechanisms described in
arXiv:2602.21189, ensuring Diogenes optimizes for reliable single-response
decisions rather than multi-sampling success.
"""

import pytest
import numpy as np

from diogenes.eval_metrics import (
    compute_pass_at_k,
    compute_expected_calibration_error,
    compute_brier_score,
    compute_abstention_auroc,
    compute_hallucination_rate,
    compute_core_reliability_metrics,
    compute_special_metrics,
    run_regression_test,
    CoreReliabilityMetrics,
    SpecialMetrics,
    RegressionTestResult,
)
from diogenes.pass1_protection import (
    Pass1RegressionTracker,
    run_pass1_protection_test,
    check_dpo_for_prompt_interference,
)


class TestPassAtK:
    """Test Pass@k computation."""

    def test_pass_at_1_perfect(self):
        """Test Pass@1 with perfect accuracy."""
        correct = np.array([1, 1, 1, 1, 1])
        total = np.array([1, 1, 1, 1, 1])
        assert compute_pass_at_k(correct, total, k=1) == 1.0

    def test_pass_at_1_zero(self):
        """Test Pass@1 with zero accuracy."""
        correct = np.array([0, 0, 0, 0, 0])
        total = np.array([1, 1, 1, 1, 1])
        assert compute_pass_at_k(correct, total, k=1) == 0.0

    def test_pass_at_1_partial(self):
        """Test Pass@1 with partial accuracy."""
        correct = np.array([1, 0, 1, 0, 1])
        total = np.array([1, 1, 1, 1, 1])
        assert compute_pass_at_k(correct, total, k=1) == 0.6

    def test_pass_at_k_increases_with_k(self):
        """Test that Pass@k increases with k (for non-perfect accuracy)."""
        correct = np.array([1, 0, 1, 0, 1])
        total = np.array([5, 5, 5, 5, 5])  # 5 samples per problem
        
        pass_at_1 = compute_pass_at_k(correct, total, k=1)
        pass_at_3 = compute_pass_at_k(correct, total, k=3)
        pass_at_5 = compute_pass_at_k(correct, total, k=5)
        
        assert pass_at_1 <= pass_at_3 <= pass_at_5
        assert pass_at_5 == 1.0  # With 5 samples and 3 correct, Pass@5 should be 1.0

    def test_pass_at_k_invalid_input(self):
        """Test Pass@k with invalid input."""
        with pytest.raises(ValueError):
            compute_pass_at_k(np.array([1, 2]), np.array([1, 1]), k=1)  # correct > total
        
        with pytest.raises(ValueError):
            compute_pass_at_k(np.array([1]), np.array([1, 1]), k=1)  # length mismatch


class TestCalibrationMetrics:
    """Test calibration metrics."""

    def test_ece_perfect_calibration(self):
        """Test ECE with perfect calibration."""
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        accuracies = np.array([1, 1, 1, 0, 0])  # Matches confidence
        ece = compute_expected_calibration_error(confidences, accuracies, n_bins=5)
        assert ece < 0.1  # Should be very low

    def test_ece_poor_calibration(self):
        """Test ECE with poor calibration."""
        confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        accuracies = np.array([0, 0, 0, 0, 0])  # All wrong despite high confidence
        ece = compute_expected_calibration_error(confidences, accuracies, n_bins=5)
        assert ece > 0.5  # Should be high

    def test_brier_score_perfect(self):
        """Test Brier score with perfect predictions."""
        predicted_probs = np.array([1.0, 1.0, 0.0, 0.0])
        actual_outcomes = np.array([1, 1, 0, 0])
        assert compute_brier_score(predicted_probs, actual_outcomes) == 0.0

    def test_brier_score_worst(self):
        """Test Brier score with worst predictions."""
        predicted_probs = np.array([1.0, 1.0, 1.0, 1.0])
        actual_outcomes = np.array([0, 0, 0, 0])
        assert compute_brier_score(predicted_probs, actual_outcomes) == 1.0


class TestAbstentionAUROC:
    """Test abstention AUROC computation."""

    def test_abstention_auroc_perfect(self):
        """Test AUROC with perfect abstention."""
        confidences = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        is_knowable = np.array([1, 1, 1, 0, 0])  # High confidence = knowable
        auroc = compute_abstention_auroc(confidences, is_knowable)
        assert auroc == 1.0

    def test_abstention_auroc_random(self):
        """Test AUROC with random abstention."""
        np.random.seed(42)
        confidences = np.random.rand(100)
        is_knowable = np.random.randint(0, 2, 100)
        auroc = compute_abstention_auroc(confidences, is_knowable)
        assert 0.3 < auroc < 0.7  # Should be around 0.5

    def test_abstention_auroc_inverse(self):
        """Test AUROC with inverse abstention (wrong)."""
        confidences = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        is_knowable = np.array([1, 1, 1, 0, 0])  # Low confidence = knowable (wrong)
        auroc = compute_abstention_auroc(confidences, is_knowable)
        assert auroc < 0.5


class TestRegressionTest:
    """Test Pass@k vs Pass@1 regression detection."""

    def test_regression_critical(self):
        """Test critical regression detection."""
        result = run_regression_test(
            current_pass_at_1=0.70,
            current_pass_at_k=0.95,
            baseline_pass_at_1=0.75,
            baseline_pass_at_k=0.90,
            k=5,
        )
        assert result.is_regression is True
        assert result.regression_severity == "critical"
        assert result.should_promote is False

    def test_regression_warning(self):
        """Test warning-level regression detection."""
        result = run_regression_test(
            current_pass_at_1=0.74,
            current_pass_at_k=0.92,
            baseline_pass_at_1=0.75,
            baseline_pass_at_k=0.90,
            k=5,
        )
        assert result.is_regression is True
        assert result.regression_severity == "warning"

    def test_no_regression_improvement(self):
        """Test no regression when Pass@1 improves."""
        result = run_regression_test(
            current_pass_at_1=0.78,
            current_pass_at_k=0.92,
            baseline_pass_at_1=0.75,
            baseline_pass_at_k=0.90,
            k=5,
        )
        assert result.is_regression is False
        assert result.should_promote is True

    def test_no_regression_stable(self):
        """Test no regression when metrics are stable."""
        result = run_regression_test(
            current_pass_at_1=0.75,
            current_pass_at_k=0.90,
            baseline_pass_at_1=0.75,
            baseline_pass_at_k=0.90,
            k=5,
        )
        assert result.is_regression is False
        assert result.should_promote is True


class TestCoreReliabilityMetrics:
    """Test Core Reliability Metrics computation."""

    def test_core_metrics_computation(self):
        """Test full core metrics computation."""
        n = 100
        predictions = ["answer"] * n
        ground_truth = ["answer"] * 80 + ["other"] * 20
        confidences = [0.8] * n
        epistemic_modes = ["direct_answer"] * n
        gold_modes = ["direct_answer"] * n
        is_knowable = [True] * n
        needs_tool = [False] * n
        tool_requests = [False] * n
        false_premise_flags = [False] * n
        predicted_false_premise = [False] * n
        
        metrics = compute_core_reliability_metrics(
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
        )
        
        assert metrics.n_samples == 100
        assert metrics.pass_at_1 == 0.8
        assert metrics.mode_accuracy == 1.0
        assert 0 < metrics.pass_at_1_ci_lower < metrics.pass_at_1 < metrics.pass_at_1_ci_upper < 1


class TestSpecialMetrics:
    """Test Special Metrics computation."""

    def test_special_metrics_math(self):
        """Test special metrics for math domain."""
        math_predictions = ["42", "42", "43", "42", "42"]
        math_ground_truth = ["42", "42", "42", "42", "42"]
        
        metrics = compute_special_metrics(
            math_predictions=math_predictions,
            math_ground_truth=math_ground_truth,
            k_values=[1, 3, 5],
        )
        
        assert metrics.n_math_samples == 5
        assert metrics.math_accuracy == 0.8
        assert 1 in metrics.pass_at_k_math
        assert 3 in metrics.pass_at_k_math
        assert 5 in metrics.pass_at_k_math

    def test_special_metrics_empty(self):
        """Test special metrics with no data."""
        metrics = compute_special_metrics()
        
        assert metrics.n_math_samples == 0
        assert metrics.n_code_samples == 0
        assert metrics.n_retrieval_samples == 0
        assert metrics.pass_at_k_math == {}


class TestDPOAudit:
    """Test DPO audit for prompt interference."""

    def test_dpo_audit_healthy(self):
        """Test DPO audit with healthy data."""
        dpo_pairs = [
            {
                "question": "What is 2+2?",
                "chosen_answer": "4",
                "rejected_answer": "5",
                "gold_mode": "direct_answer",
            },
            {
                "question": "What is the capital of France?",
                "chosen_answer": "Paris",
                "rejected_answer": "London",
                "gold_mode": "direct_answer",
            },
        ] * 50  # 100 pairs total
        
        result = check_dpo_for_prompt_interference(dpo_pairs)
        
        assert result["status"] == "complete"
        assert result["n_pairs"] == 100
        assert result["difficulty_bias"] is False
        assert result["verbosity_bias"] is False

    def test_dpo_audit_difficulty_bias(self):
        """Test DPO audit with difficulty bias."""
        # Create pairs with very long, complex questions
        dpo_pairs = [
            {
                "question": "Given the complex interplay of multiple factors including X, Y, and Z, "
                           "and considering the nuanced implications for A, B, and C, "
                           "explain in detail why this is the case and compare it to alternative scenarios?",
                "chosen_answer": "Complex answer",
                "rejected_answer": "Simple answer",
                "gold_mode": "probabilistic",
            },
        ] * 100
        
        result = check_dpo_for_prompt_interference(dpo_pairs, threshold_difficulty_bias=0.2)
        
        assert result["difficulty_bias"] is True
        assert "Difficulty bias detected" in result["concerns"][0]

    def test_dpo_audit_abstain_underrepresentation(self):
        """Test DPO audit with underrepresented abstain mode."""
        # Create pairs with no abstain examples
        dpo_pairs = [
            {
                "question": "What is 2+2?",
                "chosen_answer": "4",
                "rejected_answer": "5",
                "gold_mode": "direct_answer",
            },
        ] * 100
        
        result = check_dpo_for_prompt_interference(dpo_pairs)
        
        assert result["abstain_underrepresentation"] is True
        assert "Abstain mode underrepresented" in result["concerns"][0]

    def test_dpo_audit_verbosity_bias(self):
        """Test DPO audit with verbosity bias."""
        # Create pairs where chosen is much longer than rejected
        dpo_pairs = [
            {
                "question": "What is 2+2?",
                "chosen_answer": "The answer is 4, and this is because when we add 2 and 2 together, "
                               "we get a sum of 4, which is a fundamental mathematical fact that "
                               "has been known since ancient times and is taught in schools worldwide.",
                "rejected_answer": "4",
                "gold_mode": "direct_answer",
            },
        ] * 100
        
        result = check_dpo_for_prompt_interference(dpo_pairs, threshold_multisample_bias=0.1)
        
        assert result["verbosity_bias"] is True
        assert result["avg_length_ratio"] > 1.2


class TestPass1RegressionTracker:
    """Test Pass@1 regression tracker."""

    def test_tracker_recording(self, tmp_path):
        """Test checkpoint recording and regression detection."""
        tracker = Pass1RegressionTracker(checkpoint_dir=str(tmp_path))
        
        # Record first checkpoint (baseline)
        core_metrics_1 = CoreReliabilityMetrics(
            pass_at_1=0.75,
            n_samples=100,
        )
        result_1 = tracker.record_checkpoint(
            checkpoint_name="checkpoint_1",
            core_metrics=core_metrics_1,
            pass_at_k_math={5: 0.90},
        )
        
        assert result_1.is_regression is False  # No baseline yet
        
        # Record second checkpoint with regression pattern
        core_metrics_2 = CoreReliabilityMetrics(
            pass_at_1=0.70,  # Decreased
            n_samples=100,
        )
        result_2 = tracker.record_checkpoint(
            checkpoint_name="checkpoint_2",
            core_metrics=core_metrics_2,
            pass_at_k_math={5: 0.95},  # Increased
        )
        
        assert result_2.is_regression is True
        assert result_2.regression_severity == "critical"
        assert result_2.should_promote is False

    def test_tracker_trend_analysis(self, tmp_path):
        """Test trend analysis over multiple checkpoints."""
        tracker = Pass1RegressionTracker(checkpoint_dir=str(tmp_path))
        
        # Record multiple checkpoints
        for i in range(5):
            core_metrics = CoreReliabilityMetrics(
                pass_at_1=0.75 - i * 0.01,  # Declining
                n_samples=100,
            )
            tracker.record_checkpoint(
                checkpoint_name=f"checkpoint_{i}",
                core_metrics=core_metrics,
                pass_at_k_math={5: 0.90 + i * 0.01},  # Improving
            )
        
        trend = tracker.get_trend_analysis(n_checkpoints=5)
        
        assert trend["n_checkpoints"] == 5
        assert trend["pass_at_1_trend"] < 0  # Negative trend
        assert trend["pass_at_k_trend"] > 0  # Positive trend
        assert len(trend["concerning_patterns"]) > 0

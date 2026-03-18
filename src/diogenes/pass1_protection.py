"""Regression Tests for Pass@1 Protection.

This module implements regression tests to detect the "prompt interference"
phenomenon described in arXiv:2602.21189, where optimizing for Pass@k
degrades Pass@1 performance.

Key Test:
    "Verbessert sich Pass@k, während Pass@1 fällt?"
    (Does Pass@k improve while Pass@1 falls?)

If this pattern is detected, the checkpoint should NOT be promoted.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from diogenes.eval_metrics import (
    compute_core_reliability_metrics,
    compute_special_metrics,
    run_regression_test,
    CoreReliabilityMetrics,
    RegressionTestResult,
)


logger = logging.getLogger(__name__)


class Pass1RegressionTracker:
    """Tracks Pass@1 vs Pass@k across training checkpoints.
    
    This tracker monitors for the regression pattern where Pass@k
    optimization degrades Pass@1 performance.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        metrics_file: str = "metrics_history.json",
        k_values: Optional[list[int]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_file = self.checkpoint_dir / metrics_file
        self.k_values = k_values or [1, 3, 5, 10]
        
        self.history: list[dict] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load metrics history from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                self.history = json.load(f)
            logger.info(f"Loaded metrics history with {len(self.history)} checkpoints")
    
    def _save_history(self) -> None:
        """Save metrics history to file."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def record_checkpoint(
        self,
        checkpoint_name: str,
        core_metrics: CoreReliabilityMetrics,
        pass_at_k_math: Optional[dict[int, float]] = None,
        pass_at_k_code: Optional[dict[int, float]] = None,
        timestamp: Optional[str] = None,
    ) -> RegressionTestResult:
        """Record a new checkpoint and run regression test.
        
        Args:
            checkpoint_name: Name/ID of the checkpoint
            core_metrics: Core reliability metrics for this checkpoint
            pass_at_k_math: Pass@k scores for math (optional)
            pass_at_k_code: Pass@k scores for code (optional)
            timestamp: ISO format timestamp (auto-generated if None)
            
        Returns:
            RegressionTestResult with promotion recommendation
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        # Get baseline (previous checkpoint)
        if self.history:
            baseline = self.history[-1]
            baseline_pass_at_1 = baseline.get("core_metrics", {}).get("pass_at_1", 0.0)
            baseline_pass_at_k = baseline.get("pass_at_k_math", {}).get("5", 0.0)
        else:
            # No baseline, use current as baseline (no regression possible)
            baseline_pass_at_1 = core_metrics.pass_at_1
            baseline_pass_at_k = pass_at_k_math.get(5, 0.0) if pass_at_k_math else 0.0
        
        # Get current Pass@k (use math domain as canonical)
        current_pass_at_k = pass_at_k_math.get(5, 0.0) if pass_at_k_math else 0.0
        
        # Run regression test
        regression_result = run_regression_test(
            current_pass_at_1=core_metrics.pass_at_1,
            current_pass_at_k=current_pass_at_k,
            baseline_pass_at_1=baseline_pass_at_1,
            baseline_pass_at_k=baseline_pass_at_k,
            k=5,
        )
        
        # Record checkpoint
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "timestamp": timestamp,
            "core_metrics": core_metrics.to_dict(),
            "pass_at_k_math": pass_at_k_math or {},
            "pass_at_k_code": pass_at_k_code or {},
            "regression_test": regression_result.to_dict(),
        }
        
        self.history.append(checkpoint_data)
        self._save_history()
        
        # Log result
        self._log_regression_result(checkpoint_name, regression_result)
        
        return regression_result
    
    def _log_regression_result(
        self,
        checkpoint_name: str,
        result: RegressionTestResult,
    ) -> None:
        """Log regression test result."""
        if result.regression_severity == "critical":
            logger.error(f"CHECKPOINT {checkpoint_name}: {result.regression_details}")
            logger.error(f"RECOMMENDATION: {result.recommendation}")
        elif result.regression_severity == "warning":
            logger.warning(f"CHECKPOINT {checkpoint_name}: {result.regression_details}")
            logger.warning(f"RECOMMENDATION: {result.recommendation}")
        else:
            logger.info(f"CHECKPOINT {checkpoint_name}: {result.regression_details}")
            logger.info(f"RECOMMENDATION: {result.recommendation}")
    
    def get_trend_analysis(self, n_checkpoints: int = 10) -> dict:
        """Analyze trends over recent checkpoints.
        
        Args:
            n_checkpoints: Number of recent checkpoints to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 checkpoints"}
        
        recent = self.history[-min(n_checkpoints, len(self.history)):]
        
        pass_at_1_values = [c["core_metrics"]["pass_at_1"] for c in recent]
        pass_at_k_values = [c.get("pass_at_k_math", {}).get("5", 0.0) for c in recent]
        
        # Compute trends
        pass_at_1_trend = np.polyfit(range(len(pass_at_1_values)), pass_at_1_values, 1)[0]
        pass_at_k_trend = np.polyfit(range(len(pass_at_k_values)), pass_at_k_values, 1)[0]
        
        # Detect concerning patterns
        concerning_patterns = []
        if pass_at_1_trend < -0.001 and pass_at_k_trend > 0.001:
            concerning_patterns.append(
                "Pass@1 declining while Pass@k rising - potential prompt interference"
            )
        
        return {
            "n_checkpoints": len(recent),
            "pass_at_1_trend": float(pass_at_1_trend),
            "pass_at_k_trend": float(pass_at_k_trend),
            "pass_at_1_avg": float(np.mean(pass_at_1_values)),
            "pass_at_k_avg": float(np.mean(pass_at_k_values)),
            "concerning_patterns": concerning_patterns,
            "recommendation": "Monitor closely" if concerning_patterns else "Trends look healthy",
        }


def run_pass1_protection_test(
    predictions: list[str],
    ground_truth: list[str],
    confidences: list[float],
    epistemic_modes: list[str],
    gold_modes: list[str],
    is_knowable: list[bool],
    needs_tool: list[bool],
    tool_requests: list[bool],
    false_premise_flags: list[bool],
    predicted_false_premise: list[bool],
    baseline_pass_at_1: float,
    baseline_pass_at_k: float,
    math_predictions: Optional[list[str]] = None,
    math_ground_truth: Optional[list[str]] = None,
    k: int = 5,
) -> RegressionTestResult:
    """Run complete Pass@1 protection test suite.
    
    This is the main entry point for evaluating a checkpoint against
    the Pass@1 protection criteria.
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth answers
        confidences: Model confidence scores
        epistemic_modes: Predicted epistemic modes
        gold_modes: Gold standard epistemic modes
        is_knowable: Whether each question is knowable
        needs_tool: Whether each question requires tools
        tool_requests: Whether model requested tools
        false_premise_flags: Whether question has false premise
        predicted_false_premise: Whether model detected false premise
        baseline_pass_at_1: Baseline Pass@1 from previous checkpoint
        baseline_pass_at_k: Baseline Pass@k from previous checkpoint
        math_predictions: Math-specific predictions (optional)
        math_ground_truth: Math-specific ground truth (optional)
        k: Value of k for Pass@k computation
        
    Returns:
        RegressionTestResult with promotion recommendation
    """
    # Compute Core Reliability Metrics
    core_metrics = compute_core_reliability_metrics(
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
    
    logger.info("Core Reliability Metrics:")
    logger.info(f"  Pass@1: {core_metrics.pass_at_1:.4f} [{core_metrics.pass_at_1_ci_lower:.4f}, {core_metrics.pass_at_1_ci_upper:.4f}]")
    logger.info(f"  Hallucination Rate: {core_metrics.hallucination_rate:.4f}")
    logger.info(f"  ECE: {core_metrics.expected_calibration_error:.4f}")
    logger.info(f"  Mode Accuracy: {core_metrics.mode_accuracy:.4f}")
    logger.info(f"  Abstention AUROC: {core_metrics.abstention_auroc:.4f}")
    
    # Compute Pass@k for math domain (if provided)
    pass_at_k = 0.0
    if math_predictions and math_ground_truth:
        from diogenes.eval_metrics import compute_pass_at_k
        
        n = len(math_predictions)
        correct = np.array([1 if p == g else 0 for p, g in zip(math_predictions, math_ground_truth)])
        total = np.ones(n, dtype=int)
        pass_at_k = compute_pass_at_k(correct, total, k)
        
        logger.info(f"Pass@{k} (math): {pass_at_k:.4f}")
    
    # Run regression test
    regression_result = run_regression_test(
        current_pass_at_1=core_metrics.pass_at_1,
        current_pass_at_k=pass_at_k,
        baseline_pass_at_1=baseline_pass_at_1,
        baseline_pass_at_k=baseline_pass_at_k,
        k=k,
    )
    
    logger.info(f"\nRegression Test Result:")
    logger.info(f"  Pass@1 Delta: {regression_result.pass_at_1_delta:+.4f}")
    logger.info(f"  Pass@{k} Delta: {regression_result.pass_at_k_delta:+.4f}")
    logger.info(f"  Is Regression: {regression_result.is_regression}")
    logger.info(f"  Severity: {regression_result.regression_severity}")
    logger.info(f"  Should Promote: {regression_result.should_promote}")
    
    return regression_result


def check_dpo_for_prompt_interference(
    dpo_pairs: list[dict],
    threshold_difficulty_bias: float = 0.3,
    threshold_multisample_bias: float = 0.2,
) -> dict:
    """Audit DPO training data for prompt interference patterns.
    
    Checks for the patterns described in arXiv:2602.21189 that can lead
    to Pass@k optimization degrading Pass@1:
    
    1. Over-weighting of difficult prompts
    2. Bias toward multi-sampling friendly examples
    3. Preference for "plausibly verbose" over "correctly concise"
    
    Args:
        dpo_pairs: List of DPO preference pairs
        threshold_difficulty_bias: Threshold for difficulty bias detection
        threshold_multisample_bias: Threshold for multi-sample bias detection
        
    Returns:
        Audit result dictionary
    """
    n_pairs = len(dpo_pairs)
    
    if n_pairs == 0:
        return {
            "status": "error",
            "message": "No DPO pairs provided",
        }
    
    # Analyze difficulty distribution
    difficulty_scores = []
    for pair in dpo_pairs:
        # Estimate difficulty from metadata or heuristics
        question = pair.get("question", "")
        
        # Heuristic: longer, more complex questions are harder
        difficulty = len(question.split()) / 50.0  # Normalize roughly to [0, 1]
        
        # Check for multi-hop indicators
        multi_hop_indicators = ["explain", "compare", "analyze", "evaluate", "why"]
        if any(ind in question.lower() for ind in multi_hop_indicators):
            difficulty += 0.2
        
        difficulty_scores.append(min(difficulty, 1.0))
    
    avg_difficulty = np.mean(difficulty_scores)
    
    # Check for difficulty bias (over-representation of hard examples)
    hard_ratio = np.mean([1 if d > 0.6 else 0 for d in difficulty_scores])
    difficulty_bias = hard_ratio > threshold_difficulty_bias
    
    # Analyze chosen vs rejected length ratio
    # DPO can learn to prefer verbose answers if not careful
    length_ratios = []
    for pair in dpo_pairs:
        chosen_len = len(pair.get("chosen_answer", "").split())
        rejected_len = len(pair.get("rejected_answer", "").split())
        
        if rejected_len > 0:
            length_ratios.append(chosen_len / rejected_len)
    
    avg_length_ratio = np.mean(length_ratios) if length_ratios else 1.0
    
    # Check for verbosity bias
    verbosity_bias = avg_length_ratio > (1 + threshold_multisample_bias)
    
    # Analyze epistemic mode distribution
    mode_counts = {}
    for pair in dpo_pairs:
        mode = pair.get("gold_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    # Check for under-representation of ABSTAIN mode
    abstain_ratio = mode_counts.get("abstain", 0) / n_pairs
    abstain_underrepresentation = abstain_ratio < 0.05  # Less than 5% abstain
    
    # Compile results
    concerns = []
    if difficulty_bias:
        concerns.append(
            f"Difficulty bias detected: {hard_ratio:.1%} of examples are 'hard'. "
            "This can cause gradient conflicts and prompt interference."
        )
    if verbosity_bias:
        concerns.append(
            f"Verbosity bias detected: chosen answers are {avg_length_ratio:.2f}x longer "
            "than rejected answers. This can incentivize multi-sampling behavior."
        )
    if abstain_underrepresentation:
        concerns.append(
            f"Abstain mode underrepresented: only {abstain_ratio:.1%} of examples. "
            "This can degrade the model's ability to recognize knowledge boundaries."
        )
    
    return {
        "status": "complete",
        "n_pairs": n_pairs,
        "avg_difficulty": float(avg_difficulty),
        "hard_ratio": float(hard_ratio),
        "difficulty_bias": difficulty_bias,
        "avg_length_ratio": float(avg_length_ratio),
        "verbosity_bias": verbosity_bias,
        "mode_distribution": mode_counts,
        "abstain_ratio": float(abstain_ratio),
        "abstain_underrepresentation": abstain_underrepresentation,
        "concerns": concerns,
        "recommendation": "Review DPO data construction" if concerns else "DPO data looks healthy",
    }

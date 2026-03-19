"""Evaluation Metrics for Diogenes - Pass@1 Protection & Core Reliability.

This module implements a two-tier evaluation system as per the Diogenes
research directive (arXiv:2602.21189):

Tier 1: Core Reliability Metrics (Primary optimization target)
  - Pass@1
  - Hallucination Rate
  - Expected Calibration Error (ECE)
  - Epistemic Mode Accuracy
  - False Premise Detection Rate
  - Tool Request Accuracy

Tier 2: Optional Special Metrics (Verifiable tasks only)
  - Pass@k for mathematics
  - Pass@k for code
  - Pass@k for retrieval + verifier tasks
  - Multi-sampling for tool-assisted control paths

Tier 3: SUA Metrics (Phase 3.5 - Staleness/Unknown/Ambiguity)
  - Staleness Detection Rate
  - Unknown Detection AUROC
  - Ambiguity Resolution Accuracy
  - Combined SUA Score

Key Principle: Diogenes optimizes for reliable single-response decisions,
not multi-sampling success.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class CoreReliabilityMetrics:
    """Tier 1: Core reliability metrics for Diogenes evaluation.
    
    These metrics are the primary optimization target and must be
    monitored to prevent Pass@k optimization from degrading Pass@1.
    """
    # Pass@1 - Primary metric
    pass_at_1: float = 0.0
    pass_at_1_ci_lower: float = 0.0  # 95% confidence interval
    pass_at_1_ci_upper: float = 0.0
    
    # Hallucination metrics
    hallucination_rate: float = 0.0
    hallucination_severity_avg: float = 0.0  # 1-5 scale
    
    # Calibration metrics
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    calibration_curve: list[tuple[float, float]] = field(default_factory=list)
    
    # Epistemic mode metrics
    mode_accuracy: float = 0.0
    mode_confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    abstention_auroc: float = 0.0  # Ability to abstain on unknowable questions
    
    # False premise detection
    false_premise_detection_rate: float = 0.0
    false_premise_false_positive_rate: float = 0.0
    
    # Tool request metrics
    tool_request_precision: float = 0.0
    tool_request_recall: float = 0.0
    tool_request_f1: float = 0.0
    
    # Metadata
    n_samples: int = 0
    evaluation_timestamp: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_1_ci_lower": self.pass_at_1_ci_lower,
            "pass_at_1_ci_upper": self.pass_at_1_ci_upper,
            "hallucination_rate": self.hallucination_rate,
            "hallucination_severity_avg": self.hallucination_severity_avg,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "mode_accuracy": self.mode_accuracy,
            "abstention_auroc": self.abstention_auroc,
            "false_premise_detection_rate": self.false_premise_detection_rate,
            "false_premise_false_positive_rate": self.false_premise_false_positive_rate,
            "tool_request_f1": self.tool_request_f1,
            "n_samples": self.n_samples,
        }


@dataclass
class SpecialMetrics:
    """Tier 2: Optional special metrics for verifiable tasks only.
    
    These metrics should ONLY be computed for:
    - Mathematics problems with verifiable solutions
    - Code generation with testable outputs
    - Retrieval tasks with ground truth verification
    - Tool-assisted control paths
    
    Never use these metrics for global reward optimization.
    """
    # Pass@k metrics (only for verifiable domains)
    pass_at_k_math: dict[int, float] = field(default_factory=dict)  # k -> Pass@k
    pass_at_k_code: dict[int, float] = field(default_factory=dict)
    pass_at_k_retrieval: dict[int, float] = field(default_factory=dict)
    
    # Multi-sampling metrics (only for tool-assisted paths)
    best_of_k_tool: dict[int, float] = field(default_factory=dict)
    sampling_efficiency: float = 0.0  # Improvement per additional sample
    
    # Domain-specific breakdowns
    math_accuracy: float = 0.0
    code_pass_rate: float = 0.0
    retrieval_precision: float = 0.0
    
    # Metadata
    n_math_samples: int = 0
    n_code_samples: int = 0
    n_retrieval_samples: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "pass_at_k_math": self.pass_at_k_math,
            "pass_at_k_code": self.pass_at_k_code,
            "pass_at_k_retrieval": self.pass_at_k_retrieval,
            "best_of_k_tool": self.best_of_k_tool,
            "sampling_efficiency": self.sampling_efficiency,
            "math_accuracy": self.math_accuracy,
            "code_pass_rate": self.code_pass_rate,
            "retrieval_precision": self.retrieval_precision,
        }


@dataclass
class SUAMetrics:
    """Tier 3: SUA (Staleness/Unknown/Ambiguity) Metrics for Phase 3.5.

    These metrics measure the model's ability to:
    - Recognize stale/outdated information (temporal knowledge boundaries)
    - Identify unknown/unknowable information (fundamental knowledge gaps)
    - Detect and handle ambiguous queries (clarification needs)

    These metrics are optimized during Phase 3.5 SUA Fine-Tuning.
    """
    # Staleness Detection Metrics
    staleness_detection_rate: float = 0.0  # % correctly identified as stale
    staleness_false_positive_rate: float = 0.0  # % falsely flagged as stale
    staleness_precision: float = 0.0
    staleness_f1: float = 0.0
    staleness_n_samples: int = 0

    # Unknown Detection Metrics
    unknown_detection_auroc: float = 0.5  # AUROC for unknowable questions
    unknown_precision_at_50_recall: float = 0.0
    unknown_recall_at_90_precision: float = 0.0
    unknown_n_samples: int = 0

    # Ambiguity Handling Metrics
    ambiguity_resolution_accuracy: float = 0.0  # % correctly resolved
    clarification_quality_score: float = 0.0  # Semantic similarity of clarifications
    clarification_rate: float = 0.0  # % of ambiguous queries that triggered clarification
    ambiguity_n_samples: int = 0

    # Combined SUA Score (weighted average)
    combined_sua_score: float = 0.0
    sua_weights: dict[str, float] = field(default_factory=lambda: {
        "staleness": 0.30,
        "unknown": 0.40,
        "ambiguity": 0.30
    })

    # Per-category breakdown
    staleness_by_category: dict[str, float] = field(default_factory=dict)
    unknown_by_category: dict[str, float] = field(default_factory=dict)
    ambiguity_by_category: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "staleness_detection_rate": self.staleness_detection_rate,
            "staleness_false_positive_rate": self.staleness_false_positive_rate,
            "staleness_precision": self.staleness_precision,
            "staleness_f1": self.staleness_f1,
            "unknown_detection_auroc": self.unknown_detection_auroc,
            "unknown_precision_at_50_recall": self.unknown_precision_at_50_recall,
            "ambiguity_resolution_accuracy": self.ambiguity_resolution_accuracy,
            "clarification_quality_score": self.clarification_quality_score,
            "combined_sua_score": self.combined_sua_score,
            "sua_weights": self.sua_weights,
            "staleness_by_category": self.staleness_by_category,
            "unknown_by_category": self.unknown_by_category,
            "ambiguity_by_category": self.ambiguity_by_category,
        }


@dataclass
class RegressionTestResult:
    """Result of Pass@k vs Pass@1 regression test.
    
    This test detects the "prompt interference" phenomenon described in
    arXiv:2602.21189, where Pass@k optimization degrades Pass@1.
    """
    # Current metrics
    current_pass_at_1: float = 0.0
    current_pass_at_k: float = 0.0
    
    # Baseline metrics (from previous checkpoint)
    baseline_pass_at_1: float = 0.0
    baseline_pass_at_k: float = 0.0
    
    # Change detection
    pass_at_1_delta: float = 0.0
    pass_at_k_delta: float = 0.0
    
    # Regression detection
    is_regression: bool = False
    regression_severity: str = "none"  # none, warning, critical
    regression_details: str = ""
    
    # Recommendation
    should_promote: bool = True
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "current_pass_at_1": self.current_pass_at_1,
            "current_pass_at_k": self.current_pass_at_k,
            "baseline_pass_at_1": self.baseline_pass_at_1,
            "baseline_pass_at_k": self.baseline_pass_at_k,
            "pass_at_1_delta": self.pass_at_1_delta,
            "pass_at_k_delta": self.pass_at_k_delta,
            "is_regression": self.is_regression,
            "regression_severity": self.regression_severity,
            "regression_details": self.regression_details,
            "should_promote": self.should_promote,
            "recommendation": self.recommendation,
        }


def compute_pass_at_k(
    correct: np.ndarray,
    total: np.ndarray,
    k: int = 1,
) -> float:
    """Compute Pass@k metric.
    
    Args:
        correct: Number of correct answers per problem
        total: Total number of samples per problem
        k: Number of samples to consider
        
    Returns:
        Pass@k score
        
    Note:
        For Diogenes, Pass@1 is the primary metric. Pass@k should only
        be computed for verifiable domains (math, code, retrieval).
    """
    if len(correct) != len(total):
        raise ValueError("correct and total arrays must have same length")
    
    if np.any(correct > total):
        raise ValueError("correct cannot be greater than total")
    
    if k < 1:
        raise ValueError("k must be at least 1")
    
    # Handle edge case where k >= total
    effective_k = np.minimum(k, total)
    
    # Compute Pass@k using the formula from Chen et al. (2021)
    # Pass@k = 1 - C(n-c, k) / C(n, k)
    # where n = total, c = correct
    
    def comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Compute binomial coefficient C(n, k) element-wise."""
        result = np.ones_like(n, dtype=np.float64)
        for i in range(len(k)):
            if k[i] > n[i]:
                result[i] = 0.0
            else:
                for j in range(int(k[i])):
                    result[i] = result[i] * (n[i] - j) / (j + 1)
        return result
    
    n = total
    c = correct
    
    # Avoid division by zero
    valid = n >= effective_k
    if not np.any(valid):
        return 0.0
    
    pass_at_k = np.zeros_like(n, dtype=np.float64)
    pass_at_k[valid] = 1.0 - comb(n[valid] - c[valid], effective_k[valid]) / comb(n[valid], effective_k[valid])
    
    return float(np.mean(pass_at_k))


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual
    accuracy, binned by confidence level.
    
    Args:
        confidences: Model confidence scores [0, 1]
        accuracies: Binary accuracy (1 = correct, 0 = incorrect)
        n_bins: Number of bins for calibration curve
        
    Returns:
        ECE score (lower is better)
    """
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have same length")
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.sum(in_bin) / total_samples
        
        if np.sum(in_bin) > 0:
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            avg_accuracy_in_bin = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier_score(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> float:
    """Compute Brier Score for probability calibration.
    
    Args:
        predicted_probs: Predicted probabilities [0, 1]
        actual_outcomes: Binary actual outcomes (1 = event occurred, 0 = didn't)
        
    Returns:
        Brier score (lower is better)
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("predicted_probs and actual_outcomes must have same length")
    
    return float(np.mean((predicted_probs - actual_outcomes) ** 2))


def compute_abstention_auroc(
    confidence_scores: np.ndarray,
    is_knowable: np.ndarray,
) -> float:
    """Compute AUROC for abstention decisions.
    
    Measures how well the model abstains on unknowable questions.
    
    Args:
        confidence_scores: Model confidence scores
        is_knowable: Binary indicator (1 = question is knowable, 0 = unknowable)
        
    Returns:
        AUROC score (higher is better)
    """
    # Simple AUROC computation
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidence_scores)
    sorted_labels = is_knowable[sorted_indices]
    
    n_pos = np.sum(is_knowable)
    n_neg = len(is_knowable) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Random baseline
    
    # Compute AUROC using trapezoidal rule
    tpr_prev, fpr_prev = 0.0, 0.0
    auroc = 0.0
    tp, fp = 0, 0
    
    for i, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        # Trapezoidal area
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        
        tpr_prev, fpr_prev = tpr, fpr
    
    return float(auroc)


def compute_hallucination_rate(
    responses: list[str],
    ground_truth: list[str],
    hallucination_detector: Optional[callable] = None,
) -> tuple[float, float]:
    """Compute hallucination rate and average severity.
    
    Args:
        responses: Model responses
        ground_truth: Ground truth answers
        hallucination_detector: Optional custom detector function
        
    Returns:
        Tuple of (hallucination_rate, avg_severity)
    """
    if len(responses) != len(ground_truth):
        raise ValueError("responses and ground_truth must have same length")
    
    hallucinations = 0
    severities = []
    
    for response, truth in zip(responses, ground_truth):
        # Simple heuristic: check for factual contradictions
        # In production, this should use a more sophisticated detector
        is_hallucination, severity = _detect_hallucination_simple(response, truth)
        
        if is_hallucination:
            hallucinations += 1
            severities.append(severity)
    
    rate = hallucinations / len(responses)
    avg_severity = np.mean(severities) if severities else 0.0
    
    return rate, avg_severity


def _detect_hallucination_simple(
    response: str,
    ground_truth: str,
) -> tuple[bool, float]:
    """Simple heuristic-based hallucination detector.
    
    This is a placeholder. In production, use NLI models or
    fact-checking APIs.
    
    Args:
        response: Model response
        ground_truth: Ground truth answer
        
    Returns:
        Tuple of (is_hallucination, severity_1_to_5)
    """
    response_lower = response.lower()
    truth_lower = ground_truth.lower()
    
    # Check for obvious contradictions (simplified)
    contradictions = [
        ("yes", "no"),
        ("true", "false"),
        ("always", "never"),
        ("all", "none"),
    ]
    
    severity = 1.0
    is_hallucination = False
    
    for pos, neg in contradictions:
        if pos in truth_lower and neg in response_lower:
            is_hallucination = True
            severity = 3.0
            break
        if neg in truth_lower and pos in response_lower:
            is_hallucination = True
            severity = 3.0
            break
    
    # Check for fabricated specifics (numbers, dates)
    import re
    numbers_response = re.findall(r'\d+', response)
    numbers_truth = re.findall(r'\d+', ground_truth)
    
    if numbers_response and not numbers_truth:
        is_hallucination = True
        severity = 4.0
    
    return is_hallucination, severity


def run_regression_test(
    current_pass_at_1: float,
    current_pass_at_k: float,
    baseline_pass_at_1: float,
    baseline_pass_at_k: float,
    k: int = 5,
) -> RegressionTestResult:
    """Run regression test for Pass@k vs Pass@1 trade-off.
    
    Detects the "prompt interference" phenomenon where Pass@k optimization
    degrades Pass@1 performance.
    
    Args:
        current_pass_at_1: Current Pass@1 score
        current_pass_at_k: Current Pass@k score
        baseline_pass_at_1: Baseline Pass@1 score (from previous checkpoint)
        baseline_pass_at_k: Baseline Pass@k score
        k: Value of k used for Pass@k
        
    Returns:
        RegressionTestResult with recommendation
    """
    pass_at_1_delta = current_pass_at_1 - baseline_pass_at_1
    pass_at_k_delta = current_pass_at_k - baseline_pass_at_k
    
    result = RegressionTestResult(
        current_pass_at_1=current_pass_at_1,
        current_pass_at_k=current_pass_at_k,
        baseline_pass_at_1=baseline_pass_at_1,
        baseline_pass_at_k=baseline_pass_at_k,
        pass_at_1_delta=pass_at_1_delta,
        pass_at_k_delta=pass_at_k_delta,
    )
    
    # Regression detection logic
    if pass_at_1_delta < -0.02 and pass_at_k_delta > 0.01:
        # Critical: Pass@1 decreased significantly while Pass@k increased
        result.is_regression = True
        result.regression_severity = "critical"
        result.regression_details = (
            f"CRITICAL REGRESSION: Pass@1 decreased by {abs(pass_at_1_delta):.2%} "
            f"while Pass@{k} increased by {pass_at_k_delta:.2%}. "
            "This indicates prompt interference from Pass@k optimization."
        )
        result.should_promote = False
        result.recommendation = (
            "DO NOT PROMOTE. Revert to baseline and review DPO training data. "
            "Check for over-weighting of difficult prompts or multi-sampling bias."
        )
    elif pass_at_1_delta < -0.01 and pass_at_k_delta > 0.005:
        # Warning: Pass@1 decreased slightly while Pass@k increased
        result.is_regression = True
        result.regression_severity = "warning"
        result.regression_details = (
            f"WARNING: Pass@1 decreased by {abs(pass_at_1_delta):.2%} "
            f"while Pass@{k} increased by {pass_at_k_delta:.2%}. "
            "Monitor closely for prompt interference."
        )
        result.should_promote = True  # Allow with caution
        result.recommendation = (
            "Proceed with caution. Monitor Core Reliability Metrics closely. "
            "Consider adjusting DPO loss weighting to prioritize Pass@1."
        )
    elif pass_at_1_delta >= 0.01:
        # Good: Pass@1 improved
        result.is_regression = False
        result.regression_severity = "none"
        result.regression_details = (
            f"Pass@1 improved by {pass_at_1_delta:.2%}. No regression detected."
        )
        result.should_promote = True
        result.recommendation = "Safe to promote. Pass@1 is improving."
    else:
        # Stable: No significant change
        result.is_regression = False
        result.regression_severity = "none"
        result.regression_details = (
            f"No significant regression. Pass@1 delta: {pass_at_1_delta:.2%}, "
            f"Pass@{k} delta: {pass_at_k_delta:.2%}."
        )
        result.should_promote = True
        result.recommendation = "Stable metrics. Safe to promote."
    
    return result


def compute_core_reliability_metrics(
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
) -> CoreReliabilityMetrics:
    """Compute all Core Reliability Metrics (Tier 1).
    
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
        
    Returns:
        CoreReliabilityMetrics object
    """
    n = len(predictions)
    
    # Convert to numpy arrays
    accuracies = np.array([1 if p == g else 0 for p, g in zip(predictions, ground_truth)])
    confidences_arr = np.array(confidences)
    
    # Pass@1
    pass_at_1 = float(np.mean(accuracies))
    # Wilson score interval for 95% CI
    z = 1.96
    ci_lower = (
        (pass_at_1 + z**2 / (2 * n) - z * np.sqrt((pass_at_1 * (1 - pass_at_1) + z**2 / (4 * n)) / n))
        / (1 + z**2 / n)
    )
    ci_upper = (
        (pass_at_1 + z**2 / (2 * n) + z * np.sqrt((pass_at_1 * (1 - pass_at_1) + z**2 / (4 * n)) / n))
        / (1 + z**2 / n)
    )
    
    # Hallucination rate
    hall_rate, hall_severity = compute_hallucination_rate(predictions, ground_truth)
    
    # Calibration metrics
    ece = compute_expected_calibration_error(confidences_arr, accuracies)
    brier = compute_brier_score(confidences_arr, accuracies)
    
    # Mode accuracy
    mode_correct = sum(1 for p, g in zip(epistemic_modes, gold_modes) if p == g)
    mode_accuracy = mode_correct / n
    
    # Abstention AUROC
    abstention_auroc = compute_abstention_auroc(
        confidences_arr,
        np.array(is_knowable, dtype=int)
    )
    
    # False premise detection
    fp_tp = sum(1 for f, p in zip(false_premise_flags, predicted_false_premise) if f and p)
    fp_fn = sum(1 for f, p in zip(false_premise_flags, predicted_false_premise) if f and not p)
    fp_fp = sum(1 for f, p in zip(false_premise_flags, predicted_false_premise) if not f and p)
    fp_tn = sum(1 for f, p in zip(false_premise_flags, predicted_false_premise) if not f and not p)
    
    fp_detection_rate = fp_tp / (fp_tp + fp_fn) if (fp_tp + fp_fn) > 0 else 0.0
    fp_false_positive_rate = fp_fp / (fp_fp + fp_tn) if (fp_fp + fp_tn) > 0 else 0.0
    
    # Tool request metrics
    tool_tp = sum(1 for n, t in zip(needs_tool, tool_requests) if n and t)
    tool_fn = sum(1 for n, t in zip(needs_tool, tool_requests) if n and not t)
    tool_fp = sum(1 for n, t in zip(needs_tool, tool_requests) if not n and t)
    tool_tn = sum(1 for n, t in zip(needs_tool, tool_requests) if not n and not t)
    
    tool_precision = tool_tp / (tool_tp + tool_fp) if (tool_tp + tool_fp) > 0 else 0.0
    tool_recall = tool_tp / (tool_tp + tool_fn) if (tool_tp + tool_fn) > 0 else 0.0
    tool_f1 = (
        2 * tool_precision * tool_recall / (tool_precision + tool_recall)
        if (tool_precision + tool_recall) > 0 else 0.0
    )
    
    return CoreReliabilityMetrics(
        pass_at_1=pass_at_1,
        pass_at_1_ci_lower=float(ci_lower),
        pass_at_1_ci_upper=float(ci_upper),
        hallucination_rate=hall_rate,
        hallucination_severity_avg=hall_severity,
        expected_calibration_error=ece,
        brier_score=brier,
        mode_accuracy=mode_accuracy,
        abstention_auroc=abstention_auroc,
        false_premise_detection_rate=fp_detection_rate,
        false_premise_false_positive_rate=fp_false_positive_rate,
        tool_request_f1=tool_f1,
        n_samples=n,
    )


def compute_special_metrics(
    math_predictions: Optional[list[str]] = None,
    math_ground_truth: Optional[list[str]] = None,
    math_samples: Optional[int] = None,
    code_predictions: Optional[list[str]] = None,
    code_ground_truth: Optional[list[str]] = None,
    code_samples: Optional[int] = None,
    retrieval_predictions: Optional[list[str]] = None,
    retrieval_ground_truth: Optional[list[str]] = None,
    retrieval_samples: Optional[int] = None,
    k_values: Optional[list[int]] = None,
) -> SpecialMetrics:
    """Compute Optional Special Metrics (Tier 2).
    
    These metrics should only be computed for verifiable domains.
    
    Args:
        math_predictions: Math problem predictions
        math_ground_truth: Math problem ground truth
        code_predictions: Code generation predictions
        code_ground_truth: Code generation ground truth (test results)
        retrieval_predictions: Retrieval task predictions
        retrieval_ground_truth: Retrieval task ground truth
        k_values: List of k values for Pass@k computation
        
    Returns:
        SpecialMetrics object
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    metrics = SpecialMetrics()
    
    # Math Pass@k
    if math_predictions and math_ground_truth:
        n = len(math_predictions)
        metrics.n_math_samples = n
        correct = np.array([1 if p == g else 0 for p, g in zip(math_predictions, math_ground_truth)])
        total = np.ones(n, dtype=int)
        
        for k in k_values:
            metrics.pass_at_k_math[k] = compute_pass_at_k(correct, total, k)
        
        metrics.math_accuracy = float(np.mean(correct))
    
    # Code Pass@k
    if code_predictions and code_ground_truth:
        n = len(code_predictions)
        metrics.n_code_samples = n
        correct = np.array([1 if p == g else 0 for p, g in zip(code_predictions, code_ground_truth)])
        total = np.ones(n, dtype=int)
        
        for k in k_values:
            metrics.pass_at_k_code[k] = compute_pass_at_k(correct, total, k)
        
        metrics.code_pass_rate = float(np.mean(correct))
    
    # Retrieval Pass@k
    if retrieval_predictions and retrieval_ground_truth:
        n = len(retrieval_predictions)
        metrics.n_retrieval_samples = n
        correct = np.array([1 if p == g else 0 for p, g in zip(retrieval_predictions, retrieval_ground_truth)])
        total = np.ones(n, dtype=int)
        
        for k in k_values:
            metrics.pass_at_k_retrieval[k] = compute_pass_at_k(correct, total, k)
        
        metrics.retrieval_precision = float(np.mean(correct))
    
    # Sampling efficiency (if we have multiple k values)
    if len(k_values) >= 2:
        # Compute improvement per additional sample
        if metrics.pass_at_k_math:
            k1, k2 = k_values[0], k_values[-1]
            improvement = metrics.pass_at_k_math.get(k2, 0) - metrics.pass_at_k_math.get(k1, 0)
            metrics.sampling_efficiency = improvement / (k2 - k1) if k2 > k1 else 0.0

    return metrics


# =============================================================================
# SUA (Staleness/Unknown/Ambiguity) Metrics - Phase 3.5
# =============================================================================

def compute_sua_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> SUAMetrics:
    """Compute all SUA Metrics (Tier 3) for Phase 3.5 evaluation.

    Args:
        predictions: List of model predictions with structure:
            {
                "id": str,
                "category": "staleness|unknown|ambiguity",
                "predicted_mode": str,
                "confidence": float,
                "is_stale": bool (optional),
                "unknown_confidence": float (optional),
                "clarification_needed": bool (optional),
                "clarification": str (optional),
                "response": str
            }
        ground_truth: List of ground truth labels with structure:
            {
                "id": str,
                "category": "staleness|unknown|ambiguity",
                "gold_mode": str,
                "is_stale": bool (for staleness samples),
                "is_unknown": bool (for unknown samples),
                "needs_clarification": bool (for ambiguity samples),
                "gold_clarification": str (optional)
            }

    Returns:
        SUAMetrics object with all SUA metrics computed
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")

    # Separate by category
    staleness_preds = []
    staleness_gt = []
    unknown_preds = []
    unknown_gt = []
    ambiguity_preds = []
    ambiguity_gt = []

    for pred, gt in zip(predictions, ground_truth):
        category = gt.get("category", "")
        if category == "staleness":
            staleness_preds.append(pred)
            staleness_gt.append(gt)
        elif category == "unknown":
            unknown_preds.append(pred)
            unknown_gt.append(gt)
        elif category == "ambiguity":
            ambiguity_preds.append(pred)
            ambiguity_gt.append(gt)

    # Compute per-category metrics
    staleness_metrics = compute_staleness_metrics(staleness_preds, staleness_gt)
    unknown_metrics = compute_unknown_metrics(unknown_preds, unknown_gt)
    ambiguity_metrics = compute_ambiguity_metrics(ambiguity_preds, ambiguity_gt)

    # Compute combined SUA score
    combined_score = (
        staleness_metrics["sua_weights"]["staleness"] * staleness_metrics["staleness_f1"] +
        unknown_metrics["sua_weights"]["unknown"] * unknown_metrics["unknown_detection_auroc"] +
        ambiguity_metrics["sua_weights"]["ambiguity"] * ambiguity_metrics["clarification_quality_score"]
    )

    return SUAMetrics(
        staleness_detection_rate=staleness_metrics["staleness_detection_rate"],
        staleness_false_positive_rate=staleness_metrics["staleness_false_positive_rate"],
        staleness_precision=staleness_metrics["staleness_precision"],
        staleness_f1=staleness_metrics["staleness_f1"],
        staleness_n_samples=len(staleness_preds),
        unknown_detection_auroc=unknown_metrics["unknown_detection_auroc"],
        unknown_precision_at_50_recall=unknown_metrics["unknown_precision_at_50_recall"],
        unknown_recall_at_90_precision=unknown_metrics["unknown_recall_at_90_precision"],
        unknown_n_samples=len(unknown_preds),
        ambiguity_resolution_accuracy=ambiguity_metrics["ambiguity_resolution_accuracy"],
        clarification_quality_score=ambiguity_metrics["clarification_quality_score"],
        clarification_rate=ambiguity_metrics["clarification_rate"],
        ambiguity_n_samples=len(ambiguity_preds),
        combined_sua_score=combined_score,
        sua_weights=staleness_metrics["sua_weights"],
        staleness_by_category=staleness_metrics.get("by_category", {}),
        unknown_by_category=unknown_metrics.get("by_category", {}),
        ambiguity_by_category=ambiguity_metrics.get("by_category", {}),
    )


def compute_staleness_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """Compute Staleness Detection metrics.

    Args:
        predictions: List of predictions with 'is_stale' and 'staleness_confidence'
        ground_truth: List of ground truth with 'is_stale' boolean

    Returns:
        Dictionary with staleness metrics
    """
    if not predictions:
        return {
            "staleness_detection_rate": 0.0,
            "staleness_false_positive_rate": 0.0,
            "staleness_precision": 0.0,
            "staleness_f1": 0.0,
            "sua_weights": {"staleness": 0.30},
            "by_category": {}
        }

    # Binary classification metrics
    true_positives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred.get("is_stale", False) and gt.get("is_stale", False)
    )
    false_positives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred.get("is_stale", False) and not gt.get("is_stale", False)
    )
    false_negatives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if not pred.get("is_stale", False) and gt.get("is_stale", False)
    )
    true_negatives = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if not pred.get("is_stale", False) and not gt.get("is_stale", False)
    )

    total_stale = sum(1 for gt in ground_truth if gt.get("is_stale", False))
    total_fresh = len(ground_truth) - total_stale

    detection_rate = true_positives / max(total_stale, 1)
    false_positive_rate = false_positives / max(total_fresh, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = detection_rate
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Per-category breakdown
    by_category = {}
    categories = set(gt.get("subcategory", "general") for gt in ground_truth)
    for cat in categories:
        cat_preds = [p for p, gt in zip(predictions, ground_truth) if gt.get("subcategory") == cat]
        cat_gt = [gt for gt in ground_truth if gt.get("subcategory") == cat]
        if cat_preds:
            cat_metrics = compute_staleness_metrics(cat_preds, cat_gt)
            by_category[cat] = cat_metrics["staleness_f1"]

    return {
        "staleness_detection_rate": detection_rate,
        "staleness_false_positive_rate": false_positive_rate,
        "staleness_precision": precision,
        "staleness_f1": f1,
        "sua_weights": {"staleness": 0.30},
        "by_category": by_category
    }


def compute_unknown_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """Compute Unknown Detection metrics using AUROC.

    Args:
        predictions: List of predictions with 'unknown_confidence' scores
        ground_truth: List of ground truth with 'is_unknown' boolean

    Returns:
        Dictionary with unknown detection metrics
    """
    if not predictions:
        return {
            "unknown_detection_auroc": 0.5,
            "unknown_precision_at_50_recall": 0.0,
            "unknown_recall_at_90_precision": 0.0,
            "sua_weights": {"unknown": 0.40},
            "by_category": {}
        }

    # Extract scores and labels
    y_scores = [pred.get("unknown_confidence", 0.5) for pred in predictions]
    y_true = [1 if gt.get("is_unknown", False) else 0 for gt in ground_truth]

    # AUROC computation
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle edge cases (e.g., all same class)
        auroc = 0.5

    # Precision at 50% recall
    precision_at_50 = _compute_precision_at_recall(y_true, y_scores, target_recall=0.5)

    # Recall at 90% precision
    recall_at_90 = _compute_recall_at_precision(y_true, y_scores, target_precision=0.9)

    # Per-category breakdown
    by_category = {}
    categories = set(gt.get("subcategory", "general") for gt in ground_truth)
    for cat in categories:
        cat_preds = [p for p, gt in zip(predictions, ground_truth) if gt.get("subcategory") == cat]
        cat_gt = [gt for gt in ground_truth if gt.get("subcategory") == cat]
        if cat_preds:
            cat_metrics = compute_unknown_metrics(cat_preds, cat_gt)
            by_category[cat] = cat_metrics["unknown_detection_auroc"]

    return {
        "unknown_detection_auroc": auroc,
        "unknown_precision_at_50_recall": precision_at_50,
        "unknown_recall_at_90_precision": recall_at_90,
        "sua_weights": {"unknown": 0.40},
        "by_category": by_category
    }


def compute_ambiguity_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """Compute Ambiguity Resolution metrics.

    Args:
        predictions: List of predictions with 'clarification_needed' and 'clarification'
        ground_truth: List of ground truth with 'needs_clarification' and 'gold_clarification'

    Returns:
        Dictionary with ambiguity metrics
    """
    if not predictions:
        return {
            "ambiguity_resolution_accuracy": 0.0,
            "clarification_quality_score": 0.0,
            "clarification_rate": 0.0,
            "sua_weights": {"ambiguity": 0.30},
            "by_category": {}
        }

    # Binary classification accuracy for clarification detection
    correct_clarification = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred.get("clarification_needed", False) == gt.get("needs_clarification", False)
    )
    accuracy = correct_clarification / len(predictions)

    # Clarification quality (semantic similarity)
    clarification_scores = []
    for pred, gt in zip(predictions, ground_truth):
        if gt.get("needs_clarification", False):
            pred_clarification = pred.get("clarification", "")
            gold_clarification = gt.get("gold_clarification", "")
            if pred_clarification and gold_clarification:
                # Simple overlap-based similarity (replace with better metric in production)
                similarity = _semantic_similarity(pred_clarification, gold_clarification)
                clarification_scores.append(similarity)

    quality_score = sum(clarification_scores) / max(len(clarification_scores), 1) if clarification_scores else 0.0

    # Clarification rate
    clarification_rate = sum(1 for pred in predictions if pred.get("clarification_needed", False)) / len(predictions)

    # Per-category breakdown
    by_category = {}
    categories = set(gt.get("subcategory", "general") for gt in ground_truth)
    for cat in categories:
        cat_preds = [p for p, gt in zip(predictions, ground_truth) if gt.get("subcategory") == cat]
        cat_gt = [gt for gt in ground_truth if gt.get("subcategory") == cat]
        if cat_preds:
            cat_metrics = compute_ambiguity_metrics(cat_preds, cat_gt)
            by_category[cat] = cat_metrics["clarification_quality_score"]

    return {
        "ambiguity_resolution_accuracy": accuracy,
        "clarification_quality_score": quality_score,
        "clarification_rate": clarification_rate,
        "sua_weights": {"ambiguity": 0.30},
        "by_category": by_category
    }


def _compute_precision_at_recall(
    y_true: List[int],
    y_scores: List[float],
    target_recall: float = 0.5,
) -> float:
    """Compute precision at a given recall level.

    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        target_recall: Target recall level (0-1)

    Returns:
        Precision at target recall
    """
    if sum(y_true) == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    sorted_true = np.array(y_true)[sorted_indices]

    # Compute precision-recall curve
    n_pos = sum(y_true)
    tp = 0
    fp = 0
    precision_at_recall = 0.0

    for i, label in enumerate(sorted_true):
        if label == 1:
            tp += 1
        else:
            fp += 1

        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if recall >= target_recall:
            precision_at_recall = precision
            break

    return precision_at_recall


def _compute_recall_at_precision(
    y_true: List[int],
    y_scores: List[float],
    target_precision: float = 0.9,
) -> float:
    """Compute recall at a given precision level.

    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        target_precision: Target precision level (0-1)

    Returns:
        Recall at target precision
    """
    if sum(y_true) == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    sorted_true = np.array(y_true)[sorted_indices]

    # Compute precision-recall curve
    n_pos = sum(y_true)
    tp = 0
    fp = 0
    recall_at_precision = 0.0

    for i, label in enumerate(sorted_true):
        if label == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos

        if precision < target_precision and i > 0:
            # Previous point was the last one meeting precision target
            break
        recall_at_precision = recall

    return recall_at_precision


def _semantic_similarity(text1: str, text2: str) -> float:
    """Compute simple semantic similarity between two texts.

    This is a placeholder using token overlap. In production, use
    sentence embeddings or a more sophisticated metric.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Simple token overlap (Jaccard similarity)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0

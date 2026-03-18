"""Diogenes - The Reliable 32B: Epistemically optimized language model.

Diogenes optimizes for reliable single-response decisions, not multi-sampling success.
"""

__version__ = "0.1.0"
__author__ = "Diogenes Team"

from diogenes.model import DiogenesModel, load_base_model
from diogenes.inference import DiogenesInference
from diogenes.config import EpistemicMode

# Pass@1 Protection imports
from diogenes.eval_metrics import (
    compute_pass_at_k,
    compute_expected_calibration_error,
    compute_brier_score,
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

__all__ = [
    # Core
    "DiogenesModel",
    "load_base_model",
    "DiogenesInference",
    "EpistemicMode",
    # Eval Metrics
    "compute_pass_at_k",
    "compute_expected_calibration_error",
    "compute_brier_score",
    "compute_core_reliability_metrics",
    "compute_special_metrics",
    "run_regression_test",
    "CoreReliabilityMetrics",
    "SpecialMetrics",
    "RegressionTestResult",
    # Pass@1 Protection
    "Pass1RegressionTracker",
    "run_pass1_protection_test",
    "check_dpo_for_prompt_interference",
]

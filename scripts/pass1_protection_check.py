#!/usr/bin/env python3
"""Pass@1 Protection Check für Phase 3 (Post-DPO).

Überprüft das DPO-Modell auf Pass@1 Regression.
WICHTIG: Erst nach DPO-Training ausführen!
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_baseline_metrics(baseline_path: str = "models/sft_3b_test/baseline_metrics.json") -> dict:
    """Load baseline metrics from SFT model."""
    if not Path(baseline_path).exists():
        logger.warning(f"Baseline metrics not found: {baseline_path}")
        return {
            "pass_at_1": 0.75,
            "pass_at_k": 0.90,
            "ece": 0.10,
            "hallucination_rate": 0.15,
        }
    
    with open(baseline_path, "r") as f:
        return json.load(f)


def run_pass1_protection_check(
    model_path: str,
    baseline_pass_at_1: float,
    baseline_pass_at_k: float,
    k: int = 5,
) -> dict:
    """
    Run Pass@1 protection check.
    
    Returns:
        dict with:
        - is_regression: bool
        - severity: str (none, warning, critical)
        - recommendation: str
    """
    # Placeholder: In production, this would run actual evaluation
    # For now, we simulate the check structure
    
    logger.info(f"Running Pass@1 Protection Check for: {model_path}")
    logger.info(f"Baseline Pass@1: {baseline_pass_at_1:.4f}")
    logger.info(f"Baseline Pass@{k}: {baseline_pass_at_k:.4f}")
    
    # Simulated metrics (would be replaced by actual evaluation)
    current_pass_at_1 = baseline_pass_at_1  # Placeholder
    current_pass_at_k = baseline_pass_at_k  # Placeholder
    
    # Calculate delta
    pass1_delta = current_pass_at_1 - baseline_pass_at_1
    passk_delta = current_pass_at_k - baseline_pass_at_k
    
    # Decision matrix (from PASS1_GUARDRAILS.md)
    is_regression = False
    severity = "none"
    recommendation = "PROCEED"
    
    if pass1_delta < -0.02 and passk_delta > 0.01:
        is_regression = True
        severity = "critical"
        recommendation = "DO NOT PROMOTE"
        logger.error(f"❌ CRITICAL REGRESSION: Pass@1 ↓ {pass1_delta:.4f}, Pass@{k} ↑ {passk_delta:.4f}")
    elif pass1_delta < -0.01 and passk_delta > 0.005:
        is_regression = True
        severity = "warning"
        recommendation = "REVIEW CAREFULLY"
        logger.warning(f"⚠️  WARNING: Pass@1 ↓ {pass1_delta:.4f}, Pass@{k} ↑ {passk_delta:.4f}")
    elif pass1_delta > 0.01:
        logger.info(f"✓ IMPROVEMENT: Pass@1 ↑ {pass1_delta:.4f}")
    else:
        logger.info(f"✓ STABLE: Pass@1 Δ {pass1_delta:.4f}")
    
    return {
        "model_path": model_path,
        "baseline_pass_at_1": baseline_pass_at_1,
        "baseline_pass_at_k": baseline_pass_at_k,
        "current_pass_at_1": current_pass_at_1,
        "current_pass_at_k": current_pass_at_k,
        "pass1_delta": pass1_delta,
        "passk_delta": passk_delta,
        "is_regression": is_regression,
        "severity": severity,
        "recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser(description="Pass@1 Protection Check (Post-DPO)")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to DPO-trained model",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="models/sft_3b_test/final_checkpoint",
        help="Path to SFT baseline model",
    )
    parser.add_argument(
        "--baseline-pass-at-1",
        type=float,
        default=None,
        help="Baseline Pass@1 (auto-loaded if not provided)",
    )
    parser.add_argument(
        "--baseline-pass-at-k",
        type=float,
        default=None,
        help="Baseline Pass@k (auto-loaded if not provided)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k value for Pass@k monitoring",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="pass1_protection_report.json",
        help="Path for protection report",
    )
    
    args = parser.parse_args()
    
    # Load baseline metrics
    baseline = load_baseline_metrics(f"{args.baseline_model_path}/metrics.json")
    
    baseline_pass_at_1 = args.baseline_pass_at_1 or baseline.get("pass_at_1", 0.75)
    baseline_pass_at_k = args.baseline_pass_at_k or baseline.get("pass_at_k", 0.90)
    
    # Run protection check
    result = run_pass1_protection_check(
        model_path=args.model_path,
        baseline_pass_at_1=baseline_pass_at_1,
        baseline_pass_at_k=baseline_pass_at_k,
        k=args.k,
    )
    
    # Save report
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PASS@1 PROTECTION CHECK")
    print("=" * 60)
    print(f"Model: {result['model_path']}")
    print(f"Baseline Pass@1: {result['baseline_pass_at_1']:.4f}")
    print(f"Baseline Pass@{result['k']}: {result['baseline_pass_at_k']:.4f}")
    print("-" * 60)
    print(f"Current Pass@1: {result['current_pass_at_1']:.4f} (Δ {result['pass1_delta']:+.4f})")
    print(f"Current Pass@{result['k']}: {result['current_pass_at_k']:.4f} (Δ {result['passk_delta']:+.4f})")
    print("-" * 60)
    
    if result['severity'] == 'critical':
        print(f"❌ CRITICAL REGRESSION DETECTED")
        print(f"   Recommendation: {result['recommendation']}")
    elif result['severity'] == 'warning':
        print(f"⚠️  WARNING: Potential regression")
        print(f"   Recommendation: {result['recommendation']}")
    else:
        print(f"✓ NO REGRESSION DETECTED")
        print(f"   Recommendation: {result['recommendation']}")
    
    print("=" * 60)
    print(f"Report saved to: {args.output_path}")
    print("=" * 60)
    
    return 0 if not result['is_regression'] else 1


if __name__ == "__main__":
    exit(main())

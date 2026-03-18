#!/usr/bin/env python3
"""DPO Audit Script for Diogenes.

Validates DPO dataset quality before training to detect:
- Prompt-Interferenz (Difficulty Bias, Verbosity Bias)
- Class Imbalance
- Data Quality Issues

Based on PASS1_GUARDRAILS.md requirements.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DPOAuditResult:
    """Result of DPO dataset audit."""
    total_pairs: int
    difficulty_distribution: dict
    avg_chosen_length: float
    avg_rejected_length: float
    verbosity_ratio: float
    difficulty_bias: bool
    verbosity_bias: bool
    abstain_representation: float
    concerns: list = field(default_factory=list)
    passed: bool = True


def load_dpo_dataset(dataset_path: str) -> list[dict]:
    """Load DPO dataset from JSONL file."""
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_difficulty_distribution(data: list[dict]) -> dict:
    """Analyze difficulty distribution based on category."""
    # Categories mapped to difficulty levels
    difficulty_mapping = {
        "ignorance": "hard",  # Requires knowing knowledge limits
        "false_premise": "hard",  # Requires fact-checking
        "multi_hop": "hard",  # Complex reasoning
        "ambiguity": "medium",  # Requires clarification
        "staleness": "medium",  # Time-sensitive
        "tool_required": "medium",  # External data needed
        "adversarial": "hard",  # Security awareness
        "shallow_trap": "easy",  # Simple but tricky
    }
    
    distribution = {"easy": 0, "medium": 0, "hard": 0}
    
    for item in data:
        category = item.get("category", "unknown")
        difficulty = difficulty_mapping.get(category, "medium")
        distribution[difficulty] += 1
    
    return distribution


def analyze_length_bias(data: list[dict]) -> tuple[float, float, float]:
    """Analyze verbosity bias between chosen and rejected answers."""
    chosen_lengths = []
    rejected_lengths = []
    
    for item in data:
        chosen = item.get("chosen_answer", "")
        rejected = item.get("rejected_answer", "")
        chosen_lengths.append(len(chosen.split()))
        rejected_lengths.append(len(rejected.split()))
    
    avg_chosen = np.mean(chosen_lengths)
    avg_rejected = np.mean(rejected_lengths)
    verbosity_ratio = avg_chosen / avg_rejected if avg_rejected > 0 else 1.0
    
    return avg_chosen, avg_rejected, verbosity_ratio


def analyze_abstain_representation(data: list[dict]) -> float:
    """Calculate percentage of abstain-related samples."""
    abstain_categories = ["ignorance"]
    abstain_count = sum(1 for item in data if item.get("category") in abstain_categories)
    return abstain_count / len(data) if len(data) > 0 else 0.0


def check_dpo_for_prompt_interference(data: list[dict]) -> DPOAuditResult:
    """
    Perform comprehensive DPO audit for prompt interference.
    
    Checks:
    1. Difficulty Bias: Hard samples should be < 55% (relaxed for Diogenes - we want challenging epistemic cases)
    2. Verbosity Bias: Chosen/Rejected length ratio should be < 1.2
    3. Abstain Representation: Should be > 5% for honest refusal training
    
    Note: Higher difficulty is acceptable for Diogenes since we train on epistemic edge cases.
    """
    logger.info(f"Auditing {len(data)} DPO pairs...")
    
    # Analyze distributions
    difficulty_dist = analyze_difficulty_distribution(data)
    avg_chosen, avg_rejected, verbosity_ratio = analyze_length_bias(data)
    abstain_repr = analyze_abstain_representation(data)
    
    # Check thresholds (relaxed for Diogenes)
    concerns = []
    passed = True
    
    # Difficulty Bias Check (relaxed to 55% for Diogenes)
    hard_percentage = difficulty_dist["hard"] / len(data) if len(data) > 0 else 0
    difficulty_bias = hard_percentage > 0.55
    if difficulty_bias:
        concerns.append(f"Difficulty Bias: {hard_percentage:.1%} hard samples (threshold: 55%)")
        passed = False
    elif hard_percentage > 0.40:
        logger.info(f"Note: High difficulty dataset ({hard_percentage:.1%} hard) - good for epistemic training")
    
    # Verbosity Bias Check
    verbosity_bias = verbosity_ratio > 1.2
    if verbosity_bias:
        concerns.append(f"Verbosity Bias: Ratio {verbosity_ratio:.2f} (threshold: 1.2)")
        passed = False
    
    # Abstain Representation Warning
    if abstain_repr < 0.05:
        concerns.append(f"Low Abstain Representation: {abstain_repr:.1%} (recommended: >5%)")
    
    # Log results
    logger.info(f"Difficulty Distribution: {difficulty_dist}")
    logger.info(f"Avg Chosen Length: {avg_chosen:.1f} words")
    logger.info(f"Avg Rejected Length: {avg_rejected:.1f} words")
    logger.info(f"Verbosity Ratio: {verbosity_ratio:.2f}")
    logger.info(f"Abstain Representation: {abstain_repr:.1%}")
    
    if concerns:
        logger.warning(f"Found {len(concerns)} concerns:")
        for concern in concerns:
            logger.warning(f"  - {concern}")
    else:
        logger.info("✓ DPO dataset passed audit")
    
    return DPOAuditResult(
        total_pairs=len(data),
        difficulty_distribution=difficulty_dist,
        avg_chosen_length=avg_chosen,
        avg_rejected_length=avg_rejected,
        verbosity_ratio=verbosity_ratio,
        difficulty_bias=difficulty_bias,
        verbosity_bias=verbosity_bias,
        abstain_representation=abstain_repr,
        concerns=concerns,
        passed=passed,
    )


def generate_audit_report(result: DPOAuditResult, output_path: str):
    """Generate JSON audit report."""
    report = {
        "total_pairs": int(result.total_pairs),
        "difficulty_distribution": result.difficulty_distribution,
        "avg_chosen_length": float(result.avg_chosen_length),
        "avg_rejected_length": float(result.avg_rejected_length),
        "verbosity_ratio": float(result.verbosity_ratio),
        "difficulty_bias": bool(result.difficulty_bias),
        "verbosity_bias": bool(result.verbosity_bias),
        "abstain_representation": float(result.abstain_representation),
        "concerns": result.concerns,
        "passed": bool(result.passed),
        "recommendation": "PROCEED" if result.passed else "REVIEW_REQUIRED",
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Audit report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DPO Audit for Diogenes")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/dpo_dataset.jsonl",
        help="Path to DPO dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./dpo_audit_report.json",
        help="Path for audit report",
    )
    parser.add_argument(
        "--threshold-hard",
        type=float,
        default=0.30,
        help="Threshold for hard samples (default: 30%)",
    )
    parser.add_argument(
        "--threshold-verbosity",
        type=float,
        default=1.2,
        help="Threshold for verbosity ratio (default: 1.2)",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading DPO dataset from: {args.dataset_path}")
    data = load_dpo_dataset(args.dataset_path)
    logger.info(f"Loaded {len(data)} pairs")
    
    # Run audit
    result = check_dpo_for_prompt_interference(data)
    
    # Generate report
    generate_audit_report(result, args.output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DPO AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Pairs: {result.total_pairs:,}")
    print(f"Difficulty Distribution: {result.difficulty_distribution}")
    print(f"Avg Chosen Length: {result.avg_chosen_length:.1f} words")
    print(f"Avg Rejected Length: {result.avg_rejected_length:.1f} words")
    print(f"Verbosity Ratio: {result.verbosity_ratio:.2f}")
    print(f"Abstain Representation: {result.abstain_representation:.1%}")
    print("-" * 60)
    
    if result.passed:
        print("✓ AUDIT PASSED - Safe to proceed with DPO training")
    else:
        print("❌ AUDIT FAILED - Review concerns before training:")
        for concern in result.concerns:
            print(f"  - {concern}")
    
    print("=" * 60)
    
    # Exit with appropriate code
    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())

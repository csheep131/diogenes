#!/usr/bin/env python3
"""Test script for Diogenes dataset generator.

Generates small sample datasets and validates JSON schema.
"""

import argparse
import json
import logging
from pathlib import Path

from diogenes.dataset_generator import (
    DatasetGenerator,
    SFTSample,
    DPOPair,
    EpistemicMode,
    ErrorClass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# JSON Schema for SFT samples
SFT_SCHEMA = {
    "type": "object",
    "required": [
        "id",
        "question",
        "category",
        "gold_mode",
        "risk_level",
        "needs_tool",
        "time_sensitive",
        "false_premise",
        "confidence_target",
        "answer",
    ],
    "properties": {
        "id": {"type": "string"},
        "question": {"type": "string"},
        "category": {
            "type": "string",
            "enum": [e.value for e in ErrorClass],
        },
        "gold_mode": {
            "type": "string",
            "enum": [e.value for e in EpistemicMode],
        },
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "needs_tool": {"type": "boolean"},
        "time_sensitive": {"type": "boolean"},
        "false_premise": {"type": "boolean"},
        "confidence_target": {"type": "number", "minimum": 0, "maximum": 1},
        "answer": {"type": "string"},
        "reasoning_trace": {"type": "string"},
        "metadata": {"type": "object"},
    },
}

# JSON Schema for DPO pairs
DPO_SCHEMA = {
    "type": "object",
    "required": [
        "id",
        "question",
        "category",
        "gold_mode",
        "risk_level",
        "needs_tool",
        "time_sensitive",
        "false_premise",
        "confidence_target",
        "chosen_answer",
        "rejected_answer",
        "chosen_rank",
        "rejected_rank",
    ],
    "properties": {
        "id": {"type": "string"},
        "question": {"type": "string"},
        "category": {
            "type": "string",
            "enum": [e.value for e in ErrorClass],
        },
        "gold_mode": {
            "type": "string",
            "enum": [e.value for e in EpistemicMode],
        },
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "needs_tool": {"type": "boolean"},
        "time_sensitive": {"type": "boolean"},
        "false_premise": {"type": "boolean"},
        "confidence_target": {"type": "number", "minimum": 0, "maximum": 1},
        "chosen_answer": {"type": "string"},
        "rejected_answer": {"type": "string"},
        "chosen_rank": {"type": "integer", "minimum": 1, "maximum": 4},
        "rejected_rank": {"type": "integer", "minimum": 1, "maximum": 4},
        "reasoning_trace": {"type": "string"},
        "metadata": {"type": "object"},
    },
}


def validate_schema(data: dict, schema: dict, sample_type: str) -> bool:
    """Validate data against schema."""
    errors = []
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check field types and enums
    for field, spec in schema.get("properties", {}).items():
        if field not in data:
            continue
        
        value = data[field]
        expected_type = spec.get("type")
        
        # Type checking
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field}' should be boolean, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Field '{field}' should be integer, got {type(value).__name__}")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"Field '{field}' should be object, got {type(value).__name__}")
        
        # Enum checking
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"Field '{field}' value '{value}' not in allowed values: {spec['enum']}")
        
        # Range checking
        if "minimum" in spec and isinstance(value, (int, float)):
            if value < spec["minimum"]:
                errors.append(f"Field '{field}' value {value} below minimum {spec['minimum']}")
        if "maximum" in spec and isinstance(value, (int, float)):
            if value > spec["maximum"]:
                errors.append(f"Field '{field}' value {value} above maximum {spec['maximum']}")
    
    if errors:
        logger.error(f"{sample_type} validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True


def test_sft_sample(generator: DatasetGenerator) -> bool:
    """Test single SFT sample generation."""
    logger.info("Testing SFT sample generation...")
    
    sample = generator.generate_sft_sample()
    
    # Print sample for inspection
    logger.info(f"Generated SFT Sample:")
    logger.info(f"  ID: {sample.id}")
    logger.info(f"  Category: {sample.category}")
    logger.info(f"  Mode: {sample.gold_mode}")
    logger.info(f"  Question: {sample.question[:80]}...")
    logger.info(f"  Confidence Target: {sample.confidence_target}")
    logger.info(f"  Risk Level: {sample.risk_level}")
    
    # Validate against schema
    sample_dict = {
        "id": sample.id,
        "question": sample.question,
        "category": sample.category,
        "gold_mode": sample.gold_mode,
        "risk_level": sample.risk_level,
        "needs_tool": sample.needs_tool,
        "time_sensitive": sample.time_sensitive,
        "false_premise": sample.false_premise,
        "confidence_target": sample.confidence_target,
        "answer": sample.answer,
        "reasoning_trace": sample.reasoning_trace,
        "metadata": sample.metadata,
    }
    
    return validate_schema(sample_dict, SFT_SCHEMA, "SFT Sample")


def test_dpo_pair(generator: DatasetGenerator) -> bool:
    """Test single DPO pair generation."""
    logger.info("Testing DPO pair generation...")
    
    pair = generator.generate_dpo_pair()
    
    # Print pair for inspection
    logger.info(f"Generated DPO Pair:")
    logger.info(f"  ID: {pair.id}")
    logger.info(f"  Category: {pair.category}")
    logger.info(f"  Mode: {pair.gold_mode}")
    logger.info(f"  Question: {pair.question[:80]}...")
    logger.info(f"  Chosen Rank: {pair.chosen_rank}")
    logger.info(f"  Rejected Rank: {pair.rejected_rank}")
    
    # Validate against schema
    pair_dict = {
        "id": pair.id,
        "question": pair.question,
        "category": pair.category,
        "gold_mode": pair.gold_mode,
        "risk_level": pair.risk_level,
        "needs_tool": pair.needs_tool,
        "time_sensitive": pair.time_sensitive,
        "false_premise": pair.false_premise,
        "confidence_target": pair.confidence_target,
        "chosen_answer": pair.chosen_answer,
        "rejected_answer": pair.rejected_answer,
        "chosen_rank": pair.chosen_rank,
        "rejected_rank": pair.rejected_rank,
        "reasoning_trace": pair.reasoning_trace,
        "metadata": pair.metadata,
    }
    
    return validate_schema(pair_dict, DPO_SCHEMA, "DPO Pair")


def test_dataset_generation(output_dir: str, num_samples: int = 10) -> bool:
    """Test full dataset generation."""
    logger.info(f"Testing dataset generation with {num_samples} samples each...")
    
    generator = DatasetGenerator(
        sft_samples=num_samples,
        dpo_pairs=num_samples,
        seed=42,
        output_dir=output_dir,
    )
    
    # Generate SFT dataset
    logger.info("Generating SFT dataset...")
    sft_data = generator.generate_sft_dataset(num_samples)
    sft_path = generator.save_dataset(sft_data, "test_sft_dataset.jsonl")
    
    # Validate SFT samples
    sft_valid = all(
        validate_schema(sample, SFT_SCHEMA, "SFT Sample")
        for sample in sft_data
    )
    
    # Generate DPO dataset
    logger.info("Generating DPO dataset...")
    dpo_data = generator.generate_dpo_dataset(num_samples)
    dpo_path = generator.save_dataset(dpo_data, "test_dpo_dataset.jsonl")
    
    # Validate DPO pairs
    dpo_valid = all(
        validate_schema(pair, DPO_SCHEMA, "DPO Pair")
        for pair in dpo_data
    )
    
    # Generate statistics
    generator._generate_statistics(sft_data, "test_sft_statistics.json")
    generator._generate_statistics(dpo_data, "test_dpo_statistics.json")
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics:")
    logger.info("=" * 60)
    
    with open(Path(output_dir) / "test_sft_statistics.json") as f:
        sft_stats = json.load(f)
        logger.info(f"SFT Dataset:")
        logger.info(f"  Total samples: {sft_stats['total_samples']}")
        logger.info(f"  Category distribution: {sft_stats['category_distribution']}")
        logger.info(f"  Avg confidence: {sft_stats['avg_confidence_target']:.3f}")
    
    with open(Path(output_dir) / "test_dpo_statistics.json") as f:
        dpo_stats = json.load(f)
        logger.info(f"DPO Dataset:")
        logger.info(f"  Total pairs: {dpo_stats['total_samples']}")
        logger.info(f"  Category distribution: {dpo_stats['category_distribution']}")
    
    logger.info("=" * 60)
    
    return sft_valid and dpo_valid


def test_error_class_coverage(generator: DatasetGenerator) -> bool:
    """Test that all error classes are covered."""
    logger.info("Testing error class coverage...")
    
    generated_classes = set()
    generated_modes = set()
    
    # Generate samples until all classes are covered
    for _ in range(1000):
        sample = generator.generate_sft_sample()
        generated_classes.add(sample.category)
        generated_modes.add(sample.gold_mode)
    
    expected_classes = {e.value for e in ErrorClass}
    expected_modes = {e.value for e in EpistemicMode}
    
    missing_classes = expected_classes - generated_classes
    missing_modes = expected_modes - generated_modes
    
    if missing_classes:
        logger.warning(f"Missing error classes: {missing_classes}")
    else:
        logger.info("All error classes covered ✓")
    
    if missing_modes:
        logger.warning(f"Missing epistemic modes: {missing_modes}")
    else:
        logger.info("All epistemic modes covered ✓")
    
    return len(missing_classes) == 0 and len(missing_modes) == 0


def main():
    parser = argparse.ArgumentParser(description="Test Diogenes dataset generator")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory for test datasets",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for dataset generation test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Diogenes Dataset Generator Tests")
    logger.info("=" * 60)
    
    # Initialize generator
    generator = DatasetGenerator(
        sft_samples=args.num_samples,
        dpo_pairs=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run tests
    all_passed = True
    
    # Test 1: Single SFT sample
    logger.info("\n[Test 1] Single SFT Sample Generation")
    if not test_sft_sample(generator):
        all_passed = False
    
    # Test 2: Single DPO pair
    logger.info("\n[Test 2] Single DPO Pair Generation")
    if not test_dpo_pair(generator):
        all_passed = False
    
    # Test 3: Error class coverage
    logger.info("\n[Test 3] Error Class Coverage")
    if not test_error_class_coverage(generator):
        all_passed = False
    
    # Test 4: Full dataset generation
    logger.info("\n[Test 4] Full Dataset Generation")
    if not test_dataset_generation(args.output_dir, args.num_samples):
        all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All tests passed! ✓")
    else:
        logger.info("Some tests failed! ✗")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

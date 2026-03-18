#!/usr/bin/env python3
"""Test script for Diogenes inference.

This script tests basic inference capabilities after loading the model.
"""

import argparse
import logging
import sys

import torch

from diogenes import DiogenesInference, load_base_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TEST_PROMPTS = [
    # Direct answer test
    "What is the capital of France?",
    # Uncertainty test
    "What will be the exact temperature in Munich on January 15, 2030?",
    # False premise test
    "Who was the first president of Germany in 1800?",
    # Clarification test
    "What is the best way to fix it?",
    # Knowledge boundary test
    "What are the private thoughts of Angela Merkel right now?",
]


def verify_cuda() -> bool:
    """Verify CUDA availability."""
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    return cuda_available


def run_inference_tests(model_path: str | None = None, use_4bit: bool = False) -> None:
    """Run inference tests.

    Args:
        model_path: Path to model (local or HF)
        use_4bit: Use 4-bit quantization
    """
    # Verify environment
    cuda_ok = verify_cuda()
    if not cuda_ok:
        logger.warning("CUDA not available - inference will be slow")

    # Load model - default to small Qwen3-0.6B for Phase 0
    model_name = model_path or "Qwen/Qwen3-0.6B"
    logger.info(f"Loading model: {model_name}")
    logger.info("Phase 0: Using small model for pipeline validation")

    try:
        model = load_base_model(model_name=model_name, cache_dir="./models")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("\nMake sure you have:")
        logger.error("  1. HuggingFace CLI login configured")
        logger.error("  2. Access to Qwen3-32B model")
        logger.error("  3. Sufficient disk space (~70GB)")
        sys.exit(1)

    # Create inference engine
    inference = DiogenesInference(model)

    # Run tests
    logger.info("\n" + "=" * 60)
    logger.info("Running inference tests")
    logger.info("=" * 60)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        logger.info(f"\n[Test {i}/5]")
        logger.info(f"Prompt: {prompt}")
        logger.info("-" * 40)

        try:
            result = inference.generate(prompt, return_logprobs=True)
            logger.info(f"Mode: {result.epistemic_mode.value}")
            logger.info(f"Confidence: {result.confidence:.4f}")
            logger.info(f"Response: {result.text[:300]}")
        except Exception as e:
            logger.error(f"Test failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Inference tests completed")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Diogenes inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (local or HuggingFace)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    args = parser.parse_args()

    run_inference_tests(
        model_path=args.model_path,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()

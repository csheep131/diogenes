#!/usr/bin/env python3
"""Download Qwen3-32B model from HuggingFace.

This script downloads the Qwen3-32B base model for local development.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import login, snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: str = "./models",
    token: str | None = None,
) -> None:
    """Download model from HuggingFace.

    Args:
        model_name: HuggingFace model name. 
                    Recommended: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen2.5-3B-Instruct
        cache_dir: Directory to store model
        token: HuggingFace API token
    """
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Cache directory: {cache_path}")
    logger.info("Phase 0: Small model for pipeline validation")

    # Login if token provided
    if token:
        login(token=token)
        logger.info("Logged in to HuggingFace")

    # Download model
    try:
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_path,
            local_dir=cache_path / model_name.split("/")[-1],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Only download safetensors
        )
        logger.info(f"Model downloaded successfully: {local_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("\nPossible issues:")
        logger.error("  1. Not logged in: run 'huggingface-cli login'")
        logger.error("  2. No access to Qwen3-32B: request access on HuggingFace")
        logger.error("  3. Insufficient disk space: need ~70GB")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-32B model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Cache directory for model",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token",
    )
    args = parser.parse_args()

    download_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        token=args.token,
    )


if __name__ == "__main__":
    main()

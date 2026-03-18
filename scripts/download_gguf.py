#!/usr/bin/env python3
"""Download GGUF quantized models for efficient local inference.

This script downloads GGUF quantized models optimized for consumer GPUs.
Recommended for RTX 3050 (4-8GB VRAM).
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Recommended GGUF models for Phase 0
RECOMMENDED_MODELS = {
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B-GGUF",
        "files": ["qwen3-0.6b-q4_k_m.gguf", "qwen3-0.6b-q5_k_m.gguf"],
        "description": "Smallest Qwen3, ~500MB, good for pipeline tests",
    },
    "qwen3-1.7b": {
        "repo": "Qwen/Qwen3-1.7B-GGUF",
        "files": ["qwen3-1.7b-q4_k_m.gguf", "qwen3-1.7b-q5_k_m.gguf"],
        "description": "Small Qwen3, ~1GB, better quality",
    },
    "qwen2.5-3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "files": ["Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-3B-Instruct-Q5_K_M.gguf"],
        "description": "Best balance for RTX 3050, ~2GB",
    },
}


def download_gguf(
    repo_id: str,
    filename: str,
    cache_dir: str = "./models",
) -> Path:
    """Download GGUF model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        filename: GGUF filename
        cache_dir: Cache directory

    Returns:
        Path to downloaded file
    """
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {filename} from {repo_id}")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_path,
        local_dir=cache_path / "gguf",
    )

    logger.info(f"Downloaded to: {local_path}")
    return Path(local_path)


def list_available_models() -> None:
    """List available GGUF models."""
    print("\nAvailable GGUF Models for Phase 0:")
    print("=" * 60)
    for key, info in RECOMMENDED_MODELS.items():
        print(f"\n{key.upper()}")
        print(f"  Repo: {info['repo']}")
        print(f"  Description: {info['description']}")
        print(f"  Files:")
        for f in info["files"]:
            print(f"    - {f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF quantized models for efficient inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-0.6b",
        choices=list(RECOMMENDED_MODELS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m"],
        help="Quantization level",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Cache directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    model_info = RECOMMENDED_MODELS[args.model]
    filename = next(
        (f for f in model_info["files"] if args.quantization.replace("_", "_") in f.lower()),
        model_info["files"][0],
    )

    download_gguf(
        repo_id=model_info["repo"],
        filename=filename,
        cache_dir=args.cache_dir,
    )

    logger.info("\nDownload complete!")
    logger.info("Use with llama.cpp or LM Studio for efficient inference")


if __name__ == "__main__":
    main()

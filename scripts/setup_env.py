#!/usr/bin/env python3
"""Setup script for Diogenes development environment.

This script verifies the development environment and helps with initial setup.
"""

import subprocess
import sys
from pathlib import Path


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def run_command(cmd: list[str], check: bool = False) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)
    except Exception as e:
        return False, str(e)


def check_python_version() -> bool:
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✓ Python version is compatible (>= 3.10)")
        return True
    else:
        print("✗ Python version must be 3.10 or higher")
        return False


def check_cuda() -> bool:
    """Check CUDA availability."""
    print_header("CUDA Check")

    success, output = run_command(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"]
    )

    if not success:
        print("✗ PyTorch not installed or CUDA check failed")
        return False

    if output == "True":
        print("✓ CUDA is available")

        # Get GPU info
        success, gpu_name = run_command(
            [sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"]
        )
        if success:
            print(f"  GPU: {gpu_name}")

        success, cuda_version = run_command(
            [sys.executable, "-c", "import torch; print(torch.version.cuda)"]
        )
        if success:
            print(f"  CUDA version: {cuda_version}")

        return True
    else:
        print("✗ CUDA not available")
        print("  Note: CPU-only mode is possible but not recommended for training")
        return True  # Still allow CPU mode


def check_dependencies() -> bool:
    """Check required dependencies."""
    print_header("Dependencies Check")

    required = [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
    ]

    optional = [
        "bitsandbytes",
        "wandb",
    ]

    all_ok = True

    print("Required packages:")
    for pkg in required:
        success, _ = run_command([sys.executable, "-m", "pip", "show", pkg])
        if success:
            print(f"  ✓ {pkg}")
        else:
            print(f"  ✗ {pkg} - MISSING")
            all_ok = False

    print("\nOptional packages:")
    for pkg in optional:
        success, _ = run_command([sys.executable, "-m", "pip", "show", pkg])
        if success:
            print(f"  ✓ {pkg}")
        else:
            print(f"  ○ {pkg} - not installed")

    return all_ok


def check_huggingface_login() -> bool:
    """Check HuggingFace CLI login."""
    print_header("HuggingFace Login Check")

    # Check for token file
    hf_token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]

    for path in hf_token_paths:
        if path.exists():
            print(f"✓ HuggingFace token found: {path}")
            print("  You should have access to Qwen3-32B")
            return True

    print("✗ No HuggingFace token found")
    print("\nTo login, run:")
    print("  huggingface-cli login")
    print("\nMake sure you have access to Qwen3-32B on HuggingFace")
    return False


def check_disk_space() -> bool:
    """Check available disk space."""
    print_header("Disk Space Check")

    import shutil

    # Get disk usage of current directory
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)

    print(f"Free disk space: {free_gb:.1f} GB")

    if free_gb >= 70:
        print("✓ Sufficient space for model (~70GB needed)")
        return True
    elif free_gb >= 30:
        print("○ May be sufficient with 4-bit quantization")
        return True
    else:
        print("✗ Insufficient disk space")
        return False


def check_directory_structure() -> bool:
    """Check project directory structure."""
    print_header("Directory Structure Check")

    required_dirs = [
        "src/diogenes",
        "configs",
        "datasets",
        "models",
        "scripts",
        "tests",
        "docs",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            all_ok = False

    return all_ok


def install_dependencies() -> None:
    """Install project dependencies."""
    print_header("Installing Dependencies")

    print("Installing project in editable mode...")
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=False,
    )

    if success:
        print("✓ Dependencies installed successfully")
    else:
        print("✗ Installation failed")
        print(output)


def main():
    print_header("Diogenes Environment Setup")

    # Run checks
    checks = {
        "Python Version": check_python_version(),
        "CUDA": check_cuda(),
        "Dependencies": check_dependencies(),
        "HuggingFace Login": check_huggingface_login(),
        "Disk Space": check_disk_space(),
        "Directory Structure": check_directory_structure(),
    }

    # Summary
    print_header("Summary")

    all_passed = all(checks.values())

    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    if all_passed:
        print("\n✓ All checks passed!")
        print("\nNext steps:")
        print("  1. If not logged in: huggingface-cli login")
        print("  2. Download model: python scripts/download_model.py")
        print("  3. Test inference: python scripts/test_inference.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies, run:")
        print("  pip install -e .")

    # Offer to install dependencies
    if not checks["Dependencies"]:
        response = input("\nInstall dependencies now? [y/N] ")
        if response.lower() == "y":
            install_dependencies()


if __name__ == "__main__":
    main()

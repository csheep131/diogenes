#!/usr/bin/env python3
"""
prepare_remote_machine.py

Script to prepare a remote machine for Diogenes SFT training.
Handles SSH connection, hardware validation, dependency installation,
and code/dataset synchronization.

Usage:
    python scripts/prepare_remote_machine.py --host <IP> --user <username>
    python scripts/prepare_remote_machine.py --config configs/remote_config.yaml
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional
import yaml


class RemoteMachinePreparer:
    """Prepares a remote machine for Diogenes training."""

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 22,
        project_dir: str = "/opt/diogenes",
        ssh_key: Optional[str] = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.project_dir = project_dir
        self.ssh_key = ssh_key
        self.ssh_cmd = self._build_ssh_cmd()

    def _build_ssh_cmd(self) -> list[str]:
        """Build base SSH command."""
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if self.ssh_key:
            cmd.extend(["-i", self.ssh_key])
        if self.port:
            cmd.extend(["-p", str(self.port)])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _build_scp_cmd(self) -> list[str]:
        """Build base SCP command."""
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-r"]
        if self.ssh_key:
            cmd.extend(["-i", self.ssh_key])
        if self.port:
            cmd.extend(["-P", str(self.port)])
        return cmd

    def run_remote(self, command: str, capture: bool = False) -> Optional[str]:
        """Execute a command on the remote machine."""
        full_cmd = self.ssh_cmd + [command]
        print(f"  → {command}")
        try:
            if capture:
                result = subprocess.run(
                    full_cmd, capture_output=True, text=True, check=True
                )
                return result.stdout.strip()
            else:
                subprocess.run(full_cmd, check=True)
                return None
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Command failed: {e}")
            raise

    def check_connection(self) -> bool:
        """Test SSH connection to remote machine."""
        print("\n[1/7] Testing SSH connection...")
        try:
            self.run_remote("echo 'Connection successful'")
            print("  ✓ SSH connection established")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to connect via SSH")
            return False

    def check_hardware(self) -> dict:
        """Check remote hardware requirements."""
        print("\n[2/7] Checking hardware requirements...")

        hardware = {}

        # GPU check
        gpu_output = self.run_remote(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            capture=True,
        )
        if gpu_output:
            gpus = gpu_output.split("\n")
            hardware["gpus"] = len(gpus)
            hardware["gpu_model"] = gpus[0].split(",")[0].strip()
            hardware["vram_gb"] = int(gpus[0].split(",")[1].strip().replace(" MiB", "")) // 1024
            print(f"  ✓ GPU: {hardware['gpus']}x {hardware['gpu_model']} ({hardware['vram_gb']}GB VRAM)")
        else:
            print("  ✗ No NVIDIA GPU detected")
            hardware["gpus"] = 0

        # CPU check
        cpu_output = self.run_remote("nproc --all", capture=True)
        if cpu_output:
            hardware["cpu_cores"] = int(cpu_output)
            print(f"  ✓ CPU: {hardware['cpu_cores']} cores")

        # RAM check
        ram_output = self.run_remote(
            "free -g | awk '/^Mem:/{print $2}'", capture=True
        )
        if ram_output:
            hardware["ram_gb"] = int(ram_output)
            print(f"  ✓ RAM: {hardware['ram_gb']}GB")

        # Disk check
        disk_output = self.run_remote(
            f"df -BG {self.project_dir} 2>/dev/null | awk 'NR==2{{gsub(/G/,\"\",$4); print $4}}'",
            capture=True,
        )
        if disk_output and disk_output.isdigit():
            hardware["disk_free_gb"] = int(disk_output)
            print(f"  ✓ Disk free: {hardware['disk_free_gb']}GB")
        else:
            disk_output = self.run_remote(
                "df -BG / | awk 'NR==2{{gsub(/G/,\"\",$4); print $4}}'", capture=True
            )
            hardware["disk_free_gb"] = int(disk_output) if disk_output.isdigit() else 0
            print(f"  ✓ Disk free: {hardware['disk_free_gb']}GB")

        # Validate requirements
        valid = True
        if hardware.get("vram_gb", 0) < 70:
            print("  ⚠ Warning: VRAM < 70GB, training may be slow or fail")
            valid = False
        if hardware.get("disk_free_gb", 0) < 50:
            print("  ⚠ Warning: Disk space < 50GB, may not be enough for datasets")
            valid = False

        return hardware

    def install_dependencies(self) -> bool:
        """Install required dependencies on remote machine."""
        print("\n[3/7] Installing dependencies...")

        # Check if Docker is installed
        docker_check = self.run_remote("docker --version", capture=True)
        if not docker_check:
            print("  → Installing Docker...")
            self.run_remote(
                "curl -fsSL https://get.docker.com | sudo sh",
            )
            self.run_remote("sudo usermod -aG docker $USER")

        # Check if NVIDIA Container Toolkit is installed
        nvidia_check = self.run_remote(
            "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi",
            capture=True,
        )
        if not nvidia_check:
            print("  → Installing NVIDIA Container Toolkit...")
            self.run_remote(
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
                "sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg"
            )
            self.run_remote(
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
                "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | "
                "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
            )
            self.run_remote("sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit")
            self.run_remote("sudo systemctl restart docker")

        # Check Python version
        python_check = self.run_remote("python3 --version", capture=True)
        if not python_check or "3.10" not in python_check and "3.11" not in python_check:
            print("  → Installing Python 3.10+...")
            self.run_remote("sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv")

        print("  ✓ Dependencies installed")
        return True

    def setup_project_directory(self) -> bool:
        """Create project directory structure on remote machine."""
        print("\n[4/7] Setting up project directory...")

        # Create directory
        self.run_remote(f"sudo mkdir -p {self.project_dir}")
        self.run_remote(f"sudo chown -R $USER:$USER {self.project_dir}")

        # Create subdirectories
        subdirs = ["datasets", "models", "checkpoints", "logs", "outputs"]
        for subdir in subdirs:
            self.run_remote(f"mkdir -p {self.project_dir}/{subdir}")

        print(f"  ✓ Project directory created at {self.project_dir}")
        return True

    def sync_code(self) -> bool:
        """Sync local code to remote machine."""
        print("\n[5/7] Syncing code to remote machine...")

        project_root = Path(__file__).parent.parent
        scp_cmd = self._build_scp_cmd()

        # Sync project files
        items_to_sync = [
            "src",
            "configs",
            "pyproject.toml",
            "scripts",
            "phasen",
        ]

        for item in items_to_sync:
            local_path = project_root / item
            if local_path.exists():
                print(f"  → Syncing {item}...")
                cmd = scp_cmd + [str(local_path), f"{self.user}@{self.host}:{self.project_dir}/"]
                subprocess.run(cmd, check=True)

        print("  ✓ Code synced")
        return True

    def sync_datasets(self, dataset_path: Optional[str] = None) -> bool:
        """Sync datasets to remote machine."""
        print("\n[6/7] Syncing datasets...")

        if dataset_path:
            scp_cmd = self._build_scp_cmd()
            print(f"  → Syncing dataset from {dataset_path}...")
            cmd = scp_cmd + [dataset_path, f"{self.user}@{self.host}:{self.project_dir}/datasets/"]
            subprocess.run(cmd, check=True)
            print("  ✓ Datasets synced")
        else:
            print("  ℹ No dataset path provided, skipping dataset sync")
            print("  → Datasets will be downloaded during training")

        return True

    def create_training_script(self) -> bool:
        """Create a convenience training launch script on remote machine."""
        print("\n[7/7] Creating training launch script...")

        training_script = '''#!/bin/bash
# Diogenes SFT Training Launcher
# Generated by prepare_remote_machine.py

set -e

PROJECT_DIR="/opt/diogenes"
cd "$PROJECT_DIR"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# For Unsloth support (optional, faster training)
# pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Start training
echo "Starting SFT training..."
python src/train_sft.py --config configs/config.yaml

echo "Training completed!"
'''

        # Write script to local temp file first
        temp_script = "/tmp/remote_train.sh"
        with open(temp_script, "w") as f:
            f.write(training_script)

        # Copy to remote
        scp_cmd = self._build_scp_cmd()
        cmd = scp_cmd + [temp_script, f"{self.user}@{self.host}:{self.project_dir}/train.sh"]
        subprocess.run(cmd, check=True)

        # Make executable
        self.run_remote(f"chmod +x {self.project_dir}/train.sh")

        # Cleanup
        os.remove(temp_script)

        print(f"  ✓ Training script created at {self.project_dir}/train.sh")
        return True

    def print_summary(self, hardware: dict) -> None:
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("REMOTE MACHINE PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Host: {self.user}@{self.host}")
        print(f"Project Directory: {self.project_dir}")
        print(f"\nHardware:")
        print(f"  - GPUs: {hardware.get('gpus', 0)}x {hardware.get('gpu_model', 'N/A')}")
        print(f"  - VRAM: {hardware.get('vram_gb', 0)}GB")
        print(f"  - CPU Cores: {hardware.get('cpu_cores', 'N/A')}")
        print(f"  - RAM: {hardware.get('ram_gb', 'N/A')}GB")
        print(f"  - Free Disk: {hardware.get('disk_free_gb', 'N/A')}GB")
        print(f"\nNext steps:")
        print(f"  1. SSH into remote: ssh {self.user}@{self.host}")
        print(f"  2. Navigate to project: cd {self.project_dir}")
        print(f"  3. Run training: ./train.sh")
        print(f"\nOr run training directly via SSH:")
        print(f"  ssh {self.user}@{self.host} 'cd {self.project_dir} && ./train.sh'")
        print("=" * 60)

    def prepare(self, sync_datasets: bool = False, dataset_path: Optional[str] = None) -> bool:
        """Run full preparation workflow."""
        try:
            if not self.check_connection():
                return False

            hardware = self.check_hardware()
            self.install_dependencies()
            self.setup_project_directory()
            self.sync_code()

            if sync_datasets:
                self.sync_datasets(dataset_path)

            self.create_training_script()
            self.print_summary(hardware)

            return True
        except Exception as e:
            print(f"\n✗ Preparation failed: {e}")
            return False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a remote machine for Diogenes SFT training"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (alternative to CLI args)",
    )
    parser.add_argument("--host", type=str, help="Remote machine hostname or IP")
    parser.add_argument("--user", type=str, default="root", help="SSH username")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument(
        "--project-dir",
        type=str,
        default="/opt/diogenes",
        help="Project directory on remote machine",
    )
    parser.add_argument("--ssh-key", type=str, help="Path to SSH private key")
    parser.add_argument(
        "--sync-datasets",
        action="store_true",
        help="Sync datasets to remote machine",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Local path to datasets directory",
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        config = load_config(args.config)
        host = config.get("host")
        user = config.get("user", "root")
        port = config.get("port", 22)
        project_dir = config.get("project_dir", "/opt/diogenes")
        ssh_key = config.get("ssh_key")
        sync_datasets = config.get("sync_datasets", False)
        dataset_path = config.get("dataset_path")
    else:
        host = args.host
        user = args.user
        port = args.port
        project_dir = args.project_dir
        ssh_key = args.ssh_key
        sync_datasets = args.sync_datasets
        dataset_path = args.dataset_path

    if not host:
        parser.error("--host is required (or use --config)")

    preparer = RemoteMachinePreparer(
        host=host,
        user=user,
        port=port,
        project_dir=project_dir,
        ssh_key=ssh_key,
    )

    success = preparer.prepare(
        sync_datasets=sync_datasets,
        dataset_path=dataset_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

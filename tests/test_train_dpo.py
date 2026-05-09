"""Tests for DPO training pipeline."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from diogenes.train_dpo import (
    DPOTrainingArguments,
)


class TestDPOTrainingArguments:
    """Test DPOTrainingArguments dataclass."""

    def test_default_values(self):
        """Test default argument values."""
        args = DPOTrainingArguments()
        assert args.model_name == "Qwen/Qwen3-0.6B"
        assert args.lora_rank == 32
        assert args.lora_alpha == 64
        assert args.lora_dropout == 0.05
        assert args.use_qlora is True
        assert args.num_train_epochs == 2
        assert args.learning_rate == 5e-7
        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 8
        assert args.seed == 42

    def test_dpo_specific_defaults(self):
        """Test DPO-specific default parameters."""
        args = DPOTrainingArguments()
        assert args.beta == 0.1
        assert args.label_smoothing == 0.0
        assert args.loss_type == "sigmoid"

    def test_sft_model_path_default_none(self):
        """Test that sft_model_path defaults to None."""
        args = DPOTrainingArguments()
        assert args.sft_model_path is None

    def test_custom_values(self):
        """Test custom argument values."""
        args = DPOTrainingArguments(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            sft_model_path="./models/sft_3b_test",
            lora_rank=16,
            lora_alpha=32,
            beta=0.2,
            loss_type="hinge",
            num_train_epochs=1,
            learning_rate=1e-6,
        )
        assert args.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert args.sft_model_path == "./models/sft_3b_test"
        assert args.lora_rank == 16
        assert args.lora_alpha == 32
        assert args.beta == 0.2
        assert args.loss_type == "hinge"
        assert args.num_train_epochs == 1
        assert args.learning_rate == 1e-6

    def test_target_modules_default(self):
        """Test default target modules."""
        args = DPOTrainingArguments()
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert args.target_modules == expected_modules

    def test_qlora_enabled_by_default(self):
        """Test QLoRA is enabled by default."""
        args = DPOTrainingArguments()
        assert args.use_qlora is True
        assert args.load_in_4bit is True
        assert args.bnb_4bit_use_double_quant is True

    def test_qlora_can_be_disabled(self):
        """Test that QLoRA can be disabled."""
        args = DPOTrainingArguments(use_qlora=False)
        assert args.use_qlora is False

    def test_loss_type_options(self):
        """Test different loss type options."""
        for loss_type in ["sigmoid", "hinge", "ipo", "kto_pair"]:
            args = DPOTrainingArguments(loss_type=loss_type)
            assert args.loss_type == loss_type

    def test_dataset_path_default(self):
        """Test default dataset path."""
        args = DPOTrainingArguments()
        assert args.dataset_path == "./datasets/dpo_dataset.jsonl"

    def test_max_seq_length(self):
        """Test max sequence length defaults."""
        args = DPOTrainingArguments()
        assert args.max_seq_length == 2048
        assert args.max_prompt_length == 512

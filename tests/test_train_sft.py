"""Tests for SFT training pipeline."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from diogenes.train_sft import (
    SFTTrainingArguments,
    format_sft_sample,
    create_qlora_config,
    create_lora_config,
)


class TestSFTTrainingArguments:
    """Test SFTTrainingArguments dataclass."""

    def test_default_values(self):
        """Test default argument values."""
        args = SFTTrainingArguments()
        assert args.model_name == "Qwen/Qwen3-0.6B"
        assert args.lora_rank == 32
        assert args.lora_alpha == 64
        assert args.lora_dropout == 0.05
        assert args.use_qlora is True
        assert args.num_train_epochs == 3
        assert args.learning_rate == 2e-4
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 4
        assert args.seed == 42

    def test_target_modules_default(self):
        """Test default target modules."""
        args = SFTTrainingArguments()
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert args.target_modules == expected_modules

    def test_custom_values(self):
        """Test custom argument values."""
        args = SFTTrainingArguments(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            lora_rank=16,
            lora_alpha=32,
            num_train_epochs=1,
            learning_rate=1e-4,
        )
        assert args.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert args.lora_rank == 16
        assert args.lora_alpha == 32
        assert args.num_train_epochs == 1
        assert args.learning_rate == 1e-4


class TestFormatSFTSample:
    """Test SFT sample formatting."""

    def test_format_sample_basic(self):
        """Test basic sample formatting."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted text>"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
        }

        sample = {
            "question": "What is 2+2?",
            "answer": "2+2 equals 4.",
            "category": "direct_answer",
        }

        result = format_sft_sample(sample, mock_tokenizer, max_length=512)

        # Verify chat template was called
        mock_tokenizer.apply_chat_template.assert_called_once()
        messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[1]["content"] == "What is 2+2?"
        assert messages[2]["content"] == "2+2 equals 4."

        # Verify labels are set
        assert "labels" in result
        assert result["labels"] == [1, 2, 3, 4, 5]

    def test_format_sample_contains_diogenes_system_prompt(self):
        """Test that system prompt mentions Diogenes and epistemic reliability."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted text>"
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        sample = {"question": "Q?", "answer": "A."}
        format_sft_sample(sample, mock_tokenizer)

        messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "Diogenes" in system_prompt
        assert "honest" in system_prompt.lower() or "epistemic" in system_prompt.lower()

    def test_format_sample_fallback_no_template(self):
        """Test fallback formatting when chat template fails."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("No template")
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        sample = {"question": "Q?", "answer": "A."}
        result = format_sft_sample(sample, mock_tokenizer)

        # Should still produce output (fallback path)
        assert "labels" in result


class TestQLoRAConfig:
    """Test QLoRA configuration creation."""

    def test_create_qlora_config_basic(self):
        """Test basic QLoRA config creation."""
        args = SFTTrainingArguments()

        with patch("diogenes.train_sft.BitsAndBytesConfig") as mock_bnb, \
             patch("diogenes.train_sft.LoraConfig") as mock_lora, \
             patch("diogenes.train_sft.torch") as mock_torch:
            mock_torch.float16 = torch.float16

            bnb_config, lora_config = create_qlora_config(args)

            # Verify BnB config was created with correct params
            mock_bnb.assert_called_once()
            bnb_kwargs = mock_bnb.call_args[1]
            assert bnb_kwargs["load_in_4bit"] is True
            assert bnb_kwargs["bnb_4bit_quant_type"] == "nf4"
            assert bnb_kwargs["bnb_4bit_use_double_quant"] is True

            # Verify LoRA config was created
            mock_lora.assert_called_once()
            lora_kwargs = mock_lora.call_args[1]
            assert lora_kwargs["r"] == 32
            assert lora_kwargs["lora_alpha"] == 64
            assert lora_kwargs["lora_dropout"] == 0.05
            assert lora_kwargs["inference_mode"] is False

    def test_create_qlora_config_custom_params(self):
        """Test QLoRA config with custom parameters."""
        args = SFTTrainingArguments(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bnb_4bit_compute_dtype="bfloat16",
        )

        with patch("diogenes.train_sft.BitsAndBytesConfig") as mock_bnb, \
             patch("diogenes.train_sft.LoraConfig") as mock_lora, \
             patch("diogenes.train_sft.torch") as mock_torch:
            mock_torch.float16 = torch.float16
            mock_torch.bfloat16 = torch.bfloat16

            create_qlora_config(args)

            lora_kwargs = mock_lora.call_args[1]
            assert lora_kwargs["r"] == 16
            assert lora_kwargs["lora_alpha"] == 32
            assert lora_kwargs["lora_dropout"] == 0.1


class TestLoRAConfig:
    """Test standard LoRA configuration."""

    def test_create_lora_config_basic(self):
        """Test basic LoRA config creation (without quantization)."""
        args = SFTTrainingArguments()

        with patch("diogenes.train_sft.LoraConfig") as mock_lora:
            lora_config = create_lora_config(args)

            mock_lora.assert_called_once()
            kwargs = mock_lora.call_args[1]
            assert kwargs["r"] == 32
            assert kwargs["lora_alpha"] == 64
            assert kwargs["inference_mode"] is False
            assert kwargs["bias"] == "none"

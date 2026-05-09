"""Tests for DiogenesModel class."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from diogenes.model import (
    DiogenesModel,
    EpistemicMode,
    load_base_model,
)


class TestDiogenesModelCreation:
    """Test DiogenesModel instantiation."""

    def test_create_model_with_mock_objects(self):
        """Test creating DiogenesModel with mock model and tokenizer."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"

        model = DiogenesModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_path="/fake/path",
        )

        assert model.model == mock_model
        assert model.tokenizer == mock_tokenizer
        assert model.model_path == "/fake/path"
        assert model.device == torch.device("cpu")
        assert model._lora_config is None

    def test_model_stores_device_correctly(self):
        """Test that device is correctly stored from model."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda:0")
        mock_tokenizer = MagicMock()

        model = DiogenesModel(model=mock_model, tokenizer=mock_tokenizer)
        assert model.device == torch.device("cuda:0")


class TestLoRAConfiguration:
    """Test LoRA configuration."""

    def test_configure_lora_default_modules(self):
        """Test LoRA configuration with default target modules."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_tokenizer = MagicMock()

        model = DiogenesModel(model=mock_model, tokenizer=mock_tokenizer)

        with patch("diogenes.model.prepare_model_for_kbit_training") as mock_prepare, \
             patch("diogenes.model.get_peft_model") as mock_peft:
            mock_prepare.return_value = mock_model
            mock_peft.return_value = mock_model

            model.configure_lora(rank=16, alpha=32)

            # Verify LoraConfig was created with correct params
            from peft import LoraConfig
            call_args = mock_peft.call_args
            lora_config = call_args[0][1]  # Second positional arg is LoraConfig
            assert lora_config.r == 16
            assert lora_config.lora_alpha == 32
            assert set(lora_config.target_modules) == {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            }
            assert model._lora_config is not None

    def test_configure_lora_custom_modules(self):
        """Test LoRA configuration with custom target modules."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_tokenizer = MagicMock()

        model = DiogenesModel(model=mock_model, tokenizer=mock_tokenizer)

        with patch("diogenes.model.prepare_model_for_kbit_training") as mock_prepare, \
             patch("diogenes.model.get_peft_model") as mock_peft:
            mock_prepare.return_value = mock_model
            mock_peft.return_value = mock_model

            model.configure_lora(
                rank=8,
                alpha=16,
                target_modules=["q_proj", "v_proj"],
            )

            call_args = mock_peft.call_args
            lora_config = call_args[0][1]
            assert lora_config.r == 8
            assert lora_config.lora_alpha == 16
            assert set(lora_config.target_modules) == {"q_proj", "v_proj"}


class TestModelSave:
    """Test model saving."""

    def test_save_model(self, tmp_path):
        """Test saving model and tokenizer."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_tokenizer = MagicMock()

        model = DiogenesModel(model=mock_model, tokenizer=mock_tokenizer)
        model.save(str(tmp_path / "test_model"))

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()


class TestLoadBaseModel:
    """Test load_base_model convenience function."""

    @patch("diogenes.model.DiogenesModel.from_pretrained")
    def test_load_base_model_default(self, mock_from_pretrained):
        """Test loading with default parameters."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        result = load_base_model()

        mock_from_pretrained.assert_called_once_with(
            model_name_or_path="Qwen/Qwen3-0.6B",
            cache_dir=None,
            attn_implementation="eager",
        )
        assert result == mock_model

    @patch("diogenes.model.DiogenesModel.from_pretrained")
    def test_load_base_model_custom(self, mock_from_pretrained):
        """Test loading with custom parameters."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        result = load_base_model(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            cache_dir="/tmp/cache",
        )

        mock_from_pretrained.assert_called_once_with(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            cache_dir="/tmp/cache",
            attn_implementation="eager",
        )


class TestEpistemicMode:
    """Additional tests for EpistemicMode enum."""

    def test_all_modes_exist(self):
        """Test that all 7 epistemic modes exist."""
        assert len(EpistemicMode) == 7

    def test_mode_values(self):
        """Test mode value strings."""
        assert EpistemicMode.DIRECT_ANSWER.value == "direct_answer"
        assert EpistemicMode.CAUTIOUS_LIMIT.value == "cautious_limit"
        assert EpistemicMode.ABSTAIN.value == "abstain"
        assert EpistemicMode.CLARIFY.value == "clarify"
        assert EpistemicMode.REJECT_PREMISE.value == "reject_premise"
        assert EpistemicMode.REQUEST_TOOL.value == "request_tool"
        assert EpistemicMode.PROBABILISTIC.value == "probabilistic"

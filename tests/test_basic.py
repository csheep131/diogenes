"""Basic tests for Diogenes model loading and inference."""

import pytest
import torch

from diogenes.model import EpistemicMode, DiogenesModel
from diogenes.inference import DiogenesInference, InferenceResult


class TestEpistemicMode:
    """Test epistemic mode enum."""

    def test_epistemic_modes_exist(self):
        """Test that all seven epistemic modes exist."""
        modes = [mode.value for mode in EpistemicMode]
        expected = [
            "direct_answer",
            "cautious_limit",
            "abstain",
            "clarify",
            "reject_premise",
            "request_tool",
            "probabilistic",
        ]
        assert sorted(modes) == sorted(expected)

    def test_epistemic_mode_from_string(self):
        """Test creating epistemic mode from string."""
        mode = EpistemicMode("direct_answer")
        assert mode == EpistemicMode.DIRECT_ANSWER


class TestInferenceResult:
    """Test InferenceResult dataclass."""

    def test_create_result(self):
        """Test creating an inference result."""
        result = InferenceResult(
            text="Test response",
            epistemic_mode=EpistemicMode.DIRECT_ANSWER,
            confidence=0.95,
            tokens=[1, 2, 3],
        )
        assert result.text == "Test response"
        assert result.epistemic_mode == EpistemicMode.DIRECT_ANSWER
        assert result.confidence == 0.95


class TestCudaAvailability:
    """Test CUDA availability."""

    def test_cuda_available(self):
        """Test that CUDA is available."""
        # This test will pass if CUDA is available, skip otherwise
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert torch.cuda.is_available()

    def test_cuda_device_count(self):
        """Test that at least one CUDA device is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert torch.cuda.device_count() >= 1

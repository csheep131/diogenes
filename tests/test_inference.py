"""Tests for DiogenesInference engine."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from diogenes.model import EpistemicMode
from diogenes.inference import DiogenesInference, InferenceResult


class TestInferenceResult:
    """Test InferenceResult dataclass."""

    def test_create_result_minimal(self):
        """Test creating InferenceResult with minimal fields."""
        result = InferenceResult(
            text="Hello",
            epistemic_mode=EpistemicMode.DIRECT_ANSWER,
            confidence=0.9,
            tokens=[1, 2, 3],
        )
        assert result.text == "Hello"
        assert result.epistemic_mode == EpistemicMode.DIRECT_ANSWER
        assert result.confidence == 0.9
        assert result.tokens == [1, 2, 3]
        assert result.logprobs is None

    def test_create_result_with_logprobs(self):
        """Test creating InferenceResult with logprobs."""
        result = InferenceResult(
            text="Hello",
            epistemic_mode=EpistemicMode.DIRECT_ANSWER,
            confidence=0.9,
            tokens=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
        )
        assert result.logprobs == [-0.1, -0.2, -0.3]


class TestEpistemicModeDetection:
    """Test epistemic mode detection heuristics."""

    @pytest.fixture
    def inference_engine(self):
        """Create inference engine with mock model."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.tokenizer = MagicMock()
        mock_model.tokenizer.pad_token_id = 0
        mock_model.tokenizer.eos_token_id = 1
        return DiogenesInference(mock_model)

    def test_detect_abstain(self, inference_engine):
        """Test abstention detection."""
        assert inference_engine._detect_epistemic_mode(
            "I don't know the answer", 0.9
        ) == EpistemicMode.ABSTAIN

        assert inference_engine._detect_epistemic_mode(
            "I'm not sure about that", 0.8
        ) == EpistemicMode.ABSTAIN

        assert inference_engine._detect_epistemic_mode(
            "I can't help with that", 0.95
        ) == EpistemicMode.ABSTAIN

    def test_detect_clarify(self, inference_engine):
        """Test clarification detection."""
        assert inference_engine._detect_epistemic_mode(
            "Could you clarify what you mean?", 0.7
        ) == EpistemicMode.CLARIFY

        assert inference_engine._detect_epistemic_mode(
            "Can you provide more details?", 0.8
        ) == EpistemicMode.CLARIFY

    def test_detect_reject_premise(self, inference_engine):
        """Test premise rejection detection."""
        assert inference_engine._detect_epistemic_mode(
            "Actually, that's not correct", 0.9
        ) == EpistemicMode.REJECT_PREMISE

        assert inference_engine._detect_epistemic_mode(
            "This premise is incorrect", 0.85
        ) == EpistemicMode.REJECT_PREMISE

    def test_detect_request_tool(self, inference_engine):
        """Test tool request detection."""
        assert inference_engine._detect_epistemic_mode(
            "I need to search for that", 0.7
        ) == EpistemicMode.REQUEST_TOOL

        assert inference_engine._detect_epistemic_mode(
            "I need access to external data", 0.6
        ) == EpistemicMode.REQUEST_TOOL

    def test_detect_probabilistic_low_confidence(self, inference_engine):
        """Test probabilistic mode for low confidence."""
        assert inference_engine._detect_epistemic_mode(
            "Maybe it could be this or that", 0.4
        ) == EpistemicMode.PROBABILISTIC

    def test_detect_cautious_limit_medium_confidence(self, inference_engine):
        """Test cautious limit mode for medium confidence."""
        assert inference_engine._detect_epistemic_mode(
            "It might be around 50%", 0.6
        ) == EpistemicMode.CAUTIOUS_LIMIT

    def test_detect_direct_answer_high_confidence(self, inference_engine):
        """Test direct answer for high confidence."""
        assert inference_engine._detect_epistemic_mode(
            "The capital of France is Paris.", 0.95
        ) == EpistemicMode.DIRECT_ANSWER

    def test_abstain_overrides_confidence(self, inference_engine):
        """Test that abstention patterns override confidence-based detection."""
        # Even with high confidence, "I don't know" should be ABSTAIN
        assert inference_engine._detect_epistemic_mode(
            "I don't know", 0.99
        ) == EpistemicMode.ABSTAIN


class TestInferenceEngineInit:
    """Test DiogenesInference initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        engine = DiogenesInference(mock_model)

        assert engine.default_max_length == 512
        assert engine.default_temperature == 0.7
        assert engine.device == torch.device("cpu")

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        engine = DiogenesInference(
            mock_model,
            default_max_length=1024,
            default_temperature=0.5,
        )

        assert engine.default_max_length == 1024
        assert engine.default_temperature == 0.5


class TestBatchGeneration:
    """Test batch generation."""

    def test_generate_batch_returns_same_length(self):
        """Test that batch generation returns results for all prompts."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.tokenizer = MagicMock()
        mock_model.tokenizer.pad_token_id = 0
        mock_model.tokenizer.eos_token_id = 1

        engine = DiogenesInference(mock_model)

        # Mock generate to return predictable results
        with patch.object(engine, "generate") as mock_gen:
            mock_gen.side_effect = lambda **kwargs: InferenceResult(
                text="test",
                epistemic_mode=EpistemicMode.DIRECT_ANSWER,
                confidence=0.9,
                tokens=[1],
            )

            prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
            results = engine.generate_batch(prompts)

            assert len(results) == 5
            assert mock_gen.call_count == 5

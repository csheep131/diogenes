"""Inference engine for Diogenes with epistemic mode detection."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import GenerationConfig

from diogenes.model import DiogenesModel, EpistemicMode


logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of an inference call."""

    text: str
    epistemic_mode: EpistemicMode
    confidence: float
    tokens: list[int]
    logprobs: Optional[list[float]] = None


class DiogenesInference:
    """Inference engine with epistemic mode detection."""

    def __init__(
        self,
        model: DiogenesModel,
        default_max_length: int = 512,
        default_temperature: float = 0.7,
    ):
        self.model = model
        self.device = model.device
        self.default_max_length = default_max_length
        self.default_temperature = default_temperature

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_logprobs: bool = False,
    ) -> InferenceResult:
        """Generate response with epistemic mode detection.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Use sampling
            return_logprobs: Return log probabilities

        Returns:
            InferenceResult with text, mode, and confidence
        """
        max_length = max_length or self.default_max_length
        temperature = temperature or self.default_temperature

        # Tokenize input
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length // 2,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # Configure generation
        generation_config = GenerationConfig(
            max_new_tokens=max_length - input_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
        )

        # Generate
        with torch.inference_mode():
            outputs = self.model.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode response
        generated_ids = outputs.sequences[0]
        generated_text = self.model.tokenizer.decode(
            generated_ids[input_length:],
            skip_special_tokens=True,
        )

        # Calculate confidence from logits if available
        confidence = 1.0
        logprobs = None
        if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
            logprobs = []
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                max_prob = probs.max().item()
                logprobs.append(torch.log(torch.tensor(max_prob)).item())
            confidence = sum(logprobs) / len(logprobs) if logprobs else 1.0

        # Detect epistemic mode
        epistemic_mode = self._detect_epistemic_mode(generated_text, confidence)

        return InferenceResult(
            text=generated_text,
            epistemic_mode=epistemic_mode,
            confidence=confidence,
            tokens=generated_ids.tolist(),
            logprobs=logprobs,
        )

    def _detect_epistemic_mode(self, text: str, confidence: float) -> EpistemicMode:
        """Detect epistemic mode from generated text.

        This is a simple heuristic-based detector. In production, this should
        be replaced with a trained epistemic routing head.

        Args:
            text: Generated text
            confidence: Model confidence score

        Returns:
            Detected EpistemicMode
        """
        text_lower = text.lower()

        # Check for abstention patterns
        abstain_patterns = [
            "i don't know",
            "i cannot",
            "i can't",
            "i'm not sure",
            "i am not sure",
            "i don't have information",
            "i don't have access",
            "unable to",
        ]
        if any(pattern in text_lower for pattern in abstain_patterns):
            return EpistemicMode.ABSTAIN

        # Check for clarification patterns
        clarify_patterns = [
            "could you clarify",
            "can you clarify",
            "please clarify",
            "what do you mean",
            "could you provide",
            "can you provide",
        ]
        if any(pattern in text_lower for pattern in clarify_patterns):
            return EpistemicMode.CLARIFY

        # Check for premise rejection patterns
        reject_patterns = [
            "this premise is incorrect",
            "this assumption is wrong",
            "that's not correct",
            "actually,",
            "this is based on a false",
        ]
        if any(pattern in text_lower for pattern in reject_patterns):
            return EpistemicMode.REJECT_PREMISE

        # Check for tool request patterns
        tool_patterns = [
            "i need to search",
            "let me check",
            "i should look up",
            "i need access to",
            "requires external",
        ]
        if any(pattern in text_lower for pattern in tool_patterns):
            return EpistemicMode.REQUEST_TOOL

        # Check confidence for cautious/probabilistic modes
        if confidence < 0.5:
            return EpistemicMode.PROBABILISTIC
        elif confidence < 0.7:
            return EpistemicMode.CAUTIOUS_LIMIT

        return EpistemicMode.DIRECT_ANSWER

    def generate_batch(
        self,
        prompts: list[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[InferenceResult]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            List of InferenceResult
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
            )
            results.append(result)
        return results


def run_inference_test(model: DiogenesModel) -> None:
    """Run basic inference tests.

    Args:
        model: Loaded DiogenesModel
    """
    inference = DiogenesInference(model)

    test_prompts = [
        "What is the capital of France?",
        "What is the meaning of life?",
        "I don't know, you tell me.",
    ]

    logger.info("Running inference tests...")
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        result = inference.generate(prompt, return_logprobs=True)
        logger.info(f"Mode: {result.epistemic_mode.value}")
        logger.info(f"Confidence: {result.confidence:.4f}")
        logger.info(f"Response: {result.text[:200]}...")

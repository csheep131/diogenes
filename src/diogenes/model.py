"""Model loading and configuration for Diogenes."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


logger = logging.getLogger(__name__)


class EpistemicMode(Enum):
    """Seven epistemic response modes."""

    DIRECT_ANSWER = "direct_answer"
    CAUTIOUS_LIMIT = "cautious_limit"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    REJECT_PREMISE = "reject_premise"
    REQUEST_TOOL = "request_tool"
    PROBABILISTIC = "probabilistic"


class DiogenesModel:
    """Diogenes model wrapper with epistemic capabilities."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.device = model.device
        self._lora_config: Optional[LoraConfig] = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        use_4bit: bool = False,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
    ) -> "DiogenesModel":
        """Load Diogenes model from pretrained weights.

        Args:
            model_name_or_path: HuggingFace model name or local path
            use_4bit: Use 4-bit quantization (QLoRA)
            trust_remote_code: Trust remote code from HF
            cache_dir: Cache directory for model weights

        Returns:
            DiogenesModel instance
        """
        logger.info(f"Loading model: {model_name_or_path}")

        # Configure quantization
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (QLoRA)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            padding_side="right",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if not use_4bit else torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )

        logger.info(f"Model loaded successfully on device: {model.device}")

        return cls(model=model, tokenizer=tokenizer, model_path=model_name_or_path)

    def configure_lora(
        self,
        rank: int = 32,
        alpha: int = 64,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        """Configure LoRA adapters for fine-tuning.

        Args:
            rank: LoRA rank
            alpha: LoRA alpha scaling
            target_modules: Modules to apply LoRA to
        """
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        self._lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self._lora_config)

        logger.info(f"LoRA configured: rank={rank}, alpha={alpha}")
        self.model.print_trainable_parameters()

    def save(self, save_path: Union[str, Path]) -> None:
        """Save model and tokenizer.

        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "DiogenesModel":
        """Load model from local directory.

        Args:
            model_path: Path to model directory

        Returns:
            DiogenesModel instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        return cls.from_pretrained(str(model_path), use_4bit=False)


def load_base_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: Optional[str] = None,
) -> DiogenesModel:
    """Convenience function to load the Qwen3 base model.

    Args:
        model_name: Model name on HuggingFace. 
                    Recommended for Phase 0: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
        cache_dir: Cache directory

    Returns:
        Loaded DiogenesModel
    """
    return DiogenesModel.from_pretrained(
        model_name_or_path=model_name,
        cache_dir=cache_dir,
    )

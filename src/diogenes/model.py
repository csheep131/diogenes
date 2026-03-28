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

# Optional import for AttnRes (only if installed)
try:
    from diogenes.attnres import AttnResConfig, AttnResVariant, AttnResWrapper
    ATTNRES_AVAILABLE = True
except ImportError:
    ATTNRES_AVAILABLE = False


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
        attn_implementation: str = "eager",
    ) -> "DiogenesModel":
        """Load Diogenes model from pretrained weights.

        Args:
            model_name_or_path: HuggingFace model name or local path
            use_4bit: Use 4-bit quantization (QLoRA)
            trust_remote_code: Trust remote code from HF
            cache_dir: Cache directory for model weights
            attn_implementation: Attention implementation to use.
                                 Options: "eager", "flash_attention_2", "sdpa"
                                 Default "eager" works on all systems.

        Returns:
            DiogenesModel instance
        """
        logger.info(f"Loading model: {model_name_or_path}")
        logger.info(f"Attention implementation: {attn_implementation}")

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
            attn_implementation=attn_implementation,
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
    
    def enable_attnres(
        self,
        variant: str = "full",
        num_blocks: int = 8,
        apply_to: str = "both",
        init_scale: float = 0.02,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        cache_on_cpu: bool = False,
    ) -> None:
        """Enable Attention Residuals (AttnRes).
        
        AttnRes replaces fixed-weight residual connections with learned,
        input-dependent attention weights over preceding layer outputs.
        
        Based on: arXiv:2603.15031 (Attention Residuals)
        
        Args:
            variant: AttnRes variant ("full" or "block")
                - "full": Each layer attends over all preceding outputs
                - "block": Layers grouped into blocks, attends within block only
            num_blocks: Number of blocks for Block variant (default: 8)
            apply_to: Where to apply AttnRes ("attention", "mlp", or "both")
            init_scale: Initialization scale for query weights
            dropout: Dropout rate for attention weights
            use_layer_norm: Apply LayerNorm before attention computation
            cache_on_cpu: Cache layer outputs on CPU to save VRAM
        
        Raises:
            ImportError: If attnres module is not available
            ValueError: If variant is invalid
        
        Example:
            >>> model = DiogenesModel.from_pretrained("Qwen/Qwen3-0.6B")
            >>> model.enable_attnres(variant="full")
            >>> # Or use Block variant for memory efficiency
            >>> model.enable_attnres(variant="block", num_blocks=8)
        """
        if not ATTNRES_AVAILABLE:
            raise ImportError(
                "AttnRes module not available. Install with: "
                "pip install -e '.[attnres]' (if available) or ensure "
                "src/diogenes/attnres/ exists"
            )
        
        logger.info(f"Enabling AttnRes: variant={variant}, num_blocks={num_blocks}")
        
        # Map variant string to enum
        variant_enum = AttnResVariant(variant)
        
        config = AttnResConfig(
            variant=variant_enum,
            hidden_size=self.model.config.hidden_size,
            num_layers=self.model.config.num_hidden_layers,
            num_blocks=num_blocks,
            init_scale=init_scale,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            cache_on_cpu=cache_on_cpu,
            apply_to=apply_to,
        )
        
        wrapper = AttnResWrapper(self.model, config, apply_to=apply_to)
        self.model = wrapper.wrap()
        
        logger.info(
            f"AttnRes enabled successfully: variant={variant}, "
            f"layers={config.num_layers}, hidden={config.hidden_size}"
        )

    @classmethod
    def load(cls, model_path: Union[str, Path], attn_implementation: str = "eager") -> "DiogenesModel":
        """Load model from local directory.

        Args:
            model_path: Path to model directory
            attn_implementation: Attention implementation to use

        Returns:
            DiogenesModel instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        return cls.from_pretrained(str(model_path), use_4bit=False, attn_implementation=attn_implementation)


def load_base_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: Optional[str] = None,
    attn_implementation: str = "eager",
) -> DiogenesModel:
    """Convenience function to load the Qwen3 base model.

    Args:
        model_name: Model name on HuggingFace.
                    Recommended for Phase 0: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
        cache_dir: Cache directory
        attn_implementation: Attention implementation to use

    Returns:
        Loaded DiogenesModel
    """
    return DiogenesModel.from_pretrained(
        model_name_or_path=model_name,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    )

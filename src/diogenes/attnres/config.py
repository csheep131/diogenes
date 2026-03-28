"""Configuration for Attention Residuals."""

from dataclasses import dataclass
from enum import Enum


class AttnResVariant(Enum):
    """AttnRes variant selection."""
    
    FULL = "full"
    """Full AttnRes: Each layer attends over all preceding outputs.
    
    Memory complexity: O(L·d) where L = num_layers, d = hidden_size
    Best for: Maximum flexibility, research/experimentation
    """
    
    BLOCK = "block"
    """Block AttnRes: Layers grouped into blocks.
    
    Memory complexity: O(N·d) where N = num_blocks << L
    Best for: Production deployment, memory-constrained environments
    """


@dataclass
class AttnResConfig:
    """Configuration for Attention Residuals.
    
    Attributes:
        variant: AttnRes variant (full or block)
        hidden_size: Hidden dimension (d_model)
        num_layers: Total number of transformer layers
        num_blocks: Number of blocks for Block variant (default: 8)
        init_scale: Initialization scale for query weights
        use_layer_norm: Apply LayerNorm before attention computation
        dropout: Dropout rate for attention weights
        cache_on_cpu: Cache large tensors on CPU to save VRAM
        apply_to: Where to apply AttnRes ("attention", "mlp", or "both")
    
    Example:
        >>> config = AttnResConfig(
        ...     variant=AttnResVariant.FULL,
        ...     hidden_size=2048,
        ...     num_layers=32,
        ...     init_scale=0.02,
        ... )
        >>> print(config.layers_per_block)  # Only for BLOCK variant
        4
    """
    
    variant: AttnResVariant = AttnResVariant.FULL
    hidden_size: int = 2048
    num_layers: int = 32
    num_blocks: int = 8
    init_scale: float = 0.02
    use_layer_norm: bool = True
    dropout: float = 0.0
    cache_on_cpu: bool = False
    apply_to: str = "both"  # "attention", "mlp", or "both"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.variant, AttnResVariant):
            raise ValueError(f"variant must be AttnResVariant, got {self.variant}")
        
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        
        if self.init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {self.init_scale}")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        
        if self.apply_to not in ["attention", "mlp", "both"]:
            raise ValueError(f"apply_to must be 'attention', 'mlp', or 'both', got {self.apply_to}")
        
        # Validate block divisibility for BLOCK variant
        if self.variant == AttnResVariant.BLOCK:
            if self.num_layers % self.num_blocks != 0:
                raise ValueError(
                    f"For BLOCK variant, num_layers ({self.num_layers}) must be "
                    f"divisible by num_blocks ({self.num_blocks})"
                )
    
    @property
    def layers_per_block(self) -> int:
        """Number of layers per block (BLOCK variant only).
        
        Returns:
            Number of layers in each block
            
        Raises:
            ValueError: If variant is not BLOCK
        """
        if self.variant != AttnResVariant.BLOCK:
            raise ValueError("layers_per_block is only defined for BLOCK variant")
        return self.num_layers // self.num_blocks
    
    def __str__(self) -> str:
        """String representation of config."""
        return (
            f"AttnResConfig(variant={self.variant.value}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"num_blocks={self.num_blocks}, "
            f"init_scale={self.init_scale}, "
            f"dropout={self.dropout})"
        )

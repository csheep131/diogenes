"""Core classes for Attention Residuals."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from .config import AttnResConfig, AttnResVariant
from .cache import AttnResCache


class AttnResBase(ABC, nn.Module):
    """Base class for Attention Residuals.
    
    Implements the core attention mechanism over preceding layer outputs.
    Subclasses define how layers are grouped (full vs block).
    
    The AttnRes mechanism replaces standard fixed-weight residual connections
    with learned, input-dependent attention weights:
    
        h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
    
    where α_{i→l} = softmax(q_l · k_i / √d)
    
    Attributes:
        config: AttnRes configuration
        hidden_size: Hidden dimension size
        key_proj: Linear projection for computing keys
        norm: Optional layer normalization
    
    Example:
        >>> config = AttnResConfig(variant=AttnResVariant.FULL, hidden_size=128, num_layers=4)
        >>> attn_res = FullAttnRes(config)
        >>> cache = AttnResCache()
        >>> # Store preceding outputs
        >>> for i in range(4):
        ...     cache.store_output(i, torch.randn(2, 16, 128))
        >>> # Apply AttnRes
        >>> output = torch.randn(2, 16, 128)
        >>> result = attn_res(output, cache, layer_idx=3)
    """
    
    def __init__(self, config: AttnResConfig):
        """Initialize AttnRes base module.
        
        Args:
            config: AttnRes configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Key projection for attention computation
        # Projects layer outputs to key space for attention calculation
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Optional layer normalization
        if config.use_layer_norm:
            self.norm = nn.LayerNorm(config.hidden_size)
        else:
            self.norm = nn.Identity()
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
    
    @abstractmethod
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        keys: List[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention weights for layer aggregation.
        
        Args:
            query: Query tensor [batch, seq_len, hidden_size] or [1, 1, hidden_size]
            keys: List of key tensors from preceding layers, each [batch, seq_len, hidden_size]
            layer_idx: Current layer index
            
        Returns:
            Attention weights [batch, seq_len, num_preceding]
        """
        pass
    
    @abstractmethod
    def get_preceding_outputs(
        self,
        cache: AttnResCache,
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Get preceding layer outputs from cache.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            List of preceding output tensors
        """
        pass
    
    def forward(
        self,
        current_output: torch.Tensor,
        cache: AttnResCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Apply attention residual connection.
        
        Args:
            current_output: Current layer output [batch, seq_len, hidden_size]
            cache: Cache for storing/retrieving layer outputs
            layer_idx: Current layer index
            
        Returns:
            Output with attention residual applied [batch, seq_len, hidden_size]
        """
        # Normalize current output
        if self.config.use_layer_norm:
            current_norm = self.norm(current_output)
        else:
            current_norm = current_output
        
        # Get preceding outputs
        preceding_outputs = self.get_preceding_outputs(cache, layer_idx)
        
        if len(preceding_outputs) == 0:
            # First layer: no preceding outputs, return as-is
            return current_output
        
        # Compute keys for all preceding outputs
        keys = [self.key_proj(out) for out in preceding_outputs]
        
        # Compute attention weights
        # Query is learned pseudo-query per layer
        query = self.get_layer_query(layer_idx)
        attn_weights = self.compute_attention_weights(query, keys, layer_idx)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of preceding outputs
        aggregated = self.aggregate_outputs(preceding_outputs, attn_weights)
        
        # Combine with current output using learned mixing
        # AttnRes replaces standard residual, not adds to it
        return aggregated
    
    def get_layer_query(self, layer_idx: int) -> torch.Tensor:
        """Get learned pseudo-query for specific layer.
        
        Each layer has its own learned query vector that determines
        how it attends to preceding layer outputs.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Query tensor [1, 1, hidden_size]
        """
        query_name = f'query_{layer_idx}'
        query = getattr(self, query_name, None)
        if query is None:
            # Initialize query parameter
            query = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * self.config.init_scale
            )
            self.register_parameter(query_name, query)
        return query
    
    def aggregate_outputs(
        self,
        outputs: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted sum of outputs.
        
        Args:
            outputs: List of output tensors, each [batch, seq_len, hidden_size]
            weights: Attention weights [batch, seq_len, num_outputs]
            
        Returns:
            Weighted sum [batch, seq_len, hidden_size]
        """
        # Stack outputs: [batch, seq_len, num_outputs, hidden_size]
        stacked = torch.stack(outputs, dim=2)
        
        # Expand weights for broadcasting: [batch, seq_len, num_outputs, 1]
        weights_expanded = weights.unsqueeze(-1)
        
        # Weighted sum over num_outputs dimension
        aggregated = (stacked * weights_expanded).sum(dim=2)
        
        return aggregated
    
    def get_attention_weights_for_analysis(
        self,
        cache: AttnResCache,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Get attention weights for analysis/debugging.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            Attention weights [batch, seq_len, num_preceding] or None
        """
        preceding_outputs = self.get_preceding_outputs(cache, layer_idx)
        
        if len(preceding_outputs) == 0:
            return None
        
        keys = [self.key_proj(out) for out in preceding_outputs]
        query = self.get_layer_query(layer_idx)
        weights = self.compute_attention_weights(query, keys, layer_idx)
        
        return weights


class AttnResWrapper:
    """Wrapper for applying AttnRes to transformer models.
    
    This is the main integration point. It wraps an existing model
    and intercepts transformer block outputs to apply AttnRes.
    
    The wrapper uses forward hooks to intercept layer outputs,
    apply AttnRes, and return the modified output.
    
    Attributes:
        model: The model to wrap
        config: AttnRes configuration
        apply_to: Where to apply AttnRes ("attention", "mlp", or "both")
        attn_res: AttnRes module (FullAttnRes or BlockAttnRes)
        cache: Cache for layer outputs
        hooks: Registered forward hooks
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> config = AttnResConfig(
        ...     variant=AttnResVariant.FULL,
        ...     hidden_size=2048,
        ...     num_layers=32,
        ... )
        >>> wrapper = AttnResWrapper(model, config)
        >>> wrapped_model = wrapper.wrap()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: AttnResConfig,
        apply_to: str = "both",
    ):
        """Initialize AttnRes wrapper.
        
        Args:
            model: Model to wrap
            config: AttnRes configuration
            apply_to: Where to apply AttnRes ("attention", "mlp", or "both")
        """
        self.model = model
        self.config = config
        self.apply_to = apply_to
        
        # Import here to avoid circular dependency
        if config.variant == AttnResVariant.FULL:
            from .full import FullAttnRes
            self.attn_res = FullAttnRes(config)
        elif config.variant == AttnResVariant.BLOCK:
            from .block import BlockAttnRes
            self.attn_res = BlockAttnRes(config)
        else:
            raise ValueError(f"Unknown variant: {config.variant}")
        
        # Cache for layer outputs
        self.cache = AttnResCache(
            cpu_offload=config.cache_on_cpu,
            device=next(model.parameters()).device,
        )
        
        # Store hooks to prevent garbage collection
        self.hooks: List = []
    
    def wrap(self) -> nn.Module:
        """Apply AttnRes wrapper to model.
        
        Returns:
            Wrapped model with AttnRes applied
        """
        # Find transformer layers
        transformer_layers = self._find_transformer_layers()
        
        if not transformer_layers:
            raise RuntimeError(
                "Could not find transformer layers in model. "
                "Supported patterns: model.model.layers, model.transformer.h, model.layers"
            )
        
        # Install hooks on each layer
        self._install_hooks(transformer_layers)
        
        return self.model
    
    def unwrap(self) -> None:
        """Remove AttnRes wrapper from model.
        
        This removes all forward hooks and clears the cache.
        """
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear cache
        self.cache.clear()
    
    def _find_transformer_layers(self) -> List[nn.Module]:
        """Find transformer block layers in model.
        
        Returns:
            List of transformer block modules
            
        Note:
            Tries common layer naming patterns:
            - model.model.layers (Qwen, Llama)
            - model.transformer.h (GPT-2 style)
            - model.layers (direct)
        """
        # Pattern 1: model.model.layers (Qwen, Llama)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        
        # Pattern 2: model.transformer.h (GPT-2 style)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return list(self.model.transformer.h)
        
        # Pattern 3: Direct layers attribute
        if hasattr(self.model, 'layers'):
            return list(self.model.layers)
        
        return []
    
    def _install_hooks(self, layers: List[nn.Module]) -> None:
        """Install forward hooks on transformer layers.
        
        Args:
            layers: List of transformer layer modules
        """
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._make_hook(idx),
                always_call=True,
            )
            self.hooks.append(hook)
    
    def _make_hook(self, layer_idx: int):
        """Create forward hook for layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Forward hook function
        """
        def hook(module, inputs, output):
            # Clear cache at start of new forward pass (layer 0)
            if layer_idx == 0:
                self.cache.clear()
            
            # Store current output in cache
            self.cache.store_output(layer_idx, output)
            
            # Apply AttnRes
            attn_res_output = self.attn_res(
                current_output=output,
                cache=self.cache,
                layer_idx=layer_idx,
            )
            
            return attn_res_output
        
        return hook
    
    def reset_cache(self) -> None:
        """Reset cache between forward passes."""
        self.cache.clear()
    
    def get_attention_weights(
        self,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Get attention weights for specific layer (for analysis).
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Attention weights or None if not available
        """
        return self.attn_res.get_attention_weights_for_analysis(
            self.cache,
            layer_idx,
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks."""
        self.unwrap()

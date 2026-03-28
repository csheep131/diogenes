"""Full Attention Residuals implementation.

Full AttnRes: Each layer attends over ALL preceding layer outputs.

Memory complexity: O(L·d) where L = num_layers, d = hidden_size
Best for: Maximum flexibility, research/experimentation
"""

import torch
from typing import List
from .config import AttnResConfig
from .core import AttnResBase
from .cache import AttnResCache


class FullAttnRes(AttnResBase):
    """Full Attention Residuals.
    
    Each layer attends over ALL preceding layer outputs.
    
    The attention mechanism computes:
        h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
        α_{i→l} = softmax(q_l · k_i / √d)
    
    where:
        - v_i: Output of layer i (value)
        - k_i: Key projection of layer i output
        - q_l: Learned pseudo-query for layer l
        - d: Hidden dimension
    
    Memory complexity: O(L·d) where L = num_layers
    Compute complexity: O(L²·b·s·d) for attention computation
    
    Example:
        >>> config = AttnResConfig(
        ...     variant=AttnResVariant.FULL,
        ...     hidden_size=128,
        ...     num_layers=4,
        ... )
        >>> attn_res = FullAttnRes(config)
        >>> cache = AttnResCache()
        >>> 
        >>> # Simulate forward pass through layers
        >>> batch_size, seq_len = 2, 16
        >>> for i in range(4):
        ...     output = torch.randn(batch_size, seq_len, 128)
        ...     cache.store_output(i, output)
        ... 
        >>> # Apply AttnRes to layer 3
        >>> last_output = torch.randn(batch_size, seq_len, 128)
        >>> result = attn_res(last_output, cache, layer_idx=3)
        >>> assert result.shape == last_output.shape
    """
    
    def __init__(self, config: AttnResConfig):
        """Initialize Full AttnRes.
        
        Args:
            config: AttnRes configuration
        """
        super().__init__(config)
        
        # Create learned queries for all layers
        # Each layer has its own pseudo-query that determines how it attends
        # to preceding layer outputs
        for i in range(config.num_layers):
            query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * config.init_scale
            )
            self.register_parameter(f'query_{i}', query)
    
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        keys: List[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute softmax attention over all preceding keys.
        
        Computes attention weights using scaled dot-product attention:
            α_{i→l} = softmax(q_l · k_i / √d)
        
        Args:
            query: Learned query [1, 1, hidden_size]
            keys: List of key tensors, each [batch, seq_len, hidden_size]
            layer_idx: Current layer index (not used in Full variant)
            
        Returns:
            Attention weights [batch, seq_len, num_preceding]
            
        Note:
            The softmax is computed over the num_preceding dimension,
            ensuring weights sum to 1 for each position in the sequence.
        """
        num_preceding = len(keys)
        batch_size = keys[0].shape[0]
        seq_len = keys[0].shape[1]
        
        # Stack keys: [batch, seq_len, num_preceding, hidden_size]
        keys_stacked = torch.stack(keys, dim=2)
        
        # Compute attention scores: query · keys
        # query: [1, 1, hidden_size] -> [1, 1, 1, hidden_size]
        # keys: [batch, seq_len, num_preceding, hidden_size]
        # scores: [batch, seq_len, num_preceding]
        query_expanded = query.unsqueeze(2)  # [1, 1, 1, hidden_size]
        scores = (query_expanded * keys_stacked).sum(dim=-1) / (
            self.hidden_size ** 0.5
        )
        
        # Softmax over preceding layers dimension
        # weights: [batch, seq_len, num_preceding]
        weights = torch.softmax(scores, dim=2)
        
        return weights
    
    def get_preceding_outputs(
        self,
        cache: AttnResCache,
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Get ALL preceding layer outputs.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            List of all preceding output tensors [h_0, h_1, ..., h_{layer_idx-1}]
            
        Example:
            >>> cache = AttnResCache()
            >>> for i in range(5):
            ...     cache.store_output(i, torch.randn(2, 16, 128))
            >>> preceding = attn_res.get_preceding_outputs(cache, 5)
            >>> len(preceding)
            5
        """
        return cache.get_all_preceding(layer_idx)
    
    def get_num_preceding(self, layer_idx: int) -> int:
        """Get number of preceding layers for a given layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Number of preceding layers (equal to layer_idx for Full variant)
        """
        return layer_idx
    
    def get_effective_receptive_field(self, layer_idx: int) -> int:
        """Get effective receptive field size for a layer.
        
        For Full AttnRes, each layer can attend to all preceding layers,
        so the receptive field grows linearly with depth.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Receptive field size (layer_idx + 1, including current layer)
        """
        return layer_idx + 1

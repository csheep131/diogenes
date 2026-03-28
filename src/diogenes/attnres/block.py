"""Block Attention Residuals implementation.

Block AttnRes: Layers grouped into blocks. Each layer attends over outputs within its block.

Memory complexity: O(N·d) where N = num_blocks << L
Best for: Production deployment, memory-constrained environments
"""

import torch
import torch.nn as nn
from typing import List
from .config import AttnResConfig
from .core import AttnResBase
from .cache import AttnResCache


class BlockAttnRes(AttnResBase):
    """Block Attention Residuals.
    
    Layers are grouped into blocks. Each layer attends over outputs
    within its own block only, significantly reducing memory usage.
    
    Example configuration:
        - 32 layers / 8 blocks = 4 layers per block
        - Block 0: layers 0-3
        - Block 1: layers 4-7
        - Block 2: layers 8-11
        - ...
    
    The attention mechanism computes:
        h_l = Σ_{i=block_start}^{l-1} α_{i→l} · v_i
        α_{i→l} = softmax(q_l · k_i / √d)
    
    where:
        - block_start = (layer_idx // layers_per_block) * layers_per_block
        - v_i: Output of layer i within the same block
        - k_i: Key projection of layer i output
        - q_l: Learned pseudo-query for layer l
    
    Memory complexity: O(N·d) where N = num_blocks (vs O(L·d) for Full)
    Compute complexity: O(L·(L/N)·b·s·d) = O(L²·b·s·d/N)
    
    Example:
        >>> config = AttnResConfig(
        ...     variant=AttnResVariant.BLOCK,
        ...     hidden_size=128,
        ...     num_layers=8,
        ...     num_blocks=2,  # 4 layers per block
        ... )
        >>> attn_res = BlockAttnRes(config)
        >>> cache = AttnResCache()
        >>> 
        >>> # Simulate forward pass through layers
        >>> batch_size, seq_len = 2, 16
        >>> for i in range(8):
        ...     output = torch.randn(batch_size, seq_len, 128)
        ...     cache.store_output(i, output)
        ... 
        >>> # Apply AttnRes to layer 5 (in block 1, attends to layer 4 only)
        >>> last_output = torch.randn(batch_size, seq_len, 128)
        >>> result = attn_res(last_output, cache, layer_idx=5)
        >>> assert result.shape == last_output.shape
    """
    
    def __init__(self, config: AttnResConfig):
        """Initialize Block AttnRes.
        
        Args:
            config: AttnRes configuration
            
        Raises:
            ValueError: If num_layers is not divisible by num_blocks
        """
        super().__init__(config)
        
        self.layers_per_block = config.layers_per_block
        self.num_blocks = config.num_blocks
        
        # Create learned queries for all layers
        # Each layer has its own pseudo-query
        for i in range(config.num_layers):
            query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * config.init_scale
            )
            self.register_parameter(f'query_{i}', query)
        
        # Optional: Block-level queries for inter-block attention (future enhancement)
        # Currently not used, but available for extended variants
        for b in range(config.num_blocks):
            block_query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * config.init_scale
            )
            self.register_parameter(f'block_query_{b}', block_query)
    
    def get_block_idx(self, layer_idx: int) -> int:
        """Get block index for a layer.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            
        Returns:
            Block index (0 to num_blocks-1)
            
        Example:
            >>> config = AttnResConfig(variant=AttnResVariant.BLOCK, num_layers=32, num_blocks=8)
            >>> attn_res = BlockAttnRes(config)
            >>> attn_res.get_block_idx(0)
            0
            >>> attn_res.get_block_idx(3)
            0
            >>> attn_res.get_block_idx(4)
            1
            >>> attn_res.get_block_idx(31)
            7
        """
        return layer_idx // self.layers_per_block
    
    def get_block_start_layer(self, block_idx: int) -> int:
        """Get starting layer index for a block.
        
        Args:
            block_idx: Block index (0 to num_blocks-1)
            
        Returns:
            Starting layer index (inclusive)
            
        Example:
            >>> config = AttnResConfig(variant=AttnResVariant.BLOCK, num_layers=32, num_blocks=8)
            >>> attn_res = BlockAttnRes(config)
            >>> attn_res.get_block_start_layer(0)
            0
            >>> attn_res.get_block_start_layer(1)
            4
            >>> attn_res.get_block_start_layer(7)
            28
        """
        return block_idx * self.layers_per_block
    
    def get_block_end_layer(self, block_idx: int) -> int:
        """Get ending layer index for a block.
        
        Args:
            block_idx: Block index (0 to num_blocks-1)
            
        Returns:
            Ending layer index (exclusive)
        """
        return (block_idx + 1) * self.layers_per_block
    
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        keys: List[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute softmax attention over block keys.
        
        Computes attention weights using scaled dot-product attention:
            α_{i→l} = softmax(q_l · k_i / √d)
        
        Args:
            query: Learned query [1, 1, hidden_size]
            keys: List of key tensors within block, each [batch, seq_len, hidden_size]
            layer_idx: Current layer index
            
        Returns:
            Attention weights [batch, seq_len, num_preceding_in_block]
            
        Note:
            The softmax is computed over the num_preceding_in_block dimension,
            ensuring weights sum to 1 for each position in the sequence.
        """
        num_preceding = len(keys)
        
        if num_preceding == 0:
            # No preceding layers in block, return empty weights
            batch_size = query.shape[0]
            seq_len = query.shape[1]
            return torch.zeros(batch_size, seq_len, 0, device=query.device)
        
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
        
        # Softmax over preceding layers within block
        # weights: [batch, seq_len, num_preceding]
        weights = torch.softmax(scores, dim=2)
        
        return weights
    
    def get_preceding_outputs(
        self,
        cache: AttnResCache,
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Get preceding outputs within the same block.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            List of preceding output tensors within block
            [h_{block_start}, ..., h_{layer_idx-1}]
            
        Example:
            >>> config = AttnResConfig(variant=AttnResVariant.BLOCK, num_layers=8, num_blocks=2)
            >>> attn_res = BlockAttnRes(config)
            >>> cache = AttnResCache()
            >>> for i in range(8):
            ...     cache.store_output(i, torch.randn(2, 16, 128))
            >>> 
            >>> # Layer 5 is in block 1 (layers 4-7), attends to layer 4 only
            >>> preceding = attn_res.get_preceding_outputs(cache, 5)
            >>> len(preceding)
            1
            >>> 
            >>> # Layer 7 is in block 1, attends to layers 4, 5, 6
            >>> preceding = attn_res.get_preceding_outputs(cache, 7)
            >>> len(preceding)
            3
        """
        block_idx = self.get_block_idx(layer_idx)
        block_start = self.get_block_start_layer(block_idx)
        
        # Get outputs from block_start to layer_idx-1
        outputs = []
        for i in range(block_start, layer_idx):
            output = cache.get_layer_output(i)
            if output is not None:
                outputs.append(output)
        
        return outputs
    
    def get_num_preceding(self, layer_idx: int) -> int:
        """Get number of preceding layers within block.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Number of preceding layers within the same block
        """
        block_idx = self.get_block_idx(layer_idx)
        block_start = self.get_block_start_layer(block_idx)
        
        # Position within block (0 to layers_per_block-1)
        position_in_block = layer_idx - block_start
        
        return position_in_block
    
    def get_effective_receptive_field(self, layer_idx: int) -> int:
        """Get effective receptive field size for a layer.
        
        For Block AttnRes, each layer can only attend to layers within
        its block, so the receptive field is limited to the block size.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Receptive field size (position_in_block + 1, max = layers_per_block)
        """
        return self.get_num_preceding(layer_idx) + 1
    
    def get_block_attention_summary(
        self,
        cache: AttnResCache,
        block_idx: int,
    ) -> dict:
        """Get attention weight summary for a block (for analysis).
        
        Args:
            cache: Cache containing layer outputs
            block_idx: Block index
            
        Returns:
            Dictionary with attention statistics for the block
        """
        block_start = self.get_block_start_layer(block_idx)
        block_end = self.get_block_end_layer(block_idx)
        
        summary = {
            "block_idx": block_idx,
            "layer_range": (block_start, block_end - 1),
            "layers": [],
        }
        
        for layer_idx in range(block_start, block_end):
            weights = self.get_attention_weights_for_analysis(cache, layer_idx)
            if weights is not None:
                summary["layers"].append({
                    "layer_idx": layer_idx,
                    "mean_weights": weights.mean().item(),
                    "std_weights": weights.std().item(),
                    "max_weights": weights.max().item(),
                    "min_weights": weights.min().item(),
                })
        
        return summary

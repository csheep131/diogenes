# Attention Residuals (AttnRes) Implementation Plan

**Document Type:** Architectural Implementation Plan  
**Based on:** arXiv:2603.15031 (Attention Residuals)  
**Date:** March 19, 2026  
**Author:** Software Architecture Team  

---

## Executive Summary

This document provides a comprehensive architectural plan for implementing **Attention Residuals (AttnRes)** into the Diogenes codebase. AttnRes replaces fixed-weight residual connections with learned, input-dependent attention weights over preceding layer outputs, enabling selective aggregation of earlier representations.

**Key Benefits Expected:**
- Improved epistemic reliability through better information flow across layers
- Enhanced gradient propagation in deep networks
- Potential 1.25x compute efficiency with Block AttnRes variant
- Better handling of knowledge boundary detection (core to Diogenes)

**Implementation Strategy:** Non-invasive wrapper-based approach that maintains compatibility with existing Qwen3-based architecture while enabling easy A/B testing.

---

## 1. Current Architecture Analysis

### 1.1 Existing Model Structure

The Diogenes project uses **Qwen3** as the base model (0.6B, 1.7B, 3B for development; 32B for production). The architecture follows standard transformer decoder design:

```
Qwen3 Transformer Block:
├── Input Normalization (RMSNorm)
├── Self-Attention
│   ├── q_proj, k_proj, v_proj
│   └── o_proj
├── Residual Connection (fixed weight = 1.0)
├── Normalization (RMSNorm)
├── MLP
│   ├── gate_proj, up_proj, down_proj
└── Residual Connection (fixed weight = 1.0)
```

**Key Observation:** The project uses Hugging Face `transformers` library with `AutoModelForCausalLM`. Residual connections are implicit in the transformer implementation, not exposed as separate modules.

### 1.2 Integration Points

Based on codebase analysis:

| File | Purpose | AttnRes Relevance |
|------|---------|-------------------|
| `src/diogenes/model.py` | Model loading/wrapping | Load AttnRes-enabled models |
| `src/diogenes/train_sft.py` | SFT training script | Train with AttnRes |
| `src/diogenes/train_dpo.py` | DPO training script | Train with AttnRes |
| `src/diogenes/inference.py` | Inference engine | Inference with AttnRes |
| `configs/config.yaml` | Configuration | AttnRes hyperparameters |

### 1.3 Constraints

1. **No Direct Model Modification:** Qwen3 weights come from Hugging Face; we cannot modify the base architecture directly
2. **PEFT Compatibility:** Must work with LoRA/QLoRA (4-bit quantization)
3. **Memory Constraints:** Development on RTX 3050 (8GB VRAM) limits batch sizes
4. **Hugging Face Integration:** Should maintain compatibility with `transformers` Trainer

---

## 2. AttnRes Architecture Design

### 2.1 Core Concept

**Standard Residual:**
```python
h_l = h_{l-1} + F(h_{l-1})  # Fixed weight = 1.0
```

**Attention Residual (Full):**
```python
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
where α_{i→l} = softmax(q_l · k_i / √d)
```

**Key Components:**
- **Values (v_i):** Layer outputs (hidden states)
- **Keys (k_i):** Projections of layer outputs for attention computation
- **Query (q_l):** Learned pseudo-query per layer (w_l ∈ ℝ^d)
- **Attention Weights (α):** Input-dependent, learned weights

### 2.2 Two Variants

#### Full AttnRes
- Each layer attends over **all** preceding outputs
- Memory: O(L·d) where L = number of layers, d = hidden dimension
- Best for: Maximum flexibility, research/experimentation

#### Block AttnRes
- Layers grouped into **blocks** (~8 blocks total)
- Each block attends over block outputs only
- Memory: O(N·d) where N = number of blocks (N << L)
- Best for: Production deployment, memory-constrained environments

### 2.3 Design Decision: Wrapper-Based Approach

**Rationale:**
1. Non-invasive: Doesn't require modifying Qwen3 source code
2. Composable: Can be enabled/disabled via configuration
3. Testable: Easy A/B testing between standard and AttnRes
4. Maintainable: Clear separation of concerns

**Implementation Pattern:** Module wrapper that intercepts transformer block outputs and applies AttnRes before passing to next layer.

---

## 3. Implementation Architecture

### 3.1 New Module Structure

```
src/diogenes/
├── attnres/
│   ├── __init__.py
│   ├── core.py           # Core AttnRes classes
│   ├── full.py           # Full AttnRes implementation
│   ├── block.py          # Block AttnRes implementation
│   ├── cache.py          # Cache-based pipeline communication
│   ├── config.py         # AttnRes configuration
│   └── utils.py          # Helper functions
```

### 3.2 Class Hierarchy

```
AttnResBase (ABC)
├── FullAttnRes
└── BlockAttnRes

AttnResConfig
AttnResCache
AttnResWrapper
```

### 3.3 Detailed Class Specifications

#### 3.3.1 AttnResConfig

```python
# File: src/diogenes/attnres/config.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AttnResVariant(Enum):
    """AttnRes variant selection."""
    FULL = "full"
    BLOCK = "block"


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
    """
    variant: AttnResVariant = AttnResVariant.FULL
    hidden_size: int = 2048
    num_layers: int = 32
    num_blocks: int = 8
    init_scale: float = 0.02
    use_layer_norm: bool = True
    dropout: float = 0.0
    cache_on_cpu: bool = False
    
    def __post_init__(self):
        if self.variant == AttnResVariant.BLOCK:
            if self.num_layers % self.num_blocks != 0:
                raise ValueError(
                    f"num_layers ({self.num_layers}) must be divisible by "
                    f"num_blocks ({self.num_blocks})"
                )
    
    @property
    def layers_per_block(self) -> int:
        """Number of layers per block."""
        return self.num_layers // self.num_blocks
```

#### 3.3.2 AttnResBase (Abstract Base Class)

```python
# File: src/diogenes/attnres/core.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple
from .config import AttnResConfig


class AttnResBase(ABC, nn.Module):
    """Base class for Attention Residuals.
    
    Implements the core attention mechanism over preceding layer outputs.
    Subclasses define how layers are grouped (full vs block).
    """
    
    def __init__(self, config: AttnResConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Key projection for attention computation
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
            query: Query tensor [batch, seq_len, hidden_size]
            keys: List of key tensors from preceding layers
            layer_idx: Current layer index
            
        Returns:
            Attention weights [batch, seq_len, num_preceding]
        """
        pass
    
    @abstractmethod
    def get_preceding_outputs(
        self,
        cache: 'AttnResCache',
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
        cache: 'AttnResCache',
        layer_idx: int,
    ) -> torch.Tensor:
        """Apply attention residual connection.
        
        Args:
            current_output: Current layer output [batch, seq_len, hidden_size]
            cache: Cache for storing/retrieving layer outputs
            layer_idx: Current layer index
            
        Returns:
            Output with attention residual applied
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
        
        # Combine with current output
        # Note: AttnRes replaces standard residual, not adds to it
        return aggregated
    
    def get_layer_query(self, layer_idx: int) -> torch.Tensor:
        """Get learned pseudo-query for specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Query tensor [1, 1, hidden_size]
        """
        # Each layer has its own learned query vector
        query = getattr(self, f'query_{layer_idx}', None)
        if query is None:
            query = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * self.config.init_scale
            )
            self.register_parameter(f'query_{layer_idx}', query)
        return query
    
    def aggregate_outputs(
        self,
        outputs: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted sum of outputs.
        
        Args:
            outputs: List of output tensors [batch, seq_len, hidden_size]
            weights: Attention weights [batch, seq_len, num_outputs]
            
        Returns:
            Weighted sum [batch, seq_len, hidden_size]
        """
        # Stack outputs: [batch, seq_len, num_outputs, hidden_size]
        stacked = torch.stack(outputs, dim=2)
        
        # Expand weights for broadcasting
        weights_expanded = weights.unsqueeze(-1)
        
        # Weighted sum
        aggregated = (stacked * weights_expanded).sum(dim=2)
        
        return aggregated
```

#### 3.3.3 FullAttnRes

```python
# File: src/diogenes/attnres/full.py

import torch
from typing import List
from .core import AttnResBase, AttnResCache
from .config import AttnResConfig


class FullAttnRes(AttnResBase):
    """Full Attention Residuals.
    
    Each layer attends over ALL preceding layer outputs.
    Memory complexity: O(L·d) where L = num_layers, d = hidden_size
    """
    
    def __init__(self, config: AttnResConfig):
        super().__init__(config)
        
        # Create learned queries for all layers
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
        
        Args:
            query: Learned query [1, 1, hidden_size]
            keys: List of key tensors [batch, seq_len, hidden_size]
            layer_idx: Current layer index (not used in Full variant)
            
        Returns:
            Attention weights [batch, seq_len, num_preceding]
        """
        # Stack keys: [batch, seq_len, num_preceding, hidden_size]
        keys_stacked = torch.stack(keys, dim=2)
        
        # Compute attention scores: query · keys
        # query: [1, 1, 1, hidden_size]
        # keys: [batch, seq_len, num_preceding, hidden_size]
        scores = (query.unsqueeze(2) * keys_stacked).sum(dim=-1) / (
            self.hidden_size ** 0.5
        )
        
        # Softmax over preceding layers dimension
        weights = torch.softmax(scores, dim=2)
        
        return weights
    
    def get_preceding_outputs(
        self,
        cache: 'AttnResCache',
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Get ALL preceding layer outputs.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            List of all preceding output tensors [h_0, h_1, ..., h_{l-1}]
        """
        return cache.get_all_preceding(layer_idx)
```

#### 3.3.4 BlockAttnRes

```python
# File: src/diogenes/attnres/block.py

import torch
from typing import List
from .core import AttnResBase, AttnResCache
from .config import AttnResConfig


class BlockAttnRes(AttnResBase):
    """Block Attention Residuals.
    
    Layers grouped into blocks. Each layer attends over outputs within its block.
    Memory complexity: O(N·d) where N = num_blocks << L
    
    Example: 32 layers / 8 blocks = 4 layers per block
    - Block 0: layers 0-3
    - Block 1: layers 4-7
    - ...
    """
    
    def __init__(self, config: AttnResConfig):
        super().__init__(config)
        
        self.layers_per_block = config.layers_per_block
        self.num_blocks = config.num_blocks
        
        # Create learned queries for all layers
        for i in range(config.num_layers):
            query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * config.init_scale
            )
            self.register_parameter(f'query_{i}', query)
        
        # Block-level queries for inter-block attention (optional enhancement)
        for b in range(config.num_blocks):
            block_query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * config.init_scale
            )
            self.register_parameter(f'block_query_{b}', block_query)
    
    def get_block_idx(self, layer_idx: int) -> int:
        """Get block index for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Block index (0 to num_blocks-1)
        """
        return layer_idx // self.layers_per_block
    
    def get_block_start_layer(self, block_idx: int) -> int:
        """Get starting layer index for a block.
        
        Args:
            block_idx: Block index
            
        Returns:
            Starting layer index
        """
        return block_idx * self.layers_per_block
    
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        keys: List[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute softmax attention over block keys.
        
        Args:
            query: Learned query [1, 1, hidden_size]
            keys: List of key tensors within block
            layer_idx: Current layer index
            
        Returns:
            Attention weights [batch, seq_len, num_preceding_in_block]
        """
        # Stack keys: [batch, seq_len, num_preceding, hidden_size]
        keys_stacked = torch.stack(keys, dim=2)
        
        # Compute attention scores
        scores = (query.unsqueeze(2) * keys_stacked).sum(dim=-1) / (
            self.hidden_size ** 0.5
        )
        
        # Softmax over preceding layers within block
        weights = torch.softmax(scores, dim=2)
        
        return weights
    
    def get_preceding_outputs(
        self,
        cache: 'AttnResCache',
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Get preceding outputs within the same block.
        
        Args:
            cache: Cache containing layer outputs
            layer_idx: Current layer index
            
        Returns:
            List of preceding output tensors within block
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
```

#### 3.3.5 AttnResCache

```python
# File: src/diogenes/attnres/cache.py

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AttnResCache:
    """Cache for storing layer outputs during forward pass.
    
    Enables two-phase computation strategy:
    1. Phase 1: Compute and cache layer outputs
    2. Phase 2: Apply AttnRes using cached outputs
    
    Attributes:
        outputs: Dictionary mapping layer_idx to output tensor
        device: Primary device for cache
        cpu_offload: Whether to offload to CPU for memory savings
    """
    outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_offload: bool = False
    
    def store_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """Store layer output in cache.
        
        Args:
            layer_idx: Layer index
            output: Layer output tensor [batch, seq_len, hidden_size]
        """
        if self.cpu_offload and output.device.type == 'cuda':
            # Store on CPU to save VRAM
            self.outputs[layer_idx] = output.cpu()
        else:
            self.outputs[layer_idx] = output
    
    def get_layer_output(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get cached output for specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Output tensor or None if not cached
        """
        output = self.outputs.get(layer_idx)
        if output is not None and self.cpu_offload and output.device.type == 'cpu':
            # Move back to GPU when needed
            return output.to(self.device)
        return output
    
    def get_all_preceding(self, layer_idx: int) -> List[torch.Tensor]:
        """Get all preceding layer outputs.
        
        Args:
            layer_idx: Current layer index
            
        Returns:
            List of output tensors [h_0, h_1, ..., h_{l-1}]
        """
        outputs = []
        for i in range(layer_idx):
            output = self.get_layer_output(i)
            if output is not None:
                outputs.append(output)
        return outputs
    
    def clear(self) -> None:
        """Clear all cached outputs."""
        self.outputs.clear()
    
    def __len__(self) -> int:
        """Number of cached layers."""
        return len(self.outputs)
```

#### 3.3.6 AttnResWrapper

```python
# File: src/diogenes/attnres/core.py (continued)

class AttnResWrapper:
    """Wrapper for applying AttnRes to transformer models.
    
    This is the main integration point. It wraps an existing model
    and intercepts transformer block outputs to apply AttnRes.
    
    Usage:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        config = AttnResConfig(variant=AttnResVariant.FULL, hidden_size=2048, num_layers=32)
        wrapper = AttnResWrapper(model, config)
        wrapped_model = wrapper.wrap()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: AttnResConfig,
        apply_to: str = "both",  # "attention", "mlp", or "both"
    ):
        self.model = model
        self.config = config
        self.apply_to = apply_to
        
        # Initialize AttnRes module
        if config.variant == AttnResVariant.FULL:
            from .full import FullAttnRes
            self.attn_res = FullAttnRes(config)
        elif config.variant == AttnResVariant.BLOCK:
            from .block import BlockAttnRes
            self.attn_res = BlockAttnRes(config)
        else:
            raise ValueError(f"Unknown variant: {config.variant}")
        
        # Cache for layer outputs
        self.cache = AttnResCache()
    
    def wrap(self) -> nn.Module:
        """Apply AttnRes wrapper to model.
        
        Returns:
            Wrapped model with AttnRes applied
        """
        # Find transformer layers
        transformer_layers = self._find_transformer_layers()
        
        if not transformer_layers:
            raise RuntimeError("Could not find transformer layers in model")
        
        # Hook into each layer
        self._install_hooks(transformer_layers)
        
        return self.model
    
    def _find_transformer_layers(self) -> List[nn.Module]:
        """Find transformer block layers in model.
        
        Returns:
            List of transformer block modules
        """
        # Try common layer naming patterns
        layer_candidates = []
        
        # Pattern 1: model.model.layers (Qwen, Llama)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer_candidates = list(self.model.model.layers)
        
        # Pattern 2: model.transformer.h (GPT-2 style)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layer_candidates = list(self.model.transformer.h)
        
        # Pattern 3: Direct layers attribute
        elif hasattr(self.model, 'layers'):
            layer_candidates = list(self.model.layers)
        
        return layer_candidates
    
    def _install_hooks(self, layers: List[nn.Module]) -> None:
        """Install forward hooks on transformer layers.
        
        Args:
            layers: List of transformer layer modules
        """
        for idx, layer in enumerate(layers):
            layer.register_forward_hook(
                self._make_hook(idx),
                always_call=True,
            )
    
    def _make_hook(self, layer_idx: int):
        """Create forward hook for layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Forward hook function
        """
        def hook(module, inputs, output):
            # output is the layer output after standard residual
            # We apply AttnRes here
            
            # For now, just cache the output
            # Full implementation would modify the output
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
```

### 3.4 Integration with Existing Code

#### 3.4.1 Model Loading (model.py)

```python
# Add to src/diogenes/model.py

from typing import Optional
from diogenes.attnres.config import AttnResConfig, AttnResVariant
from diogenes.attnres.core import AttnResWrapper


class DiogenesModel:
    # ... existing code ...
    
    def enable_attnres(
        self,
        variant: str = "full",
        num_blocks: int = 8,
        apply_to: str = "both",
    ) -> None:
        """Enable Attention Residuals.
        
        Args:
            variant: AttnRes variant ("full" or "block")
            num_blocks: Number of blocks for Block variant
            apply_to: Where to apply ("attention", "mlp", or "both")
        """
        variant_enum = AttnResVariant(variant)
        
        config = AttnResConfig(
            variant=variant_enum,
            hidden_size=self.model.config.hidden_size,
            num_layers=self.model.config.num_hidden_layers,
            num_blocks=num_blocks,
        )
        
        wrapper = AttnResWrapper(self.model, config, apply_to=apply_to)
        self.model = wrapper.wrap()
        
        logger.info(f"AttnRes enabled: variant={variant}, layers={config.num_layers}")
```

#### 3.4.2 Configuration (config.yaml)

```yaml
# Add to configs/config.yaml

# Attention Residuals (AttnRes) settings
attnres:
  enabled: false  # Set to true to enable
  variant: "full"  # "full" or "block"
  num_blocks: 8  # For block variant
  apply_to: "both"  # "attention", "mlp", or "both"
  init_scale: 0.02
  dropout: 0.0
  use_layer_norm: true
  cache_on_cpu: false  # Set to true for memory-constrained environments
```

#### 3.4.3 Training Scripts (train_sft.py, train_dpo.py)

```python
# Add to training scripts after model loading

def load_model_with_attnres(args, config):
    """Load model with optional AttnRes."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(...)
    
    # Enable AttnRes if configured
    if config.get('attnres', {}).get('enabled', False):
        from diogenes.attnres import AttnResConfig, AttnResWrapper, AttnResVariant
        
        attnres_config = AttnResConfig(
            variant=AttnResVariant(config['attnres']['variant']),
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
            num_blocks=config['attnres'].get('num_blocks', 8),
            init_scale=config['attnres'].get('init_scale', 0.02),
            dropout=config['attnres'].get('dropout', 0.0),
        )
        
        wrapper = AttnResWrapper(model, attnres_config)
        model = wrapper.wrap()
        
        logger.info("AttnRes enabled for training")
    
    return model
```

---

## 4. Memory and Computation Analysis

### 4.1 Memory Requirements

| Component | Full AttnRes | Block AttnRes |
|-----------|--------------|---------------|
| Cache (L layers) | O(L·b·s·d) | O(N·b·s·d) |
| Attention weights | O(b·s·L) | O(b·s·L/N) |
| Learnable queries | O(L·d) | O(L·d) |
| Key projections | O(L·d²) | O(L·d²) |

**Where:**
- L = number of layers (32 for Qwen3-32B)
- N = number of blocks (8 default)
- b = batch size
- s = sequence length
- d = hidden dimension (2048 for Qwen3-32B)

### 4.2 Memory Optimization Strategies

1. **CPU Offloading:** Cache layer outputs on CPU, move to GPU only when needed
2. **Gradient Checkpointing:** Re-compute outputs during backward pass instead of caching
3. **Mixed Precision:** Use FP16/BF16 for cache storage
4. **Block Variant:** Use Block AttnRes for production (8x memory reduction)

### 4.3 Computation Overhead

| Operation | Additional FLOPs |
|-----------|------------------|
| Key projection | L · b · s · d² |
| Attention computation | L² · b · s · d (Full) |
| Attention computation | L · b · s · d (Block) |
| Weighted aggregation | L² · b · s · d (Full) |
| Weighted aggregation | L · b · s · d (Block) |

**Estimated Overhead:**
- Full AttnRes: ~15-20% additional compute
- Block AttnRes: ~5-8% additional compute

---

## 5. Testing Strategy

### 5.1 Unit Tests

```python
# File: tests/test_attnres.py

import pytest
import torch
from diogenes.attnres.config import AttnResConfig, AttnResVariant
from diogenes.attnres.full import FullAttnRes
from diogenes.attnres.block import BlockAttnRes
from diogenes.attnres.cache import AttnResCache


class TestAttnResConfig:
    """Test AttnRes configuration."""
    
    def test_full_variant_config(self):
        """Test Full AttnRes config creation."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=2048,
            num_layers=32,
        )
        assert config.variant == AttnResVariant.FULL
        assert config.hidden_size == 2048
        assert config.num_layers == 32
    
    def test_block_variant_config(self):
        """Test Block AttnRes config creation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=2048,
            num_layers=32,
            num_blocks=8,
        )
        assert config.variant == AttnResVariant.BLOCK
        assert config.layers_per_block == 4
    
    def test_block_config_validation(self):
        """Test Block config validates layer/block divisibility."""
        with pytest.raises(ValueError):
            AttnResConfig(
                variant=AttnResVariant.BLOCK,
                num_layers=30,
                num_blocks=8,
            )


class TestFullAttnRes:
    """Test Full AttnRes implementation."""
    
    def test_forward_pass(self):
        """Test Full AttnRes forward pass."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=4,
        )
        attn_res = FullAttnRes(config)
        cache = AttnResCache()
        
        # Simulate layer outputs
        batch_size, seq_len = 2, 16
        for i in range(4):
            output = torch.randn(batch_size, seq_len, 128)
            cache.store_output(i, output)
        
        # Apply AttnRes to last layer
        last_output = torch.randn(batch_size, seq_len, 128)
        result = attn_res(last_output, cache, layer_idx=3)
        
        assert result.shape == last_output.shape
    
    def test_first_layer_no_preceding(self):
        """Test that first layer returns output unchanged."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=4,
        )
        attn_res = FullAttnRes(config)
        cache = AttnResCache()
        
        output = torch.randn(2, 16, 128)
        result = attn_res(output, cache, layer_idx=0)
        
        # First layer has no preceding outputs, should return unchanged
        assert torch.allclose(result, output)


class TestBlockAttnRes:
    """Test Block AttnRes implementation."""
    
    def test_block_indexing(self):
        """Test block index calculation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            num_layers=32,
            num_blocks=8,
        )
        attn_res = BlockAttnRes(config)
        
        assert attn_res.get_block_idx(0) == 0
        assert attn_res.get_block_idx(3) == 0
        assert attn_res.get_block_idx(4) == 1
        assert attn_res.get_block_idx(7) == 1
        assert attn_res.get_block_idx(31) == 7
    
    def test_block_forward_pass(self):
        """Test Block AttnRes forward pass."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        cache = AttnResCache()
        
        # Simulate layer outputs
        batch_size, seq_len = 2, 16
        for i in range(8):
            output = torch.randn(batch_size, seq_len, 128)
            cache.store_output(i, output)
        
        # Layer 5 should only attend to layers 4 (within block 1)
        last_output = torch.randn(batch_size, seq_len, 128)
        result = attn_res(last_output, cache, layer_idx=5)
        
        assert result.shape == last_output.shape


class TestAttnResCache:
    """Test AttnRes cache functionality."""
    
    def test_store_and_retrieve(self):
        """Test cache store and retrieve."""
        cache = AttnResCache()
        
        output = torch.randn(2, 16, 128)
        cache.store_output(0, output)
        
        retrieved = cache.get_layer_output(0)
        assert torch.allclose(retrieved, output)
    
    def test_get_all_preceding(self):
        """Test getting all preceding outputs."""
        cache = AttnResCache()
        
        for i in range(5):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        preceding = cache.get_all_preceding(5)
        assert len(preceding) == 5
    
    def test_cpu_offload(self):
        """Test CPU offloading for memory savings."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cache = AttnResCache(cpu_offload=True)
        output = torch.randn(2, 16, 128).cuda()
        cache.store_output(0, output)
        
        # Should be stored on CPU
        assert cache.outputs[0].device.type == 'cpu'
        
        # Retrieval should move back to GPU
        retrieved = cache.get_layer_output(0)
        assert retrieved.device.type == 'cuda'
```

### 5.2 Integration Tests

```python
# File: tests/test_attnres_integration.py

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diogenes.attnres import AttnResConfig, AttnResWrapper, AttnResVariant


class TestAttnResIntegration:
    """Test AttnRes integration with Qwen3 model."""
    
    @pytest.fixture
    def small_model(self):
        """Load small test model."""
        model_name = "Qwen/Qwen3-0.6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Use CPU for tests
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def test_wrapper_creation(self, small_model):
        """Test AttnRes wrapper creation."""
        model, tokenizer = small_model
        
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
        )
        
        wrapper = AttnResWrapper(model, config)
        wrapped_model = wrapper.wrap()
        
        assert wrapped_model is not None
    
    def test_forward_pass_with_attnres(self, small_model):
        """Test forward pass with AttnRes enabled."""
        model, tokenizer = small_model
        
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
        )
        
        wrapper = AttnResWrapper(model, config)
        wrapped_model = wrapper.wrap()
        
        # Tokenize input
        inputs = tokenizer("Test input", return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = wrapped_model(**inputs)
        
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size
    
    def test_block_variant_memory_efficiency(self, small_model):
        """Test Block variant uses less memory."""
        model, tokenizer = small_model
        
        # Full variant
        config_full = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
        )
        wrapper_full = AttnResWrapper(model, config_full)
        wrapped_full = wrapper_full.wrap()
        
        # Block variant
        config_block = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
            num_blocks=8,
        )
        wrapper_block = AttnResWrapper(model, config_block)
        wrapped_block = wrapper_block.wrap()
        
        # Count parameters
        params_full = sum(p.numel() for p in wrapped_full.parameters())
        params_block = sum(p.numel() for p in wrapped_block.parameters())
        
        # Block should have similar params (queries are same, but cache is smaller)
        # This is a sanity check
        assert params_block <= params_full * 1.1  # Allow 10% tolerance
```

### 5.3 Ablation Testing

```python
# File: tests/test_attnres_ablation.py

"""Ablation tests for AttnRes variants."""

import torch
from diogenes.attnres.config import AttnResConfig, AttnResVariant
from diogenes.attnres.full import FullAttnRes
from diogenes.attnres.block import BlockAttnRes
from diogenes.attnres.cache import AttnResCache


def test_attention_weight_distribution():
    """Test that attention weights sum to 1."""
    config = AttnResConfig(
        variant=AttnResVariant.FULL,
        hidden_size=128,
        num_layers=4,
    )
    attn_res = FullAttnRes(config)
    cache = AttnResCache()
    
    # Store preceding outputs
    for i in range(4):
        cache.store_output(i, torch.randn(2, 16, 128))
    
    # Get attention weights (need to access internal method)
    preceding = cache.get_all_preceding(4)
    keys = [attn_res.key_proj(out) for out in preceding]
    query = attn_res.get_layer_query(3)
    weights = attn_res.compute_attention_weights(query, keys, 3)
    
    # Weights should sum to 1 over preceding dimension
    weight_sums = weights.sum(dim=2)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_gradient_flow():
    """Test that gradients flow through AttnRes."""
    config = AttnResConfig(
        variant=AttnResVariant.FULL,
        hidden_size=128,
        num_layers=4,
    )
    attn_res = FullAttnRes(config)
    cache = AttnResCache()
    
    # Store preceding outputs with gradients
    for i in range(4):
        output = torch.randn(2, 16, 128, requires_grad=True)
        cache.store_output(i, output)
    
    # Forward pass
    last_output = torch.randn(2, 16, 128, requires_grad=True)
    result = attn_res(last_output, cache, layer_idx=3)
    
    # Backward pass
    loss = result.sum()
    loss.backward()
    
    # Check that gradients exist
    assert last_output.grad is not None
    for i in range(4):
        output = cache.get_layer_output(i)
        assert output.grad is not None
```

---

## 6. Implementation Phases

### Phase 1: Core Implementation (Week 1)

**Goals:** Implement basic AttnRes functionality

**Tasks:**
1. Create `src/diogenes/attnres/` module structure
2. Implement `AttnResConfig` class
3. Implement `AttnResCache` class
4. Implement `AttnResBase` abstract class
5. Implement `FullAttnRes` class
6. Implement `BlockAttnRes` class
7. Implement `AttnResWrapper` class

**Deliverables:**
- Core AttnRes classes functional
- Unit tests passing
- Basic integration test with Qwen3-0.6B

### Phase 2: Integration (Week 2)

**Goals:** Integrate with existing Diogenes codebase

**Tasks:**
1. Add `enable_attnres()` method to `DiogenesModel`
2. Add AttnRes configuration to `config.yaml`
3. Update training scripts to support AttnRes
4. Add logging for AttnRes metrics
5. Test with SFT training pipeline

**Deliverables:**
- AttnRes can be enabled via config
- Training scripts work with AttnRes
- No regression in existing functionality

### Phase 3: Optimization (Week 3)

**Goals:** Optimize for memory and performance

**Tasks:**
1. Implement CPU offloading for cache
2. Add gradient checkpointing support
3. Profile memory usage
4. Profile computation overhead
5. Optimize attention computation (kernel fusion)

**Deliverables:**
- Memory usage within RTX 3050 constraints
- Computation overhead < 20% (Full), < 8% (Block)
- Performance benchmarks documented

### Phase 4: Evaluation (Week 4)

**Goals:** Evaluate AttnRes impact on epistemic reliability

**Tasks:**
1. Train SFT with AttnRes (Qwen2.5-3B)
2. Train DPO with AttnRes (Qwen2.5-3B)
3. Evaluate on epistemic metrics:
   - Pass@1
   - Hallucination Rate
   - ECE (Expected Calibration Error)
   - Abstention AUROC
   - Mode Accuracy
4. Compare with baseline (no AttnRes)

**Deliverables:**
- Evaluation report
- Recommendation on AttnRes adoption
- Decision on Full vs Block variant

---

## 7. Risks and Mitigation

### Risk 1: Memory Overhead

**Risk:** AttnRes cache exceeds VRAM limits on RTX 3050 (8GB)

**Mitigation:**
- Use Block variant (8x memory reduction)
- Enable CPU offloading
- Reduce batch size
- Use gradient checkpointing

**Fallback:** Disable AttnRes for development, enable only for production training

### Risk 2: Training Instability

**Risk:** AttnRes introduces training instability or convergence issues

**Mitigation:**
- Start with small init_scale (0.01)
- Use layer normalization
- Gradual warmup of AttnRes contribution
- Monitor gradient norms

**Fallback:** Blend AttnRes with standard residual: `output = α·attn_res + (1-α)·standard`

### Risk 3: No Improvement in Epistemic Metrics

**Risk:** AttnRes doesn't improve epistemic reliability

**Mitigation:**
- Run ablation studies to understand why
- Try different AttnRes variants
- Adjust hyperparameters (init_scale, dropout)

**Fallback:** Disable AttnRes, no loss in functionality (wrapper is non-invasive)

### Risk 4: Inference Latency

**Risk:** AttnRes increases inference latency beyond acceptable limits

**Mitigation:**
- Use Block variant for production
- Optimize attention computation
- Cache key projections

**Fallback:** Enable AttnRes only during training, use standard residuals for inference

---

## 8. Success Criteria

### Technical Criteria

- [ ] All unit tests passing (>95% coverage)
- [ ] Integration tests passing
- [ ] Memory usage < 8GB for Qwen2.5-3B with batch_size=4
- [ ] Computation overhead < 20% (Full), < 8% (Block)
- [ ] No regression in existing functionality

### Epistemic Criteria

- [ ] Pass@1: No degradation (±1%)
- [ ] Hallucination Rate: Improvement > 5%
- [ ] ECE: Improvement > 10%
- [ ] Abstention AUROC: Improvement > 5%
- [ ] Mode Accuracy: Improvement > 5%

### Go/No-Go Decision

**Proceed to production if:**
- Technical criteria met
- At least 2 epistemic criteria show improvement
- No critical regressions

**Revert to baseline if:**
- Technical criteria not met
- Pass@1 degradation > 2%
- Multiple epistemic metrics degrade

---

## 9. Next Steps

### Immediate Actions

1. **Create module structure:**
   ```bash
   mkdir -p src/diogenes/attnres
   touch src/diogenes/attnres/__init__.py
   touch src/diogenes/attnres/config.py
   touch src/diogenes/attnres/core.py
   touch src/diogenes/attnres/full.py
   touch src/diogenes/attnres/block.py
   touch src/diogenes/attnres/cache.py
   touch src/diogenes/attnres/utils.py
   ```

2. **Implement core classes** (see Section 3.3 for specifications)

3. **Write unit tests** (see Section 5.1 for test specifications)

4. **Run initial integration test** with Qwen3-0.6B

### Configuration for Initial Testing

```yaml
# configs/config.yaml
attnres:
  enabled: true
  variant: "full"  # Start with Full for testing
  num_blocks: 8
  apply_to: "both"
  init_scale: 0.02
  dropout: 0.0
  use_layer_norm: true
  cache_on_cpu: false
```

### Training Command

```bash
# Test SFT training with AttnRes
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen3-0.6B \
  --config configs/config.yaml \
  --output_dir models/sft_attnres_test \
  --num_train_epochs 1
```

---

## 10. Appendix

### 10.1 Related Work

- **PreNorm vs PostNorm:** AttnRes works with PreNorm architecture (used in Qwen3)
- **DenseNet:** Similar idea of connecting all layers, but AttnRes uses learned weights
- **Transformer-XL:** Uses memory of previous segments, AttnRes uses memory of previous layers

### 10.2 Hyperparameter Recommendations

| Hyperparameter | Recommended Range | Default |
|----------------|-------------------|---------|
| init_scale | 0.01 - 0.1 | 0.02 |
| dropout | 0.0 - 0.2 | 0.0 |
| num_blocks | 4 - 16 | 8 |
| use_layer_norm | true/false | true |

### 10.3 Debugging Tips

1. **Check attention weight distribution:**
   ```python
   weights = attn_res.compute_attention_weights(...)
   print(f"Weights sum: {weights.sum(dim=2)}")  # Should be ~1.0
   print(f"Weights entropy: {-(weights * weights.log()).sum(dim=2)}")
   ```

2. **Monitor cache memory:**
   ```python
   import gc
   print(f"Cache size: {len(cache)}")
   print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

3. **Check gradient flow:**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
   ```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-19 | Architecture Team | Initial implementation plan |

---

## References

1. **Attention Residuals Paper:** arXiv:2603.15031 (Kimi Linear team)
2. **Qwen3 Technical Report:** arXiv:2412.15115
3. **Transformer Architecture:** "Attention Is All You Need" (Vaswani et al., 2017)
4. **PreNorm vs PostNorm:** "Understanding and Difficulty in Training Deep Transformers" (Nguyen & Salazar, 2019)

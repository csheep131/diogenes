"""Utility functions for Attention Residuals."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .config import AttnResConfig, AttnResVariant
from .cache import AttnResCache


def count_attnres_parameters(
    model: nn.Module,
    config: AttnResConfig,
) -> Dict[str, int]:
    """Count parameters added by AttnRes.
    
    Args:
        model: Model with AttnRes applied
        config: AttnRes configuration
        
    Returns:
        Dictionary with parameter counts:
        - total: Total AttnRes parameters
        - queries: Parameters for learned queries
        - key_proj: Parameters for key projection
        - norm: Parameters for layer normalization (if enabled)
    """
    total = 0
    queries = 0
    key_proj = 0
    norm = 0
    
    # Count query parameters
    for name, param in model.named_parameters():
        if name.startswith('attn_res.query_'):
            queries += param.numel()
            total += param.numel()
        elif name.startswith('attn_res.key_proj'):
            key_proj += param.numel()
            total += param.numel()
        elif name.startswith('attn_res.norm'):
            norm += param.numel()
            total += param.numel()
    
    return {
        "total": total,
        "queries": queries,
        "key_proj": key_proj,
        "norm": norm,
    }


def estimate_memory_usage(
    config: AttnResConfig,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Estimate memory usage for AttnRes cache.
    
    Args:
        config: AttnRes configuration
        batch_size: Batch size
        seq_len: Sequence length
        dtype: Data type of tensors
        
    Returns:
        Dictionary with memory estimates in MB:
        - cache_full: Memory for Full AttnRes cache
        - cache_block: Memory for Block AttnRes cache
        - attention_weights_full: Memory for Full attention weights
        - attention_weights_block: Memory for Block attention weights
    """
    bytes_per_element = torch.tensor(0, dtype=dtype).element_size()
    hidden_size = config.hidden_size
    
    # Cache memory (storing layer outputs)
    # Full: stores all L layers
    # Block: stores only layers_per_block layers
    bytes_per_tensor = batch_size * seq_len * hidden_size * bytes_per_element
    
    cache_full_mb = (config.num_layers * bytes_per_tensor) / (1024 ** 2)
    cache_block_mb = (config.layers_per_block * bytes_per_tensor) / (1024 ** 2)
    
    # Attention weights memory
    # Full: [batch, seq_len, num_preceding] where num_preceding can be up to L
    # Block: [batch, seq_len, num_preceding] where num_preceding <= layers_per_block
    avg_preceding_full = config.num_layers / 2  # Average over all layers
    avg_preceding_block = config.layers_per_block / 2
    
    weights_full_mb = (batch_size * seq_len * avg_preceding_full * bytes_per_element) / (1024 ** 2)
    weights_block_mb = (batch_size * seq_len * avg_preceding_block * bytes_per_element) / (1024 ** 2)
    
    return {
        "cache_full_mb": cache_full_mb,
        "cache_block_mb": cache_block_mb,
        "attention_weights_full_mb": weights_full_mb,
        "attention_weights_block_mb": weights_block_mb,
        "total_full_mb": cache_full_mb + weights_full_mb,
        "total_block_mb": cache_block_mb + weights_block_mb,
    }


def analyze_attention_distribution(
    weights: torch.Tensor,
    layer_idx: int,
) -> Dict[str, float]:
    """Analyze attention weight distribution.
    
    Args:
        weights: Attention weights [batch, seq_len, num_preceding]
        layer_idx: Current layer index
        
    Returns:
        Dictionary with attention statistics:
        - mean: Mean attention weight
        - std: Standard deviation
        - entropy: Entropy of attention distribution
        - max: Maximum weight
        - min: Minimum weight
        - sparsity: Fraction of weights < 0.01
    """
    # Flatten over batch and seq_len for statistics
    weights_flat = weights.view(-1, weights.shape[-1])
    
    # Entropy: -Σ w · log(w)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    entropy = -(weights_flat * torch.log(weights_flat + epsilon)).sum(dim=-1).mean().item()
    
    # Sparsity: fraction of weights < 0.01
    sparsity = (weights_flat < 0.01).float().mean().item()
    
    return {
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "entropy": entropy,
        "max": weights.max().item(),
        "min": weights.min().item(),
        "sparsity": sparsity,
        "layer_idx": layer_idx,
    }


def get_attention_patterns(
    weights: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[List[int], List[float]]:
    """Identify dominant attention patterns.
    
    Finds which preceding layers receive the most attention.
    
    Args:
        weights: Attention weights [batch, seq_len, num_preceding]
        threshold: Threshold for "dominant" attention
        
    Returns:
        Tuple of (dominant_indices, dominant_weights):
        - dominant_indices: Indices of layers with attention > threshold
        - dominant_weights: Corresponding attention weights
    """
    # Find max attention for each position
    max_weights, max_indices = weights.max(dim=-1)
    
    # Find positions where max attention exceeds threshold
    mask = max_weights > threshold
    dominant_indices = max_indices[mask].tolist()
    dominant_weights = max_weights[mask].tolist()
    
    return dominant_indices, dominant_weights


def create_layer_visualization(
    config: AttnResConfig,
    cache: AttnResCache,
    attn_res: nn.Module,
) -> str:
    """Create ASCII visualization of attention patterns.
    
    Args:
        config: AttnRes configuration
        cache: Cache containing layer outputs
        attn_res: AttnRes module
        
    Returns:
        ASCII string visualization
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"AttnRes Attention Pattern ({config.variant.value} variant)")
    lines.append(f"Layers: {config.num_layers}, Hidden: {config.hidden_size}")
    if config.variant == AttnResVariant.BLOCK:
        lines.append(f"Blocks: {config.num_blocks}, Layers/block: {config.layers_per_block}")
    lines.append("=" * 60)
    
    for layer_idx in range(min(config.num_layers, 20)):  # Limit to first 20 layers
        weights = attn_res.get_attention_weights_for_analysis(cache, layer_idx)
        
        if weights is None:
            lines.append(f"Layer {layer_idx:2d}: [FIRST LAYER - no preceding]")
            continue
        
        # Get mean weights over batch and seq_len
        mean_weights = weights.mean(dim=(0, 1)).cpu().numpy()
        
        # Create bar visualization
        bar = ""
        for w in mean_weights:
            if w > 0.5:
                bar += "█"
            elif w > 0.3:
                bar += "▓"
            elif w > 0.1:
                bar += "▒"
            elif w > 0.01:
                bar += "░"
            else:
                bar += "·"
        
        lines.append(f"Layer {layer_idx:2d}: [{bar}]")
    
    if config.num_layers > 20:
        lines.append(f"... ({config.num_layers - 20} more layers)")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def validate_attnres_setup(
    model: nn.Module,
    config: AttnResConfig,
) -> Tuple[bool, List[str]]:
    """Validate AttnRes setup before training.
    
    Args:
        model: Model with AttnRes applied
        config: AttnRes configuration
        
    Returns:
        Tuple of (is_valid, errors):
        - is_valid: True if setup is valid
        - errors: List of error messages
    """
    errors = []
    
    # Check that AttnRes module exists
    if not hasattr(model, 'attn_res') and not hasattr(model, '_hf_hook'):
        errors.append("AttnRes wrapper not properly applied to model")
    
    # Check configuration
    if config.hidden_size <= 0:
        errors.append(f"Invalid hidden_size: {config.hidden_size}")
    
    if config.num_layers <= 0:
        errors.append(f"Invalid num_layers: {config.num_layers}")
    
    if config.variant == AttnResVariant.BLOCK:
        if config.num_layers % config.num_blocks != 0:
            errors.append(
                f"For BLOCK variant, num_layers ({config.num_layers}) must be "
                f"divisible by num_blocks ({config.num_blocks})"
            )
    
    # Check device consistency
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    if len(devices) > 1:
        errors.append(f"Model parameters on multiple devices: {devices}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

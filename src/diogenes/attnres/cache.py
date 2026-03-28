"""Cache for storing layer outputs during AttnRes forward pass."""

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
    
    Example:
        >>> cache = AttnResCache(cpu_offload=False)
        >>> output = torch.randn(2, 512, 2048)  # [batch, seq_len, hidden]
        >>> cache.store_output(0, output)
        >>> retrieved = cache.get_layer_output(0)
        >>> assert torch.allclose(retrieved, output)
    """
    
    outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    cpu_offload: bool = False
    
    def store_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """Store layer output in cache.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            output: Layer output tensor [batch, seq_len, hidden_size]
        
        Note:
            If cpu_offload is True and output is on CUDA, the tensor
            will be stored on CPU to save VRAM. It will be moved back
            to GPU when retrieved.
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
            Output tensor [batch, seq_len, hidden_size] or None if not cached
        
        Note:
            If cpu_offload is True and tensor is on CPU, it will be
            moved back to GPU before returning.
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
            List of output tensors [h_0, h_1, ..., h_{layer_idx-1}]
            
        Example:
            >>> cache = AttnResCache()
            >>> for i in range(5):
            ...     cache.store_output(i, torch.randn(2, 512, 2048))
            >>> preceding = cache.get_all_preceding(5)
            >>> len(preceding)
            5
        """
        outputs = []
        for i in range(layer_idx):
            output = self.get_layer_output(i)
            if output is not None:
                outputs.append(output)
        return outputs
    
    def get_preceding_in_range(
        self,
        layer_idx: int,
        start_idx: int,
        end_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Get preceding layer outputs within a specific range.
        
        Useful for Block AttnRes variant.
        
        Args:
            layer_idx: Current layer index
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (exclusive), defaults to layer_idx
            
        Returns:
            List of output tensors in range [start_idx, end_idx)
        """
        if end_idx is None:
            end_idx = layer_idx
        
        outputs = []
        for i in range(start_idx, min(end_idx, layer_idx)):
            output = self.get_layer_output(i)
            if output is not None:
                outputs.append(output)
        return outputs
    
    def clear(self) -> None:
        """Clear all cached outputs.
        
        Should be called between forward passes to prevent memory leaks.
        """
        self.outputs.clear()
    
    def __len__(self) -> int:
        """Number of cached layers."""
        return len(self.outputs)
    
    def __contains__(self, layer_idx: int) -> bool:
        """Check if layer output is cached."""
        return layer_idx in self.outputs
    
    def memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage of cache.
        
        Returns:
            Dictionary with memory usage in MB for GPU and CPU
        """
        gpu_memory = 0.0
        cpu_memory = 0.0
        
        for output in self.outputs.values():
            # Each element is 4 bytes (float32) or 2 bytes (float16)
            bytes_per_element = output.element_size()
            memory_bytes = output.numel() * bytes_per_element
            memory_mb = memory_bytes / (1024 ** 2)
            
            if output.device.type == 'cuda':
                gpu_memory += memory_mb
            else:
                cpu_memory += memory_mb
        
        return {
            "gpu_memory_mb": gpu_memory,
            "cpu_memory_mb": cpu_memory,
            "total_memory_mb": gpu_memory + cpu_memory,
        }

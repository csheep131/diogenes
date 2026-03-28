"""Unit tests for Attention Residuals (AttnRes) implementation."""

import pytest
import torch
import torch.nn as nn

from diogenes.attnres.config import AttnResConfig, AttnResVariant
from diogenes.attnres.cache import AttnResCache
from diogenes.attnres.full import FullAttnRes
from diogenes.attnres.block import BlockAttnRes
from diogenes.attnres.utils import (
    count_attnres_parameters,
    estimate_memory_usage,
    analyze_attention_distribution,
    validate_attnres_setup,
)


class TestAttnResConfig:
    """Test AttnRes configuration."""
    
    def test_full_variant_config_creation(self):
        """Test Full AttnRes config creation."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=2048,
            num_layers=32,
        )
        assert config.variant == AttnResVariant.FULL
        assert config.hidden_size == 2048
        assert config.num_layers == 32
        assert config.num_blocks == 8  # default
        assert config.init_scale == 0.02  # default
    
    def test_block_variant_config_creation(self):
        """Test Block AttnRes config creation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=2048,
            num_layers=32,
            num_blocks=8,
        )
        assert config.variant == AttnResVariant.BLOCK
        assert config.hidden_size == 2048
        assert config.num_layers == 32
        assert config.num_blocks == 8
        assert config.layers_per_block == 4
    
    def test_block_config_layers_per_block(self):
        """Test layers_per_block calculation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            num_layers=32,
            num_blocks=8,
        )
        assert config.layers_per_block == 4
        
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            num_layers=48,
            num_blocks=6,
        )
        assert config.layers_per_block == 8
    
    def test_block_config_validation_invalid_division(self):
        """Test Block config validates layer/block divisibility."""
        with pytest.raises(ValueError, match="num_layers.*must be divisible by num_blocks"):
            AttnResConfig(
                variant=AttnResVariant.BLOCK,
                num_layers=30,
                num_blocks=8,
            )
    
    def test_config_validation_negative_values(self):
        """Test config validation rejects negative values."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            AttnResConfig(hidden_size=-1)
        
        with pytest.raises(ValueError, match="num_layers must be positive"):
            AttnResConfig(num_layers=0)
        
        with pytest.raises(ValueError, match="num_blocks must be positive"):
            AttnResConfig(num_blocks=-1)
    
    def test_config_validation_init_scale(self):
        """Test config validation for init_scale."""
        with pytest.raises(ValueError, match="init_scale must be positive"):
            AttnResConfig(init_scale=0)
        
        with pytest.raises(ValueError, match="init_scale must be positive"):
            AttnResConfig(init_scale=-0.01)
    
    def test_config_validation_dropout(self):
        """Test config validation for dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            AttnResConfig(dropout=-0.1)
        
        with pytest.raises(ValueError, match="dropout must be in"):
            AttnResConfig(dropout=1.5)
        
        # Valid dropout values
        AttnResConfig(dropout=0.0)
        AttnResConfig(dropout=0.5)
        AttnResConfig(dropout=1.0)
    
    def test_config_validation_apply_to(self):
        """Test config validation for apply_to."""
        with pytest.raises(ValueError, match="apply_to must be"):
            AttnResConfig(apply_to="invalid")
        
        # Valid values
        AttnResConfig(apply_to="attention")
        AttnResConfig(apply_to="mlp")
        AttnResConfig(apply_to="both")
    
    def test_config_string_representation(self):
        """Test config string representation."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=2048,
            num_layers=32,
        )
        config_str = str(config)
        assert "FULL" in config_str
        assert "2048" in config_str
        assert "32" in config_str


class TestAttnResCache:
    """Test AttnRes cache functionality."""
    
    def test_store_and_retrieve(self):
        """Test cache store and retrieve."""
        cache = AttnResCache()
        
        output = torch.randn(2, 16, 128)
        cache.store_output(0, output)
        
        retrieved = cache.get_layer_output(0)
        assert torch.allclose(retrieved, output)
    
    def test_get_nonexistent_output(self):
        """Test getting nonexistent output returns None."""
        cache = AttnResCache()
        
        retrieved = cache.get_layer_output(5)
        assert retrieved is None
    
    def test_get_all_preceding(self):
        """Test getting all preceding outputs."""
        cache = AttnResCache()
        
        for i in range(5):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        preceding = cache.get_all_preceding(5)
        assert len(preceding) == 5
        
        # Check order
        for i, out in enumerate(preceding):
            assert torch.allclose(out, cache.outputs[i])
    
    def test_get_preceding_in_range(self):
        """Test getting preceding outputs within range."""
        cache = AttnResCache()
        
        for i in range(10):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        # Get outputs from layer 3 to 6 (exclusive)
        preceding = cache.get_preceding_in_range(8, start_idx=3, end_idx=6)
        assert len(preceding) == 3
    
    def test_clear_cache(self):
        """Test clearing cache."""
        cache = AttnResCache()
        
        for i in range(5):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        assert len(cache) == 5
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get_layer_output(0) is None
    
    def test_cache_contains(self):
        """Test cache __contains__ method."""
        cache = AttnResCache()
        
        cache.store_output(0, torch.randn(2, 16, 128))
        cache.store_output(2, torch.randn(2, 16, 128))
        
        assert 0 in cache
        assert 1 not in cache
        assert 2 in cache
        assert 5 not in cache
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_offload(self):
        """Test CPU offloading for memory savings."""
        cache = AttnResCache(cpu_offload=True)
        output = torch.randn(2, 16, 128).cuda()
        cache.store_output(0, output)
        
        # Should be stored on CPU
        assert cache.outputs[0].device.type == 'cpu'
        
        # Retrieval should move back to GPU
        retrieved = cache.get_layer_output(0)
        assert retrieved.device.type == 'cuda'
        assert torch.allclose(retrieved, output)
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        cache = AttnResCache()
        
        for i in range(5):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        usage = cache.memory_usage()
        
        assert "gpu_memory_mb" in usage
        assert "cpu_memory_mb" in usage
        assert "total_memory_mb" in usage
        assert usage["total_memory_mb"] > 0


class TestFullAttnRes:
    """Test Full AttnRes implementation."""
    
    def test_initialization(self):
        """Test FullAttnRes initialization."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=4,
        )
        attn_res = FullAttnRes(config)
        
        assert attn_res.config == config
        assert attn_res.hidden_size == 128
        
        # Check that queries are registered
        for i in range(4):
            assert hasattr(attn_res, f'query_{i}')
    
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
    
    def test_attention_weights_sum_to_one(self):
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
        
        # Get attention weights
        weights = attn_res.get_attention_weights_for_analysis(cache, 3)
        
        assert weights is not None
        
        # Weights should sum to 1 over preceding dimension
        weight_sums = weights.sum(dim=2)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_num_preceding(self):
        """Test get_num_preceding method."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=8,
        )
        attn_res = FullAttnRes(config)
        
        assert attn_res.get_num_preceding(0) == 0
        assert attn_res.get_num_preceding(1) == 1
        assert attn_res.get_num_preceding(5) == 5
        assert attn_res.get_num_preceding(7) == 7
    
    def test_receptive_field(self):
        """Test get_effective_receptive_field method."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=8,
        )
        attn_res = FullAttnRes(config)
        
        # Receptive field grows linearly with depth
        assert attn_res.get_effective_receptive_field(0) == 1
        assert attn_res.get_effective_receptive_field(3) == 4
        assert attn_res.get_effective_receptive_field(7) == 8


class TestBlockAttnRes:
    """Test Block AttnRes implementation."""
    
    def test_initialization(self):
        """Test BlockAttnRes initialization."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        
        assert attn_res.config == config
        assert attn_res.hidden_size == 128
        assert attn_res.layers_per_block == 4
        assert attn_res.num_blocks == 2
        
        # Check that queries are registered
        for i in range(8):
            assert hasattr(attn_res, f'query_{i}')
    
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
    
    def test_block_start_layer(self):
        """Test block start layer calculation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            num_layers=32,
            num_blocks=8,
        )
        attn_res = BlockAttnRes(config)
        
        assert attn_res.get_block_start_layer(0) == 0
        assert attn_res.get_block_start_layer(1) == 4
        assert attn_res.get_block_start_layer(7) == 28
    
    def test_block_end_layer(self):
        """Test block end layer calculation."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            num_layers=32,
            num_blocks=8,
        )
        attn_res = BlockAttnRes(config)
        
        assert attn_res.get_block_end_layer(0) == 4
        assert attn_res.get_block_end_layer(1) == 8
        assert attn_res.get_block_end_layer(7) == 32
    
    def test_get_preceding_outputs_within_block(self):
        """Test getting preceding outputs within block."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        cache = AttnResCache()
        
        # Store outputs for all layers
        for i in range(8):
            cache.store_output(i, torch.randn(2, 16, 128))
        
        # Layer 5 is in block 1 (layers 4-7), should attend to layer 4 only
        preceding = attn_res.get_preceding_outputs(cache, 5)
        assert len(preceding) == 1
        
        # Layer 7 is in block 1, should attend to layers 4, 5, 6
        preceding = attn_res.get_preceding_outputs(cache, 7)
        assert len(preceding) == 3
        
        # Layer 0 is first in block 0, no preceding
        preceding = attn_res.get_preceding_outputs(cache, 0)
        assert len(preceding) == 0
    
    def test_forward_pass(self):
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
        
        # Layer 5 should only attend to layers within block 1
        last_output = torch.randn(batch_size, seq_len, 128)
        result = attn_res(last_output, cache, layer_idx=5)
        
        assert result.shape == last_output.shape
    
    def test_num_preceding_within_block(self):
        """Test get_num_preceding within block."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        
        # Block 0: layers 0-3
        assert attn_res.get_num_preceding(0) == 0
        assert attn_res.get_num_preceding(1) == 1
        assert attn_res.get_num_preceding(3) == 3
        
        # Block 1: layers 4-7
        assert attn_res.get_num_preceding(4) == 0  # First in block
        assert attn_res.get_num_preceding(5) == 1
        assert attn_res.get_num_preceding(7) == 3
    
    def test_receptive_field_limited_to_block(self):
        """Test that receptive field is limited to block size."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        
        # Receptive field limited by block size (4)
        assert attn_res.get_effective_receptive_field(0) == 1
        assert attn_res.get_effective_receptive_field(3) == 4
        assert attn_res.get_effective_receptive_field(4) == 1  # Reset at block boundary
        assert attn_res.get_effective_receptive_field(7) == 4


class TestAttnResUtils:
    """Test AttnRes utility functions."""
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=2048,
            num_layers=32,
        )
        
        usage = estimate_memory_usage(
            config,
            batch_size=4,
            seq_len=512,
            dtype=torch.float32,
        )
        
        assert usage["cache_full_mb"] > 0
        assert usage["total_full_mb"] > usage["cache_full_mb"]
    
    def test_analyze_attention_distribution(self):
        """Test attention distribution analysis."""
        weights = torch.softmax(torch.randn(2, 16, 8), dim=-1)
        
        analysis = analyze_attention_distribution(weights, layer_idx=5)
        
        assert "mean" in analysis
        assert "std" in analysis
        assert "entropy" in analysis
        assert "sparsity" in analysis
        assert analysis["layer_idx"] == 5
        
        # Mean should be close to 1/num_preceding for uniform distribution
        assert 0 < analysis["mean"] < 1
    
    def test_validate_attnres_setup_valid(self):
        """Test validation with valid setup."""
        config = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=4,
        )
        
        # Create a simple mock model
        model = nn.Linear(128, 128)
        
        is_valid, errors = validate_attnres_setup(model, config)
        
        # Should have some errors since model doesn't have AttnRes applied
        # But config validation should pass
        assert isinstance(errors, list)
    
    def test_validate_attnres_setup_invalid_config(self):
        """Test validation with invalid config."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=30,  # Not divisible by default num_blocks=8
            num_blocks=8,
        )
        
        model = nn.Linear(128, 128)
        
        is_valid, errors = validate_attnres_setup(model, config)
        
        assert not is_valid
        assert len(errors) > 0
        assert any("divisible" in err for err in errors)


class TestIntegration:
    """Integration tests for AttnRes components."""
    
    def test_full_attnres_gradient_flow(self):
        """Test that gradients flow through Full AttnRes."""
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
    
    def test_block_attnres_gradient_flow(self):
        """Test that gradients flow through Block AttnRes."""
        config = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=8,
            num_blocks=2,
        )
        attn_res = BlockAttnRes(config)
        cache = AttnResCache()
        
        # Store preceding outputs with gradients
        for i in range(8):
            output = torch.randn(2, 16, 128, requires_grad=True)
            cache.store_output(i, output)
        
        # Forward pass
        last_output = torch.randn(2, 16, 128, requires_grad=True)
        result = attn_res(last_output, cache, layer_idx=5)
        
        # Backward pass
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist
        assert last_output.grad is not None
    
    def test_full_vs_block_memory_efficiency(self):
        """Test that Block variant uses less memory than Full."""
        config_full = AttnResConfig(
            variant=AttnResVariant.FULL,
            hidden_size=128,
            num_layers=32,
        )
        
        config_block = AttnResConfig(
            variant=AttnResVariant.BLOCK,
            hidden_size=128,
            num_layers=32,
            num_blocks=8,
        )
        
        usage_full = estimate_memory_usage(config_full, batch_size=4, seq_len=512)
        usage_block = estimate_memory_usage(config_block, batch_size=4, seq_len=512)
        
        # Block should use significantly less memory for cache
        assert usage_block["cache_block_mb"] < usage_full["cache_full_mb"]
        assert usage_block["total_block_mb"] < usage_full["total_full_mb"]

# Attention Residuals (AttnRes) Quickstart Guide

**Based on:** arXiv:2603.15031  
**Version:** 1.0.0  
**Date:** March 19, 2026

---

## Overview

Attention Residuals (AttnRes) replace fixed-weight residual connections with learned, input-dependent attention weights over preceding layer outputs. This enables each layer to selectively aggregate earlier representations.

**Key Benefits:**
- Improved information flow across layers
- Better gradient propagation in deep networks
- Enhanced epistemic reliability (core to Diogenes)
- Potential 1.25x compute efficiency with Block variant

---

## Quick Start

### 1. Enable AttnRes via Configuration

Add to your `configs/config.yaml`:

```yaml
attnres:
  enabled: true
  variant: "full"  # or "block" for memory efficiency
  num_blocks: 8
  apply_to: "both"
  init_scale: 0.02
  dropout: 0.0
  use_layer_norm: true
  cache_on_cpu: false
```

### 2. Enable AttnRes in Code

```python
from diogenes.model import DiogenesModel

# Load model
model = DiogenesModel.from_pretrained("Qwen/Qwen3-0.6B")

# Enable AttnRes (Full variant)
model.enable_attnres(variant="full")

# Or use Block variant for memory efficiency
model.enable_attnres(variant="block", num_blocks=8)
```

### 3. Train with AttnRes

```bash
# Standard SFT training with AttnRes enabled
python src/diogenes/train_sft.py \
  --model_name Qwen/Qwen3-0.6B \
  --config configs/config.yaml \
  --output_dir models/sft_attnres_test
```

---

## AttnRes Variants

### Full AttnRes

Each layer attends over **all** preceding layer outputs.

**Pros:**
- Maximum flexibility
- Best for research/experimentation
- Full receptive field

**Cons:**
- Higher memory usage: O(L·d)
- More computation: O(L²)

**Use when:**
- Experimenting with AttnRes
- VRAM is not constrained
- Maximum performance is needed

```python
model.enable_attnres(variant="full")
```

### Block AttnRes

Layers grouped into **blocks**. Each layer attends over outputs within its block only.

**Pros:**
- Memory efficient: O(N·d) where N << L
- Faster computation
- Suitable for production

**Cons:**
- Limited receptive field per block
- Slightly less flexible

**Use when:**
- Memory-constrained (e.g., RTX 3050 8GB)
- Production deployment
- Training large models

```python
# 32 layers / 8 blocks = 4 layers per block
model.enable_attnres(variant="block", num_blocks=8)
```

---

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable/disable AttnRes |
| `variant` | str | "full" | "full" or "block" |
| `num_blocks` | int | 8 | Number of blocks (Block variant only) |
| `apply_to` | str | "both" | "attention", "mlp", or "both" |
| `init_scale` | float | 0.02 | Initialization scale for query weights |
| `dropout` | float | 0.0 | Dropout rate for attention weights |
| `use_layer_norm` | bool | true | Apply LayerNorm before attention |
| `cache_on_cpu` | bool | false | Cache outputs on CPU to save VRAM |

---

## Memory Considerations

### Full AttnRes Memory Usage

For Qwen3-32B (hidden_size=2048, num_layers=32):

| Batch Size | Seq Len | Cache Memory | Total Memory |
|------------|---------|--------------|--------------|
| 4 | 512 | ~256 MB | ~384 MB |
| 8 | 512 | ~512 MB | ~768 MB |
| 4 | 1024 | ~512 MB | ~768 MB |
| 8 | 1024 | ~1024 MB | ~1.5 GB |

### Block AttnRes Memory Usage (8 blocks)

| Batch Size | Seq Len | Cache Memory | Total Memory |
|------------|---------|--------------|--------------|
| 4 | 512 | ~32 MB | ~64 MB |
| 8 | 512 | ~64 MB | ~128 MB |
| 4 | 1024 | ~64 MB | ~128 MB |
| 8 | 1024 | ~128 MB | ~256 MB |

**Note:** Block variant uses **8x less memory** for cache.

### Memory Optimization Tips

1. **Use Block variant** for large models
2. **Enable CPU offloading** if VRAM is constrained:
   ```python
   model.enable_attnres(variant="block", cache_on_cpu=True)
   ```
3. **Reduce batch size** if OOM errors occur
4. **Use gradient checkpointing** in training scripts

---

## Testing AttnRes

### Run Unit Tests

```bash
# Run all AttnRes tests
pytest tests/test_attnres.py -v

# Run specific test class
pytest tests/test_attnres.py::TestFullAttnRes -v

# Run with coverage
pytest tests/test_attnres.py --cov=diogenes.attnres
```

### Integration Test

```python
import torch
from diogenes.model import DiogenesModel
from diogenes.attnres import AttnResConfig, AttnResVariant

# Load small test model
model = DiogenesModel.from_pretrained("Qwen/Qwen3-0.6B")

# Enable AttnRes
model.enable_attnres(variant="full")

# Test inference
from diogenes.inference import DiogenesInference
inference = DiogenesInference(model)
result = inference.generate("Test AttnRes integration")
print(f"Response: {result.text}")
print(f"Mode: {result.epistemic_mode.value}")
```

---

## Monitoring AttnRes

### Check Attention Weights

```python
from diogenes.attnres.utils import analyze_attention_distribution

# Get attention weights from wrapper
weights = model.model.attn_res.get_attention_weights_for_analysis(
    model.model.attn_res.cache,
    layer_idx=10,
)

# Analyze distribution
analysis = analyze_attention_distribution(weights, layer_idx=10)
print(f"Mean attention: {analysis['mean']:.4f}")
print(f"Entropy: {analysis['entropy']:.4f}")
print(f"Sparsity: {analysis['sparsity']:.2%}")
```

### Visualize Attention Patterns

```python
from diogenes.attnres.utils import create_layer_visualization

config = AttnResConfig(
    variant=AttnResVariant.FULL,
    hidden_size=2048,
    num_layers=32,
)

visualization = create_layer_visualization(
    config,
    model.model.attn_res.cache,
    model.model.attn_res,
)
print(visualization)
```

---

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1:** Use Block variant
```python
model.enable_attnres(variant="block", num_blocks=8)
```

**Solution 2:** Enable CPU offloading
```python
model.enable_attnres(cache_on_cpu=True)
```

**Solution 3:** Reduce batch size or sequence length

### Training Instability

**Solution 1:** Reduce init_scale
```python
model.enable_attnres(init_scale=0.01)
```

**Solution 2:** Add dropout
```python
model.enable_attnres(dropout=0.1)
```

**Solution 3:** Enable layer normalization
```python
model.enable_attnres(use_layer_norm=True)
```

### No Improvement in Metrics

**Possible causes:**
- Model too small (AttnRes benefits scale with depth)
- Training not converged (train longer)
- Wrong hyperparameters (adjust init_scale, dropout)

**Recommendations:**
- Try Block variant with different num_blocks
- Run ablation studies
- Check attention weight distributions

---

## Performance Benchmarks

### Expected Overhead

| Variant | Compute Overhead | Memory Overhead |
|---------|------------------|-----------------|
| Full | +15-20% | +10-15% |
| Block (8 blocks) | +5-8% | +2-5% |

### Expected Improvements (Based on Paper)

| Metric | Expected Change |
|--------|-----------------|
| Pass@1 | ±0% (no degradation) |
| Hallucination Rate | -5% to -10% |
| ECE | -10% to -15% |
| Abstention AUROC | +5% to +10% |
| Mode Accuracy | +5% to +10% |

---

## Advanced Usage

### Custom AttnRes Configuration

```python
from diogenes.attnres import AttnResConfig, AttnResVariant, AttnResWrapper

# Create custom config
config = AttnResConfig(
    variant=AttnResVariant.BLOCK,
    hidden_size=2048,
    num_layers=32,
    num_blocks=8,
    init_scale=0.01,
    dropout=0.1,
    use_layer_norm=True,
    cache_on_cpu=True,
    apply_to="both",
)

# Apply to model
wrapper = AttnResWrapper(model.model, config)
model.model = wrapper.wrap()
```

### Gradient Analysis

```python
# Check gradient norms for AttnRes parameters
for name, param in model.model.named_parameters():
    if 'attn_res' in name and param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.4f}")
```

### Export Attention Weights

```python
import json

# Collect attention weights for all layers
attention_data = {}
for layer_idx in range(model.model.config.num_hidden_layers):
    weights = model.model.attn_res.get_attention_weights_for_analysis(
        model.model.attn_res.cache,
        layer_idx,
    )
    if weights is not None:
        attention_data[f"layer_{layer_idx}"] = {
            "mean": weights.mean().item(),
            "std": weights.std().item(),
            "max": weights.max().item(),
        }

# Save to file
with open("attention_weights.json", "w") as f:
    json.dump(attention_data, f, indent=2)
```

---

## References

1. **Attention Residuals Paper:** arXiv:2603.15031 (Kimi Linear team)
2. **Implementation Plan:** `docs/ATTNRES_IMPLEMENTATION_PLAN.md`
3. **Unit Tests:** `tests/test_attnres.py`
4. **Source Code:** `src/diogenes/attnres/`

---

## Next Steps

1. **Start with Full variant** for initial testing
2. **Switch to Block variant** for production training
3. **Monitor epistemic metrics** during training
4. **Run ablation studies** to optimize hyperparameters
5. **Document results** for team review

For detailed implementation information, see `docs/ATTNRES_IMPLEMENTATION_PLAN.md`.

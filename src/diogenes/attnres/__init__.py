"""Attention Residuals (AttnRes) for Diogenes.

This module implements Attention Residuals as described in arXiv:2603.15031.
AttnRes replaces fixed-weight residual connections with learned, input-dependent
attention weights over preceding layer outputs.

Example usage:
    from diogenes.attnres import AttnResConfig, AttnResWrapper, AttnResVariant
    
    config = AttnResConfig(
        variant=AttnResVariant.FULL,
        hidden_size=2048,
        num_layers=32,
    )
    
    wrapper = AttnResWrapper(model, config)
    wrapped_model = wrapper.wrap()
"""

from .config import AttnResConfig, AttnResVariant
from .core import AttnResBase, AttnResWrapper
from .full import FullAttnRes
from .block import BlockAttnRes
from .cache import AttnResCache

__all__ = [
    "AttnResConfig",
    "AttnResVariant",
    "AttnResBase",
    "AttnResWrapper",
    "FullAttnRes",
    "BlockAttnRes",
    "AttnResCache",
]

__version__ = "1.0.0"

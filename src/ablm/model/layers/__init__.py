"""Architecture building blocks for the ABLM backbone.

These are the config-driven components the architecture screen sweeps —
normalization, RoPE, embedding, feed-forward, attention, the transformer
block/stack, and the attention-mask helper. `modeling_ablm.py` assembles them
into the public `Ablm*` classes; keeping them here separates the experiment
surface from the HuggingFace contract at the package root.
"""

from __future__ import annotations

from .attention import AblmAttention
from .embedding import AblmEmbedding, cls_pool, mean_pool
from .ffn import MLP, GatedFFN, make_ffn, round_up_to
from .masking import prepare_attention_mask
from .norm import AblmLayerNorm, AblmRMSNorm, make_norm
from .rope import RotaryEmbedding
from .transformer import AblmBlock, AblmStack

__all__ = [
    "AblmAttention",
    "AblmBlock",
    "AblmEmbedding",
    "AblmLayerNorm",
    "AblmRMSNorm",
    "AblmStack",
    "GatedFFN",
    "MLP",
    "RotaryEmbedding",
    "cls_pool",
    "make_ffn",
    "make_norm",
    "mean_pool",
    "prepare_attention_mask",
    "round_up_to",
]

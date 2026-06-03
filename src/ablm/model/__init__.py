"""Public surface for the ABLM model package.

ABLM is an encoder-only protein language model built on a configurable
pre-norm transformer backbone. A single :class:`AblmConfig` selects every
architectural variant — norm operator (LayerNorm / RMSNorm), norm placement
strategy (pre / sandwich / hybrid / post-SDPA), full vs. partial RoPE, optional
QK-norm, SwiGLU feed-forward, optional Canon depthwise convolutions, and
sqrt-depth residual scaling — so the same code path covers the whole design
space. Attention runs through PyTorch's ``scaled_dot_product_attention`` — a
fused FlashAttention / memory-efficient kernel on CUDA, the math backend on
CPU. The package exposes the backbone
(:class:`AblmModel`) and the task heads (:class:`AblmForMaskedLM`,
:class:`AblmForSequenceClassification`, :class:`AblmForTokenClassification`),
all registered with the HuggingFace Auto* classes and carrying an ESM-C-style
``tokenize`` / ``encode`` / ``logits`` convenience API via
:class:`EsmcCompatMixin`. Internal building blocks (norm, rope, embedding, ffn,
conv, attention, transformer, masking) live in their own modules and are
re-exported here for convenience.

See ``docs/MODEL_ARCHITECTURE.md`` for the full architecture specification.
"""

from __future__ import annotations

from .attention import AblmAttention
from .configuration_ablm import AblmConfig
from .conv import CanonConv, resolve_canon_kernel_sizes
from .embedding import AblmEmbedding, cls_pool, mean_pool
from .ffn import SwiGLU, make_ffn, round_up_to
from .modeling_ablm import (
    AblmForMaskedLM,
    AblmForSequenceClassification,
    AblmForTokenClassification,
    AblmMLMHead,
    AblmModel,
    AblmPreTrainedModel,
    EsmcCompatMixin,
)
from .outputs import LogitsConfig, LogitsOutput
from .tokenization_ablm import AblmTokenizerFast
from .transformer import AblmBlock, AblmStack

__all__ = [
    "CanonConv",
    "EsmcCompatMixin",
    "LogitsConfig",
    "LogitsOutput",
    "AblmAttention",
    "AblmBlock",
    "AblmConfig",
    "AblmEmbedding",
    "AblmForMaskedLM",
    "AblmForSequenceClassification",
    "AblmForTokenClassification",
    "AblmMLMHead",
    "AblmModel",
    "AblmPreTrainedModel",
    "AblmStack",
    "AblmTokenizerFast",
    "SwiGLU",
    "cls_pool",
    "make_ffn",
    "mean_pool",
    "resolve_canon_kernel_sizes",
    "round_up_to",
]

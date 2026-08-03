"""Public surface for the ABLM model package.

ABLM is an encoder-only protein language model built on a configurable
pre-norm transformer backbone. A single :class:`AblmConfig` selects every
architectural variant — norm operator (LayerNorm / RMSNorm), norm placement
strategy (pre / sandwich / hybrid / post-SDPA), full vs. partial RoPE, optional
QK-norm, gated or non-gated feed-forward (swiglu / geglu / reglu / gelu_mlp), and
sqrt-depth residual scaling — so the same code path covers the whole design
space. Attention runs through PyTorch's ``scaled_dot_product_attention`` — a
fused FlashAttention / memory-efficient kernel on CUDA, the math backend on CPU.
The package exposes the backbone (:class:`AblmModel`) and the task heads
(:class:`AblmForMaskedLM`, :class:`AblmForSequenceClassification`,
:class:`AblmForTokenClassification`), all registered with the HuggingFace Auto*
classes. The architecture building blocks (norm, rope, embedding, ffn,
attention, transformer, masking) live in the :mod:`.layers` subpackage and are
re-exported here for convenience.
"""

from __future__ import annotations

from .configuration_ablm import AblmConfig
from .layers import (
    MLP,
    AblmAttention,
    AblmBlock,
    AblmEmbedding,
    AblmStack,
    GatedFFN,
    cls_pool,
    make_ffn,
    mean_pool,
    round_up_to,
)
from .modeling_ablm import (
    AblmForMaskedLM,
    AblmForSequenceClassification,
    AblmForTokenClassification,
    AblmMLMHead,
    AblmModel,
    AblmPreTrainedModel,
)
from .tokenization_ablm import AblmTokenizerFast

__all__ = [
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
    "GatedFFN",
    "MLP",
    "cls_pool",
    "make_ffn",
    "mean_pool",
    "round_up_to",
]

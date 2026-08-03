"""Token embedding plus mean / CLS pooling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .norm import make_norm

if TYPE_CHECKING:
    from .configuration_ablm import AblmConfig

__all__ = ["AblmEmbedding", "mean_pool", "cls_pool"]


# ESM-family masked-token-dropout train ratio: the fraction of input positions
# replaced by <mask> during MLM training (mask_prob 0.15 * mask_token_prob 0.8).
# Hardcoded to match ESM-2 / ESM-C; the data collator uses the same split.
_TOKEN_DROPOUT_MASK_RATIO_TRAIN = 0.15 * 0.8


class AblmEmbedding(nn.Module):
    """Token embedding lookup with optional token-dropout and post-embedding norm.

    The token IDs map straight into a learnable `(V, D)` embedding table; no
    positional embedding is added at this stage (RoPE is applied inside the
    attention sublayer).

    When `config.token_dropout` is `True` (ESM-2 / ESM-C behavior), `<mask>`
    positions are zeroed and the whole tensor is rescaled by
    `(1 - r_train) / (1 - r_observed)` — where `r_train` is the fixed training
    mask ratio and `r_observed` the per-sequence observed mask fraction — so the
    expected embedding magnitude is invariant to masking. At inference (no mask
    tokens) this scales embeddings by `1 - r_train`, matching ESM.

    When `config.post_embed_norm` is `True`, a norm is applied to the result
    before it enters the residual stream.
    """

    def __init__(self, config: AblmConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_dropout = bool(getattr(config, "token_dropout", False))
        self.mask_token_id = getattr(config, "mask_token_id", None)
        if config.post_embed_norm:
            self.post_norm: nn.Module = make_norm(
                config.norm_type,
                config.hidden_size,
                eps=config.norm_eps,
                bias=getattr(config, "norm_bias", True),
            )
        else:
            self.post_norm = nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # (B, T) int64 -> (B, T, D)
        x = self.embed_tokens(input_ids)
        if self.token_dropout:
            x = self._apply_token_dropout(x, input_ids, attention_mask)
        return self.post_norm(x)

    def _apply_token_dropout(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """ESM-style masked-token dropout: zero `<mask>` rows, then rescale."""
        if self.mask_token_id is None:
            raise ValueError("token_dropout requires config.mask_token_id to be set.")
        is_mask = input_ids == self.mask_token_id  # (B, T)
        x = x.masked_fill(is_mask.unsqueeze(-1), 0.0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        src_lengths = attention_mask.sum(dim=-1)  # (B,)
        observed = (is_mask & attention_mask.bool()).sum(dim=-1)  # (B,)
        mask_ratio_observed = observed.to(x.dtype) / src_lengths.to(x.dtype).clamp(min=1.0)
        denom = (1.0 - mask_ratio_observed).clamp(min=1e-6)
        scale = (1.0 - _TOKEN_DROPOUT_MASK_RATIO_TRAIN) / denom
        return x * scale[:, None, None].to(x.dtype)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean over the sequence dimension.

    Args:
        hidden: `(B, T, D)` tensor.
        attention_mask: `(B, T)` tensor with `1` at real tokens and `0` at
            pad positions. Cast to `hidden`'s dtype before the multiply.

    Returns:
        `(B, D)` tensor. Rows whose mask is entirely zero are returned as
        zeros (the denominator is clamped to at least `1`).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).to(hidden.dtype).clamp(min=1)
    return summed / counts


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Return the `<cls>` position (token index 0) of every row.

    Args:
        hidden: `(B, T, D)` tensor.

    Returns:
        `(B, D)` tensor.
    """
    return hidden[:, 0, :]

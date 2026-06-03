"""Multi-head self-attention.

The fast path is `torch.nn.functional.scaled_dot_product_attention`, which at
runtime auto-selects the fastest fused backend (FlashAttention / cuDNN /
memory-efficient on CUDA, math on CPU) — so this single call *is* the attention
optimization; no kernel registry or `torch.compile` is needed. The only reason
for a second path is `output_attentions=True`: SDPA does not expose the
attention weights, so a manual fp32 softmax runs instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .norm import make_norm
from .rope import RotaryEmbedding

if TYPE_CHECKING:
    from .configuration_ablm import AblmConfig

__all__ = ["AblmAttention"]


class AblmAttention(nn.Module):
    """Multi-head self-attention with QK/V-norm and RoPE.

    Both compute paths share one set of parameters and pre-attention transforms
    (Q/K/V projections, optional QK/V norm, RoPE). The default path calls
    `F.scaled_dot_product_attention` with a `(B, 1, 1, T)` boolean key-padding
    mask; when `output_attentions=True` the manual softmax path runs instead.

    Norm wiring (controlled by `config.norm_strategy`):

    * `qk_norm=True` always installs `q_norm` and `k_norm` on the per-head
      dimension (`d_head`).
    * Under `norm_strategy == "hybrid"`, an additional `v_norm` is installed on
      `d_head` — this realises the paper's "QKV-norm" main method.
    * Under every other strategy, `v_norm` is `nn.Identity()`.

    The output projection (`o_proj`) is marked `_is_residual_writer = True` so
    `AblmPreTrainedModel._init_weights` can apply the `1/sqrt(2L)` residual-stream
    scaling.
    """

    def __init__(self, config: AblmConfig) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        if head_dim * num_heads != hidden_size:
            raise ValueError(
                f"head_dim ({head_dim}) * num_attention_heads ({num_heads}) must equal "
                f"hidden_size ({hidden_size})."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.head_dim = head_dim
        self.attention_dropout = float(config.attention_dropout)
        self.hidden_dropout = float(config.hidden_dropout)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Picked up by AblmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling.
        self.o_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

        norm_bias = getattr(config, "norm_bias", True)
        if bool(config.qk_norm):
            self.q_norm: nn.Module = make_norm(
                config.norm_type, head_dim, eps=config.norm_eps, bias=norm_bias
            )
            self.k_norm: nn.Module = make_norm(
                config.norm_type, head_dim, eps=config.norm_eps, bias=norm_bias
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Hybrid strategy ("QKV-norm") also norms V; every other strategy leaves V alone.
        if config.norm_strategy == "hybrid":
            self.v_norm: nn.Module = make_norm(
                config.norm_type, head_dim, eps=config.norm_eps, bias=norm_bias
            )
        else:
            self.v_norm = nn.Identity()

        self.rotary = RotaryEmbedding(
            head_dim=head_dim,
            rope_dim=config.rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to Q/K/V and reshape from `(B, T, D)` to `(B, H, T, d_head)`."""
        batch, seq_len, _ = x.shape
        h, d = self.num_attention_heads, self.head_dim

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, h, d).transpose(1, 2)

        return split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

    def _output_projection(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Reshape `(B, H, T, d_head)` back to `(B, T, D)` and project."""
        batch, _, seq_len, _ = attn_out.shape
        merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(merged)

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Manual scaled-dot-product attention with an fp32 softmax; returns weights."""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        scores = scores.masked_fill((attention_mask == 0)[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_dropped = F.dropout(attn, p=self.attention_dropout, training=self.training)
        out = torch.matmul(attn_dropped.to(v.dtype), v)  # (B, H, T, d_head)
        return out, attn

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run multi-head attention.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` tensor with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the fp32 `(B, H, T, T)` weights
                via the manual softmax path (SDPA cannot expose them).

        Returns:
            `(output, attn_weights_or_None)`. `output` is `(B, T, D)`.
        """
        q, k, v = self._project_qkv(x)
        # Per-head QK/V norm (Identity unless enabled), then RoPE on Q/K.
        q, k = self.q_norm(q), self.k_norm(k)
        v = self.v_norm(v)
        q, k = self.rotary.apply_rotary(q, k)

        if output_attentions:
            out, attn = self._manual_attention(q, k, v, attention_mask)
        else:
            # (B, 1, 1, T) boolean key-padding mask (True = attend). Masking only
            # keys keeps every query row non-empty, so softmax never sees an
            # all-masked row. SDPA auto-selects the fastest fused backend.
            key_mask = (attention_mask == 1)[:, None, None, :]
            dropout_p = self.attention_dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=dropout_p)
            attn = None

        out = self._output_projection(out)
        out = F.dropout(out, p=self.hidden_dropout, training=self.training)
        return out, attn

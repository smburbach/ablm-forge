"""AblmBlock and AblmStack — the repeating block and backbone holder."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from .attention import AblmAttention
from .embedding import AblmEmbedding
from .ffn import make_ffn
from .masking import prepare_attention_mask
from .norm import make_norm

if TYPE_CHECKING:
    from .configuration_ablm import AblmConfig

__all__ = ["AblmBlock", "AblmStack"]


class AblmBlock(nn.Module):
    """One repeating encoder block: attention + FFN sublayers, configurable norms.

    Wires the four `norm_strategy` variants (`pre`, `sandwich`, `hybrid`,
    `post_sdpa`), the residual-stream scaling factor `alpha`, and an opt-in
    gradient checkpoint dispatch.
    """

    def __init__(self, config: AblmConfig, layer_idx: int) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.norm_strategy = config.norm_strategy
        self.residual_scaling = config.residual_scaling
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))

        if config.residual_scaling == "sqrt_num_layers":
            alpha_val = 1.0 / math.sqrt(config.num_hidden_layers)
        elif config.residual_scaling == "none":
            alpha_val = 1.0
        else:
            raise ValueError(
                f"Unknown residual_scaling {config.residual_scaling!r}; "
                "expected 'sqrt_num_layers' or 'none'."
            )
        # Register as a persistent buffer (scalar tensor) rather than a plain Python float.
        #
        # Why a buffer at all: torch.compile + DDP (DDPOptimizer) lifts plain-float
        # module attributes as graph inputs and may place them in subgraph outputs when
        # partitioning at bucket boundaries. aot_autograd then fails with
        # "AttributeError: 'float' has no attribute 'meta'" because it expects every
        # output value to be an FX Node. A buffer is a proper tensor throughout.
        #
        # Why persistent (not persistent=False): HuggingFace's from_pretrained fast-init
        # path creates model tensors uninitialized (torch.empty semantics) and only
        # restores persistent buffers from the saved state dict. Non-persistent buffers
        # stay as garbage after loading, producing near-zero alpha and broken outputs.
        self.register_buffer("alpha", torch.tensor(alpha_val), persistent=True)

        if config.norm_strategy not in {"pre", "sandwich", "hybrid", "post_sdpa"}:
            raise ValueError(
                f"Unknown norm_strategy {config.norm_strategy!r}; "
                "expected one of 'pre', 'sandwich', 'hybrid', 'post_sdpa'."
            )

        norm_bias = getattr(config, "norm_bias", True)

        # Attention pre-norm: present under every strategy except hybrid.
        if config.norm_strategy != "hybrid":
            self.attn_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps, bias=norm_bias
            )

        # Attention module self-configures v_norm under hybrid.
        self.attention = AblmAttention(config)

        # FFN pre-norm and FFN module are always present.
        self.ffn_norm: nn.Module = make_norm(
            config.norm_type, config.hidden_size, eps=config.norm_eps, bias=norm_bias
        )
        self.ffn = make_ffn(config)

        # Strategy-specific post-norms.
        if config.norm_strategy == "sandwich":
            self.attn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps, bias=norm_bias
            )
            self.ffn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps, bias=norm_bias
            )
        elif config.norm_strategy == "post_sdpa":
            self.attn_post_norm = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps, bias=norm_bias
            )

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Attention sublayer. Hybrid feeds raw `x` (QKV-norm lives inside the
        # attention module); every other strategy applies the outer pre-norm.
        a_in = x if self.norm_strategy == "hybrid" else self.attn_norm(x)

        attn_out, attn_weights = self.attention(a_in, attention_mask, output_attentions)

        if self.norm_strategy in {"sandwich", "post_sdpa"}:
            attn_out = self.attn_post_norm(attn_out)
        h = x + self.alpha * attn_out

        # FFN sublayer.
        h_norm = self.ffn_norm(h)
        ffn_out = self.ffn(h_norm)

        if self.norm_strategy == "sandwich":
            ffn_out = self.ffn_post_norm(ffn_out)
            y = h + self.alpha * ffn_out
        elif self.norm_strategy == "hybrid":
            # Hybrid reuses Norm(h) as both FFN input and FFN-side residual stream.
            y = h_norm + self.alpha * ffn_out
        else:  # "pre" or "post_sdpa"
            y = h + self.alpha * ffn_out

        return y, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one transformer block.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` mask with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the per-block attention
                weights from `AblmAttention` (forces the SDPA fallback).

        Returns:
            `(y, attn_weights_or_None)` — `y` has shape `(B, T, D)`;
            `attn_weights_or_None` has shape `(B, H, T, T)` when requested.
        """
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint.checkpoint(
                self._forward_impl,
                x,
                attention_mask,
                output_attentions,
                use_reentrant=False,
            )
        return self._forward_impl(x, attention_mask, output_attentions)


class AblmStack(nn.Module):
    """Encoder backbone: token embedding, N × AblmBlock, final norm.

    Forward returns `(last_hidden, hidden_states_or_None, attentions_or_None)`:

    * `last_hidden`: `(B, T, D)` post-final-norm activations.
    * `hidden_states`: `(L + 1)`-tuple of `(B, T, D)` tensors (the
      post-embedding state, then the output of each block, pre-final-norm).
      `None` when `output_hidden_states=False`.
    * `attentions`: `L`-tuple of `(B, H, T, T)` tensors (each entry is `None`
      when a block returned no weights). `None` when `output_attentions=False`.
    """

    def __init__(self, config: AblmConfig) -> None:
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))

        self.embed_tokens = AblmEmbedding(config)
        self.layers = nn.ModuleList(
            [AblmBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = make_norm(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
            bias=getattr(config, "norm_bias", True),
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Toggle gradient checkpointing on every block in the stack."""
        self.gradient_checkpointing = enabled
        for block in self.layers:
            block.gradient_checkpointing = enabled  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        tuple[torch.Tensor | None, ...] | None,
    ]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is not None:
            x = inputs_embeds
            batch_size, seq_len, _ = x.shape
        else:
            assert input_ids is not None  # guaranteed by the exactly-one check above
            batch_size, seq_len = input_ids.shape
            # Pass the raw (B, T) padding mask so token-dropout can rescale by the
            # observed mask fraction (no-op when token_dropout is off).
            x = self.embed_tokens(input_ids, attention_mask)

        device = x.device
        attention_mask = prepare_attention_mask(attention_mask, batch_size, seq_len, device)

        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states else None
        attentions: tuple[torch.Tensor | None, ...] | None = () if output_attentions else None

        for block in self.layers:
            x, attn = block(x, attention_mask, output_attentions)
            if hidden_states is not None:
                hidden_states = hidden_states + (x,)
            if attentions is not None:
                attentions = attentions + (attn,)

        last_hidden = self.final_norm(x)
        return last_hidden, hidden_states, attentions

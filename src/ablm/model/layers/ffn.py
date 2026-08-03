"""Feed-forward blocks (gated and non-gated) and the make_ffn factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.activations import ACT2FN

from ..config_types import FfnActivation

if TYPE_CHECKING:
    from ..configuration_ablm import AblmConfig

__all__ = ["MLP", "GatedFFN", "make_ffn", "round_up_to"]

# FfnActivation -> (structure, ACT2FN key). This registry is the single source of
# truth for the FFN variants: the `FfnActivation` enum declares them and this maps
# each to its implementation. Adding a gated variant is a line here + an enum member,
# not a class: the gate activation is the only thing that differs.
_FFN_VARIANTS: dict[FfnActivation, tuple[str, str]] = {
    FfnActivation.SWIGLU: ("gated", "silu"),
    FfnActivation.GEGLU: ("gated", "gelu"),
    FfnActivation.REGLU: ("gated", "relu"),
    FfnActivation.GELU_MLP: ("mlp", "gelu"),
}


class GatedFFN(nn.Module):
    """Gated feed-forward block: ``down(act(gate(x)) * up(x))``.

    Three linear projections share no parameters and all use the same `bias`
    setting. `activation` is an `ACT2FN` key naming the gate non-linearity:
    `silu` gives SwiGLU, `gelu` gives GeGLU, `relu` gives ReGLU.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = ACT2FN[activation]
        # Writes back into the residual stream: picked up by
        # AblmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling.
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """Non-gated feed-forward block: ``down(act(up(x)))``.

    Two projections rather than three — this is a different *structure* from
    `GatedFFN`, not a different activation, which is why it is its own class.
    Used for ESM-2-style plain-GELU FFN arms.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = ACT2FN[activation]
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(self.act(self.up_proj(x)))


def round_up_to(value: int, multiple: int) -> int:
    """Round `value` up to the nearest non-zero positive `multiple`.

    Used to align `intermediate_size` to a tensor-core / memory-friendly
    boundary when the config does not pin it explicitly.
    """
    return ((value + multiple - 1) // multiple) * multiple


def make_ffn(config: AblmConfig) -> nn.Module:
    """Construct the FFN operator selected by `config.ffn_activation`.

    New variants are registered in `_FFN_VARIANTS`; the rest of the model stays
    agnostic to which one is in use.

    Args:
        config: Carries `hidden_size`, `intermediate_size`, `ffn_activation`,
            and `ffn_bias`.

    Returns:
        A `nn.Module` mapping `(B, T, D) -> (B, T, D)`.

    Raises:
        ValueError: For an unrecognized `ffn_activation`.
    """
    try:
        structure, activation = _FFN_VARIANTS[config.ffn_activation]
    except KeyError:
        raise ValueError(
            f"Unknown ffn_activation {config.ffn_activation!r}; "
            f"expected one of {tuple(_FFN_VARIANTS)}."
        ) from None
    cls = GatedFFN if structure == "gated" else MLP
    return cls(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        activation=activation,
        bias=config.ffn_bias,
    )

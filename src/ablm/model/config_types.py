"""Categorical config values as `StrEnum`s (single source of truth).

Each enum is the authoritative set of valid values for one `AblmConfig` field:
`AblmConfig` validates by coercing through the enum (an unknown value raises
`ValueError` for free), and the model code compares against the members instead
of bare string literals. Because `StrEnum` is a `str` subclass, values serialize
to `config.json` as their plain string (`NormType.LAYERNORM` → `"layernorm"`) and
equality against a raw string still holds, so reloaded configs behave identically.

`FfnActivation` is also the key type of `ffn.py`'s `_FFN_VARIANTS` registry, so the
declared variants and their implementations share one source with no mirror list.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "NormType",
    "NormStrategy",
    "ResidualScaling",
    "FfnActivation",
    "MlmHeadActivation",
    "ClassifierPool",
]


class NormType(StrEnum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class NormStrategy(StrEnum):
    PRE = "pre"
    SANDWICH = "sandwich"
    HYBRID = "hybrid"
    POST_SDPA = "post_sdpa"


class ResidualScaling(StrEnum):
    SQRT_NUM_LAYERS = "sqrt_num_layers"
    NONE = "none"


class FfnActivation(StrEnum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    GELU_MLP = "gelu_mlp"


class MlmHeadActivation(StrEnum):
    GELU = "gelu"
    SILU = "silu"
    RELU = "relu"


class ClassifierPool(StrEnum):
    MEAN = "mean"
    CLS = "cls"

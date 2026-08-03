"""Declarative registry of muP-correct, kernel-optimized default configs.

Each entry lists only the differing knobs (layers, hidden). ``from_preset``
applies the shared muP base (d0 = 512) and lets ``AblmConfig`` derive head_dim,
the 8/3 SwiGLU FFN, and the muP multipliers — nothing derived is hand-typed.
"""

from __future__ import annotations

from .configuration_ablm import AblmConfig

__all__ = ["PRESETS", "from_preset"]

_MUP_BASE_HIDDEN_SIZE = 512

_PRESETS: dict[str, dict[str, int]] = {
    "35m": {"num_hidden_layers": 16, "hidden_size": 512},
    "150m": {"num_hidden_layers": 24, "hidden_size": 768},
    "300m": {"num_hidden_layers": 30, "hidden_size": 1024},
    "600m": {"num_hidden_layers": 36, "hidden_size": 1152},
}

PRESETS: tuple[str, ...] = tuple(_PRESETS)


def from_preset(name: str, **overrides) -> AblmConfig:
    """Build the named preset's ``AblmConfig`` (muP enabled), applying overrides.

    Args:
        name: One of :data:`PRESETS`.
        **overrides: Any ``AblmConfig`` kwarg; overrides the preset value.

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {PRESETS}")
    kwargs: dict = {
        **_PRESETS[name],
        "num_attention_heads": _PRESETS[name]["hidden_size"] // 64,
        "mup_enabled": True,
        "mup_base_hidden_size": _MUP_BASE_HIDDEN_SIZE,
    }
    kwargs.update(overrides)
    return AblmConfig(**kwargs)

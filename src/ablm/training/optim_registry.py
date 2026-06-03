"""Pluggable optimizer registry.

Each optimizer name resolves to an :class:`OptimizerSpec` describing how the
stock `transformers.Trainer` should realize it — *without* any Trainer subclass:

* **HF-native** optimizers set `hf_optim` (e.g. ``"adamw_torch_fused"``); the
  caller passes it straight to `TrainingArguments.optim` and HF builds it
  (post-FSDP-shard, with its own decay grouping).
* **Custom** optimizers set `builder`; the caller builds the instance and hands
  it to the Trainer via ``optimizer_cls_and_kwargs`` / the ``optimizers=`` tuple.

Add a new optimizer by calling :func:`register_hf_optimizer` or decorating a
builder with :func:`register_custom_optimizer` — no other code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .muon import build_muon_optimizer

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn

__all__ = [
    "OPTIMIZERS",
    "OptimizerSettings",
    "OptimizerSpec",
    "available_optimizers",
    "build_optimizer",
    "register_custom_optimizer",
    "register_hf_optimizer",
    "resolve_optimizer",
]


@dataclass(frozen=True)
class OptimizerSettings:
    """Common optimizer hyperparameters threaded from training config.

    Mirrors the fields the stock `transformers.TrainingArguments` exposes, so a
    custom optimizer is configured from the same knobs as the HF-native ones.
    """

    lr: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # Muon-specific (ignored by optimizers that don't use them).
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = "match_rms_adamw"


@dataclass(frozen=True)
class OptimizerSpec:
    """How to realize a registered optimizer."""

    name: str
    hf_optim: str | None = None
    builder: Callable[[nn.Module, OptimizerSettings], torch.optim.Optimizer] | None = None

    @property
    def is_custom(self) -> bool:
        return self.builder is not None


OPTIMIZERS: dict[str, OptimizerSpec] = {}


def register_hf_optimizer(name: str, hf_optim: str) -> None:
    """Register an optimizer backed by HF's native `TrainingArguments.optim`."""
    if name in OPTIMIZERS:
        raise ValueError(f"Optimizer {name!r} is already registered.")
    OPTIMIZERS[name] = OptimizerSpec(name=name, hf_optim=hf_optim)


def register_custom_optimizer(name: str):
    """Decorator registering a custom optimizer builder under `name`."""

    def _register(
        builder: Callable[[nn.Module, OptimizerSettings], torch.optim.Optimizer],
    ) -> Callable[[nn.Module, OptimizerSettings], torch.optim.Optimizer]:
        if name in OPTIMIZERS:
            raise ValueError(f"Optimizer {name!r} is already registered.")
        OPTIMIZERS[name] = OptimizerSpec(name=name, builder=builder)
        return builder

    return _register


def resolve_optimizer(name: str) -> OptimizerSpec:
    """Return the spec for `name`, or raise listing the registered optimizers."""
    try:
        return OPTIMIZERS[name]
    except KeyError:
        valid = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown optimizer {name!r}. Registered optimizers: {valid}.") from None


def build_optimizer(
    name: str, model: nn.Module, settings: OptimizerSettings
) -> torch.optim.Optimizer:
    """Build a *custom* optimizer instance.

    HF-native optimizers are built by the Trainer (via `TrainingArguments.optim`),
    so calling this for one is a programming error.
    """
    spec = resolve_optimizer(name)
    if spec.builder is None:
        raise ValueError(
            f"Optimizer {name!r} is HF-native (optim={spec.hf_optim!r}); set it on "
            f"TrainingArguments.optim instead of building it directly."
        )
    return spec.builder(model, settings)


def available_optimizers() -> list[str]:
    return sorted(OPTIMIZERS)


# ----------------------------------------------------------------------
# Built-in registrations
# ----------------------------------------------------------------------

register_hf_optimizer("adamw", "adamw_torch")  # AdamW (PyTorch)
register_hf_optimizer("adamw_fused", "adamw_torch_fused")  # fused AdamW (CUDA)
register_hf_optimizer("adafactor", "adafactor")  # memory-efficient


@register_custom_optimizer("muon")  # Muon for 2D hidden weights + AdamW for the rest
def _build_muon(model: nn.Module, settings: OptimizerSettings) -> torch.optim.Optimizer:
    return build_muon_optimizer(
        model,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        adam_beta1=settings.adam_beta1,
        adam_beta2=settings.adam_beta2,
        adam_eps=settings.adam_eps,
        muon_momentum=settings.muon_momentum,
        muon_nesterov=settings.muon_nesterov,
        muon_ns_steps=settings.muon_ns_steps,
        muon_adjust_lr_fn=settings.muon_adjust_lr_fn,
    )

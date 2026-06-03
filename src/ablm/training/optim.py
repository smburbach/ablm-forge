"""Optimizers.

A small registry maps a config name to either an HF-native
`TrainingArguments.optim` string or a custom builder — no `Trainer` subclass.
HF-native optimizers are selected via `optim`; custom ones (Muon) are built here
and handed to the stock Trainer through its `optimizers=` tuple. Add an optimizer
by adding an entry to `OPTIMIZERS`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

__all__ = [
    "OPTIMIZERS",
    "CombinedOptimizer",
    "OptimizerSettings",
    "OptimizerSpec",
    "available_optimizers",
    "build_muon_optimizer",
    "build_optimizer",
    "resolve_optimizer",
]


@dataclass(frozen=True)
class OptimizerSettings:
    """Common optimizer hyperparameters threaded from the training config.

    Mirrors the relevant `transformers.TrainingArguments` fields so a custom
    optimizer is configured from the same knobs as the HF-native ones.
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


# ----------------------------------------------------------------------
# Muon (2D hidden weights) + AdamW (everything else)
# ----------------------------------------------------------------------

# 2D weights that are *not* hidden linear layers and so stay on AdamW.
_MUON_EXCLUDE = ("embed", "lm_head", "decoder", "classifier")


class CombinedOptimizer(torch.optim.Optimizer):
    """Facade over child optimizers that behaves as one `torch.optim.Optimizer`.

    `param_groups` concatenates the children's groups *by reference*, so an LR
    scheduler writing `group["lr"]` mutates the real child groups. `step` /
    `zero_grad` fan out to every child; `state_dict` / `load_state_dict`
    serialize them as a list. Lets the stock Trainer treat Muon+AdamW as one
    optimizer.
    """

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CombinedOptimizer requires at least one child optimizer.")
        # Deliberately do NOT call super().__init__: the children own the param
        # groups and per-parameter state. Expose what LRScheduler / accelerate read.
        self.defaults: dict[str, Any] = {}
        self._optimizer_step_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_step_post_hooks: OrderedDict = OrderedDict()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(  # ty: ignore[invalid-method-override]  # facade matches Optimizer.step at runtime
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = closure() if closure is not None else None
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        children = state_dict["optimizers"]
        if len(children) != len(self.optimizers):
            raise ValueError("CombinedOptimizer checkpoint has a mismatched optimizer count.")
        for opt, child in zip(self.optimizers, children, strict=True):
            opt.load_state_dict(child)


def build_muon_optimizer(model: nn.Module, settings: OptimizerSettings) -> CombinedOptimizer:
    """Build Muon (2D hidden weights) + AdamW (embeddings, heads, biases, norms).

    Each trainable parameter lands in exactly one group, so coverage is exhaustive
    by construction. Muon's Newton-Schulz step assumes full 2D weights; under
    FSDP2 sharding it is only approximate — validate under single-GPU / DDP first.
    Requires torch >= 2.11 (`torch.optim.Muon`).
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is unavailable; 'muon' requires torch >= 2.11.")

    muon: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "embed" in name:  # biases, norm gains/shifts, embeddings
            no_decay.append(p)
        elif p.ndim == 2 and not any(sub in name for sub in _MUON_EXCLUDE):
            muon.append(p)
        else:  # 2D output projections (lm_head / decoder / classifier), etc.
            decay.append(p)

    if not muon:
        raise ValueError("Muon requires at least one eligible 2D hidden weight.")

    return CombinedOptimizer(
        [
            torch.optim.Muon(
                muon,
                lr=settings.lr,
                weight_decay=settings.weight_decay,
                momentum=settings.muon_momentum,
                nesterov=settings.muon_nesterov,
                ns_steps=settings.muon_ns_steps,
                adjust_lr_fn=settings.muon_adjust_lr_fn,
            ),
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": settings.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=settings.lr,
                betas=(settings.adam_beta1, settings.adam_beta2),
                eps=settings.adam_eps,
            ),
        ]
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerSpec:
    """How to realize an optimizer: an HF `optim` string or a custom builder."""

    hf_optim: str | None = None
    builder: Callable[[nn.Module, OptimizerSettings], torch.optim.Optimizer] | None = None

    @property
    def is_custom(self) -> bool:
        return self.builder is not None


# Add an optimizer here: either an HF `TrainingArguments.optim` string, or a
# `builder(model, settings) -> Optimizer` for one HF doesn't ship.
OPTIMIZERS: dict[str, OptimizerSpec] = {
    "adamw": OptimizerSpec(hf_optim="adamw_torch"),
    "adamw_fused": OptimizerSpec(hf_optim="adamw_torch_fused"),
    "adafactor": OptimizerSpec(hf_optim="adafactor"),
    "muon": OptimizerSpec(builder=build_muon_optimizer),
}


def available_optimizers() -> list[str]:
    return sorted(OPTIMIZERS)


def resolve_optimizer(name: str) -> OptimizerSpec:
    """Return the spec for `name`, or raise listing the registered optimizers."""
    try:
        return OPTIMIZERS[name]
    except KeyError:
        valid = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown optimizer {name!r}. Registered: {valid}.") from None


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

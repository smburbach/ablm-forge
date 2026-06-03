"""Muon optimizer wiring: a combined Muon (2D hidden) + AdamW (everything else).

`torch.optim.Muon` is a pure-Muon optimizer for 2D hidden weights; the documented
recipe puts every other parameter (embeddings, the LM-head / classifier
projections, biases, norms) on AdamW and steps both. `CombinedOptimizer` packages
that pair behind a single `torch.optim.Optimizer`-compatible facade so the stock
`transformers.Trainer` (and its LR scheduler, gradient clipping, and checkpoint
save/load) can treat it as one optimizer.

Build one with :func:`build_muon_optimizer`, which performs the name-aware
2D-hidden vs. rest partition via :mod:`ablm.training.grouping`.

Note on FSDP: Muon's Newton-Schulz orthogonalization operates on full 2D weight
matrices. Under FSDP2 each rank holds a sharded parameter, so the orthogonalization
is not mathematically correct without an all-gather. Validate Muon under
single-GPU / DDP first; treat FSDP + Muon as experimental.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from .grouping import partition_parameters

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

__all__ = ["CombinedOptimizer", "build_muon_optimizer"]


class CombinedOptimizer(torch.optim.Optimizer):
    """A facade over several child optimizers that acts as a single optimizer.

    `param_groups` is the concatenation of the children's groups *by reference*,
    so an LR scheduler that writes `group["lr"]` mutates the real child groups.
    `step` / `zero_grad` fan out to every child; `state_dict` / `load_state_dict`
    serialize them as an ordered list.
    """

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers: list[torch.optim.Optimizer] = list(optimizers)
        if not self.optimizers:
            raise ValueError("CombinedOptimizer requires at least one child optimizer.")
        # Deliberately do NOT call super().__init__: the children own the param
        # groups and per-parameter state. We expose the attributes torch's
        # LRScheduler / accelerate touch directly.
        self.defaults: dict[str, Any] = {}
        self._optimizer_step_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_step_post_hooks: OrderedDict = OrderedDict()

    # -- the views torch/accelerate read -------------------------------------

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, _value: Any) -> None:  # pragma: no cover - guard
        raise AttributeError(
            "CombinedOptimizer.param_groups is derived from its child optimizers "
            "and cannot be assigned directly."
        )

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    # -- stepping ------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(  # ty: ignore[invalid-method-override]  # facade matches Optimizer.step at runtime
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError(
            "CombinedOptimizer does not support add_param_group; configure the "
            "child optimizers at construction time."
        )

    # -- checkpointing -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        child_states = state_dict["optimizers"]
        if len(child_states) != len(self.optimizers):
            raise ValueError(
                f"Checkpoint has {len(child_states)} child optimizer states but this "
                f"CombinedOptimizer has {len(self.optimizers)}."
            )
        for opt, child_state in zip(self.optimizers, child_states, strict=True):
            opt.load_state_dict(child_state)


def build_muon_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: str | None = "match_rms_adamw",
) -> CombinedOptimizer:
    """Build a `CombinedOptimizer` of `torch.optim.Muon` + `torch.optim.AdamW`.

    2D hidden weights go to Muon; embeddings, the LM-head / classifier
    projections, biases, and norm parameters go to AdamW (with the standard
    decay / no-decay split). Requires torch >= 2.11 (`torch.optim.Muon`).
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is unavailable; the 'muon' optimizer requires torch >= 2.11."
        )

    groups = partition_parameters(model, use_muon=True)

    muon = torch.optim.Muon(
        groups.muon_params(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        adjust_lr_fn=muon_adjust_lr_fn,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": groups.adamw_decay_params(), "weight_decay": weight_decay},
            {"params": groups.adamw_no_decay_params(), "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    return CombinedOptimizer([muon, adamw])

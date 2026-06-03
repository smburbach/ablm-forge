"""Parameter-partitioning helpers shared by custom optimizers.

HuggingFace's stock `Trainer` already performs the standard decay / no-decay
split for its built-in optimizers, so these helpers exist only for the custom
optimizers (e.g. Muon) that need a *name-aware* partition: 2D hidden weights go
to Muon, while embeddings, the LM-head / classifier projections, biases, and
norm parameters go to AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = [
    "ParamGroups",
    "is_no_decay_param",
    "is_muon_eligible",
    "partition_parameters",
]

# 2D weights whose *semantics* are not "hidden linear layers" and so must stay on
# AdamW even though they are 2D: token embeddings and the output projections.
_MUON_EXCLUDE_SUBSTRINGS = ("embed", "lm_head", "decoder", "classifier")


def is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    """Whether a parameter should use the no-weight-decay AdamW group.

    Biases and norm gains/shifts (`ndim <= 1`) and embeddings never get weight
    decay.
    """
    return param.ndim <= 1 or "embed" in name


def is_muon_eligible(name: str, param: torch.Tensor) -> bool:
    """Whether a parameter is a 2D hidden weight that Muon should optimize."""
    if param.ndim != 2:
        return False
    return not any(sub in name for sub in _MUON_EXCLUDE_SUBSTRINGS)


@dataclass
class ParamGroups:
    """A name-aware partition of a model's trainable parameters."""

    muon: list[tuple[str, torch.nn.Parameter]] = field(default_factory=list)
    adamw_decay: list[tuple[str, torch.nn.Parameter]] = field(default_factory=list)
    adamw_no_decay: list[tuple[str, torch.nn.Parameter]] = field(default_factory=list)

    def muon_params(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self.muon]

    def adamw_decay_params(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self.adamw_decay]

    def adamw_no_decay_params(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self.adamw_no_decay]


def partition_parameters(model: nn.Module, *, use_muon: bool) -> ParamGroups:
    """Partition `model`'s trainable parameters into Muon and AdamW groups.

    When `use_muon` is False the `muon` group is empty and every decay-eligible
    parameter lands in `adamw_decay` (the plain AdamW grouping). When True, 2D
    hidden weights move to `muon`.

    Raises:
        ValueError: if `use_muon` is True but no parameter is Muon-eligible.
        RuntimeError: if the partition duplicates or misses a trainable param.
    """
    groups = ParamGroups()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_param(name, param):
            groups.adamw_no_decay.append((name, param))
        elif use_muon and is_muon_eligible(name, param):
            groups.muon.append((name, param))
        else:
            groups.adamw_decay.append((name, param))

    all_params = (
        *groups.muon_params(),
        *groups.adamw_decay_params(),
        *groups.adamw_no_decay_params(),
    )
    grouped = [id(p) for p in all_params]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    if len(grouped) != len(set(grouped)):
        raise RuntimeError("Optimizer parameter partition duplicated one or more parameters.")
    if set(grouped) != set(trainable):
        raise RuntimeError("Optimizer parameter partition did not cover all trainable parameters.")
    if use_muon and not groups.muon:
        raise ValueError("Muon requires at least one eligible 2D hidden weight.")
    return groups

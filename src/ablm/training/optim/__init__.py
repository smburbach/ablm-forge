"""Optimizer construction: Muon on the 2D body weights, AdamW on everything else.

HF-native optimizers are selected directly via `TrainingArguments.optim`; Muon is
the one HF does not ship, so it is the only optimizer forge builds itself.
`build_muon_optimizer` splits the model by name and wraps `DistributedMuon` and
`torch.optim.AdamW` in a single `CombinedOptimizer`, which the training script hands
to the stock `Trainer` via `optimizers=(opt, None)` — no `Trainer` subclass.

Import from this package, not from its modules: `muon.py` (construction + param
splitting) and `distributed_muon.py` (DDP-sharded Newton-Schulz) are an internal
split and may be rearranged.
"""

from __future__ import annotations

from .distributed_muon import DistributedMuon
from .muon import (
    MUON_OPTIM,
    MUON_PARAM_PREFIX,
    CombinedOptimizer,
    build_muon_optimizer,
    split_muon_params,
)

__all__ = [
    "MUON_OPTIM",
    "MUON_PARAM_PREFIX",
    "CombinedOptimizer",
    "DistributedMuon",
    "build_muon_optimizer",
    "split_muon_params",
]

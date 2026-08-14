"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer`. Components compose as constructor args
and callbacks: HF-native optimizers via `TrainingArguments.optim`, the collator via
`data_collator=`, metrics via `compute_metrics=`.

Everything here is a *named concern* that maps to a `Trainer` wiring point — `optim`
to `optimizers=`, `data` to `data_collator=` / `compute_metrics=`. Nothing lands
loose in `training/` without a name; a new concern gets its own subpackage rather
than a bare module, so this package never becomes the place training-adjacent code
accumulates.

- `optim` — `build_muon_optimizer` builds Muon (the one optimizer HF doesn't ship) as
  `DistributedMuon` on the 2D body weights + AdamW on the rest, wrapped in a
  `CombinedOptimizer`; hand it to the stock Trainer via `optimizers=(opt, None)`. No
  `Trainer` subclass is needed for the optimizer.
- `data` — region-weighted CDR masking (`PreferentialMaskingCollator`) plus the
  per-region eval metrics that consume its `region_mask`. `RegionEvalMixin` is a
  composable Trainer *mixin*, mixed in only when you need per-region evaluation.
"""

from __future__ import annotations

from .data import (
    PreferentialMaskingCollator,
    RegionEvalMixin,
    add_region_mask,
    compute_metrics,
    pair_mask,
    per_token_ce_and_hits,
)
from .optim import (
    MUON_OPTIM,
    MUON_PARAM_PREFIX,
    CombinedOptimizer,
    DistributedMuon,
    build_muon_optimizer,
    split_muon_params,
)

__all__ = [
    "MUON_OPTIM",
    "MUON_PARAM_PREFIX",
    "CombinedOptimizer",
    "DistributedMuon",
    "PreferentialMaskingCollator",
    "RegionEvalMixin",
    "add_region_mask",
    "build_muon_optimizer",
    "compute_metrics",
    "pair_mask",
    "per_token_ce_and_hits",
    "split_muon_params",
]

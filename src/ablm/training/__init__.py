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
- `data` — region-aware MLM collators (`RegionAwareCollator`, `WeightedMaskingCollator`)
  plus the per-region eval metrics that consume their `region_mask`. `RegionEvalMixin` is a
  composable Trainer *mixin*, mixed in only when you need per-region evaluation.
- `preemption` — Slurm preemption safety (`PreemptionSafeMixin`, `PreemptionSafeTrainer`): JIT
  checkpoint on SIGTERM, object-storage mirror/restore with `s5cmd`, DataLoader-worker shielding.
  A tracked port of coreweave-docs `model-training/single-run/jit/preemption.py`.
"""

from __future__ import annotations

from .data import (
    IGNORE,
    TIERS,
    CountMode,
    MaskingStatsMixin,
    RegionAwareCollator,
    RegionEvalMixin,
    WeightedMaskingCollator,
    add_region_mask,
    compute_metrics,
    masking_stats,
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
from .preemption import PreemptionSafeMixin, PreemptionSafeTrainer

__all__ = [
    "MUON_OPTIM",
    "MUON_PARAM_PREFIX",
    "CombinedOptimizer",
    "DistributedMuon",
    "IGNORE",
    "TIERS",
    "CountMode",
    "MaskingStatsMixin",
    "PreemptionSafeMixin",
    "PreemptionSafeTrainer",
    "RegionAwareCollator",
    "RegionEvalMixin",
    "WeightedMaskingCollator",
    "add_region_mask",
    "build_muon_optimizer",
    "compute_metrics",
    "masking_stats",
    "pair_mask",
    "per_token_ce_and_hits",
    "split_muon_params",
]

"""Region-weighted CDR masking, and the per-region eval metrics that read it back.

The two halves share one contract and are kept together because nothing else
enforces it: the collators write a `region_mask` onto every batch,
and `RegionEvalMixin` / `compute_metrics` are its only consumers. No import binds
them — the contract is the `region_mask` key itself — so colocation is what keeps
the producer and consumer readable as one unit.

`RegionAwareCollator` is the stock HF Bernoulli collator with `region_mask` /
`seq_mutated` surviving dynamic padding; `WeightedMaskingCollator` biases selection
toward CDR / SHM regions via `cdr_ratios` / `nt_ratio` under a `CountMode`.
`add_region_mask` / `pair_mask` build the `region_mask` a training script attaches to
each example before it reaches the collator.

`RegionEvalMixin` is a composable `Trainer` mixin (the one sanctioned `Trainer`
subclass; see AGENTS.md): it strips `region_mask` before `model.forward`, swaps in a
uniform eval collator, and reduces each eval step's logits to per-token CE + hits for
`compute_metrics` to aggregate by region.

Data *loading* is deliberately not here — it is a handful of 🤗 `datasets` calls in
the training script, so each run owns and can edit it.
"""

from __future__ import annotations

from .collators import (
    IGNORE,
    TIERS,
    CountMode,
    RegionAwareCollator,
    WeightedMaskingCollator,
    add_region_mask,
    pair_mask,
)
from .metrics import RegionEvalMixin, compute_metrics, per_token_ce_and_hits

__all__ = [
    "IGNORE",
    "TIERS",
    "CountMode",
    "RegionAwareCollator",
    "RegionEvalMixin",
    "WeightedMaskingCollator",
    "add_region_mask",
    "compute_metrics",
    "pair_mask",
    "per_token_ce_and_hits",
]

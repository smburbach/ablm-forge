"""Region-weighted CDR masking, and the readers that consume its side channels.

The pieces share one contract and are kept together because nothing else enforces
it: the collators write `region_mask` and `seq_mutated` onto every batch, and the
readers below are their only consumers. No import binds them — the contract is the
key names themselves — so colocation is what keeps the producers and consumers
readable as one unit.

Two collators produce it. `RegionAwareCollator` is the stock HF Bernoulli collator
with `region_mask` / `seq_mutated` surviving dynamic padding; it is the production
collator, because exact-count masking costs +0.01447 on held-out donors.
`WeightedMaskingCollator` biases selection toward CDR / SHM regions via `cdr_ratios` /
`nt_ratio` under a `CountMode` (default `CountMode.BERNOULLI`).
`add_region_mask` / `pair_mask` build the `region_mask` a training script attaches to
each example before it reaches the collator.

Two sets of readers consume it. `RegionEvalMixin` / `compute_metrics` are for eval:
`RegionEvalMixin` is a composable `Trainer` mixin (the one sanctioned `Trainer`
subclass; see AGENTS.md) that strips the side channels before `model.forward`, swaps
in a uniform eval collator, and reduces each eval step's logits to per-token CE + hits
for `compute_metrics` to aggregate by region. `MaskingStatsMixin` / `masking_stats`
are for the train batch: they report the realised masking rates of the latest
micro-batch.

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
from .metrics import (
    MaskingStatsMixin,
    RegionEvalMixin,
    compute_metrics,
    masking_stats,
    per_token_ce_and_hits,
)

__all__ = [
    "IGNORE",
    "TIERS",
    "CountMode",
    "MaskingStatsMixin",
    "RegionAwareCollator",
    "RegionEvalMixin",
    "WeightedMaskingCollator",
    "add_region_mask",
    "compute_metrics",
    "masking_stats",
    "pair_mask",
    "per_token_ce_and_hits",
]

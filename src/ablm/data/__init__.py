"""Data-loading building blocks for ABLM: region-weighted CDR masking.

`PreferentialMaskingCollator` masks an exact per-sequence token count (Gumbel-
top-k), biased toward CDR / SHM regions via `cdr_ratios` / `nt_ratio`.
`cdr_ratios=1.0, nt_ratio=1.0` makes it a uniform exact-count sampler — the
eval-collator arm, so eval/loss stays comparable across training arms.
`add_region_mask` / `pair_mask` build the `region_mask` a training script
attaches to each example before it reaches the collator.
"""

from __future__ import annotations

from .collators import PreferentialMaskingCollator, add_region_mask, pair_mask

__all__ = ["PreferentialMaskingCollator", "add_region_mask", "pair_mask"]

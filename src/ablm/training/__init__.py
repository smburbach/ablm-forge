"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer`. Components compose as constructor args
and callbacks: HF-native optimizers via `TrainingArguments.optim`, the collator via
`data_collator=`, metrics via `compute_metrics=`. Muon is the one optimizer HF
doesn't ship — `optim.py`'s `build_muon_optimizer` builds it (`DistributedMuon` on
the 2D body weights + AdamW on the rest, wrapped in a `CombinedOptimizer`); hand it
to the stock Trainer via `optimizers=(opt, None)`. No `Trainer` subclass is needed
for the optimizer.

`metrics.py` provides `RegionEvalMixin`, a composable Trainer *mixin* for per-region
eval CE/accuracy alongside `ablm.data.PreferentialMaskingCollator` — mix it into a
Trainer subclass only when you need per-region evaluation.
"""

from __future__ import annotations

from .metrics import RegionEvalMixin, compute_metrics, per_token_ce_and_hits

__all__ = [
    "RegionEvalMixin",
    "compute_metrics",
    "per_token_ce_and_hits",
]

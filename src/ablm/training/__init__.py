"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer` for every HF-native optimizer. The one
exception is Muon, which HF doesn't ship: `optim.py` provides `OptimizerTrainer`,
a thin `Trainer` subclass overriding only `create_optimizer` to build Muon (+ aux
AdamW, wrapped in a `CombinedOptimizer`). The subclass is required — HF forbids a
pre-built `optimizers=` tuple under FSDP. No training loop is overridden.

`metrics.py` provides `RegionEvalMixin`, a composable Trainer *mixin* (not a
subclass in its own right) for per-region eval CE/accuracy alongside
`ablm.data.PreferentialMaskingCollator`.
"""

from __future__ import annotations

from .metrics import RegionEvalMixin, compute_metrics, per_token_ce_and_hits

__all__ = [
    "RegionEvalMixin",
    "compute_metrics",
    "per_token_ce_and_hits",
]

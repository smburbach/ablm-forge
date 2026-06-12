"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer` for every HF-native optimizer. The one
exception is Muon, which HF doesn't ship: `optim.py` provides `OptimizerTrainer`,
a thin `Trainer` subclass overriding only `create_optimizer` to build Muon (+ aux
AdamW, wrapped in a `CombinedOptimizer`). The subclass is required — HF forbids a
pre-built `optimizers=` tuple under FSDP. No training loop is overridden.
"""

from __future__ import annotations

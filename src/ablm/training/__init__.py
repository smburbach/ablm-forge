"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer` directly (no Trainer subclass). This
package provides only the optimizer HF doesn't ship — Muon (`optim.py`) — built
as a `CombinedOptimizer` and passed to the Trainer via its `optimizers=` tuple.
"""

from __future__ import annotations

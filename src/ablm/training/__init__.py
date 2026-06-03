"""Training infrastructure for ABLM.

ABLM uses the stock `transformers.Trainer` directly (no Trainer subclass). This
package provides only the pieces HF doesn't ship: a pluggable optimizer registry
and the custom optimizers (e.g. Muon) wired in via the Trainer's
`optimizer_cls_and_kwargs` hook.
"""

from __future__ import annotations

"""Top-level import path for the preset registry; canonical impl in `.model.presets`.

Kept separate from `ablm/model/presets.py` (which `AblmConfig.from_preset`'s
lazy import depends on to avoid the presets<->config circular import) so that
both `ablm.presets` and `ablm.model.presets` resolve to the same objects.
"""

from __future__ import annotations

from .model.presets import PRESETS, from_preset

__all__ = ["PRESETS", "from_preset"]

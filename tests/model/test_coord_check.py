# tests/model/test_coord_check.py
"""muP coordinate check: activation RMS is ~width-invariant at init.

This is the CI proof the preset ladder is muP-correct; it does not surface a
transfer workflow. Small widths keep it CPU-cheap.
"""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM

_WIDTHS = (128, 256, 512)  # 4x range
_BASE = 128


def _hidden_rms(hidden: int, *, mup: bool) -> float:
    kw = dict(mup_enabled=True, mup_base_hidden_size=_BASE) if mup else {}
    cfg = AblmConfig(
        hidden_size=hidden, num_attention_heads=hidden // 64, num_hidden_layers=2, **kw
    )
    torch.manual_seed(0)
    model = AblmForMaskedLM(cfg).eval()
    ids = torch.randint(4, 33, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    with torch.no_grad():
        out = model.ablm(input_ids=ids, attention_mask=mask)
    return out.last_hidden_state.float().pow(2).mean().sqrt().item()


@pytest.mark.slow
def test_mup_activation_rms_flat_across_width():
    rms = [_hidden_rms(w, mup=True) for w in _WIDTHS]
    assert min(rms) > 0
    spread = max(rms) / min(rms)
    assert spread < 2.5, (
        f"muP activation RMS not width-stable: {dict(zip(_WIDTHS, rms, strict=True))}"
    )


@pytest.mark.slow
def test_mup_is_flatter_than_fixed_init_baseline():
    mup = [_hidden_rms(w, mup=True) for w in _WIDTHS]
    base = [_hidden_rms(w, mup=False) for w in _WIDTHS]
    assert (max(mup) / min(mup)) <= (max(base) / min(base)) + 1e-6

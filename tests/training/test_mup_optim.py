"""Readout param gets its own muP-scaled AdamW LR group."""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.optim import build_muon_optimizer

pytestmark = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="requires torch.optim.Muon (torch >= 2.11)"
)


def _model(mup: bool) -> AblmForMaskedLM:
    kw = dict(mup_enabled=True, mup_base_hidden_size=256) if mup else {}
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8, num_hidden_layers=2, **kw)
    return AblmForMaskedLM(cfg)


def _group_lrs(opt) -> list[float]:
    return [g["lr"] for g in opt.param_groups]


def test_disabled_has_no_extra_readout_group():
    opt = build_muon_optimizer(_model(mup=False), lr=1e-3)
    # 1 Muon group + 2 AdamW groups (decay / no-decay).
    assert len(opt.param_groups) == 3
    assert all(lr == 1e-3 for lr in _group_lrs(opt))


def test_enabled_scales_readout_group_lr():
    model = _model(mup=True)
    readout = model.lm_head.decoder.weight
    opt = build_muon_optimizer(model, lr=1e-3)
    # base/hidden = 256/512 = 0.5
    readout_groups = [g for g in opt.param_groups if any(p is readout for p in g["params"])]
    assert len(readout_groups) == 1
    assert readout_groups[0]["lr"] == pytest.approx(1e-3 * 0.5)
    # and the readout is NOT in any Muon group (still AdamW).

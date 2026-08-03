"""Fan-in muP init: hidden-Linear std scales as 1/sqrt(fan_in)."""

from __future__ import annotations

import math

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM


def _qkv_weight_std(hidden: int) -> float:
    cfg = AblmConfig(
        hidden_size=hidden,
        num_attention_heads=hidden // 64,
        num_hidden_layers=2,
        mup_enabled=True,
        mup_base_hidden_size=256,
        initializer_range=0.02,
    )
    torch.manual_seed(0)
    model = AblmForMaskedLM(cfg)
    # q_proj: a non-residual-writer hidden Linear with fan_in == hidden.
    q = model.ablm.backbone.layers[0].attention.q_proj.weight
    return q.detach().std().item()


def test_mup_hidden_init_scales_inverse_sqrt_fan_in():
    std_256 = _qkv_weight_std(256)  # fan_in == base -> std ~ initializer_range
    std_1024 = _qkv_weight_std(1024)  # fan_in 4x base -> std ~ / 2
    assert std_256 == pytest.approx(0.02, rel=0.15)
    assert std_1024 == pytest.approx(0.02 * math.sqrt(256 / 1024), rel=0.15)


def test_mup_disabled_init_is_fixed_range():
    cfg = AblmConfig(hidden_size=1024, num_attention_heads=16, num_hidden_layers=2)
    torch.manual_seed(0)
    model = AblmForMaskedLM(cfg)
    q = model.ablm.backbone.layers[0].attention.q_proj.weight
    assert q.detach().std().item() == pytest.approx(0.02, rel=0.15)

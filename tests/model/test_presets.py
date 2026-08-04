"""Preset ladder: kernel rules, muP derivation, from_preset round-trip."""

from __future__ import annotations

import pytest

from ablm import PRESETS, AblmConfig, from_preset

_EXPECTED = {
    "35m": (16, 512),
    "150m": (24, 768),
    "300m": (30, 1024),
    "600m": (36, 1152),
}


def test_registry_names():
    assert set(PRESETS) == set(_EXPECTED)


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_preset_obeys_kernel_rules(name):
    layers, hidden = _EXPECTED[name]
    cfg = from_preset(name)
    assert cfg.num_hidden_layers == layers
    assert cfg.hidden_size == hidden
    assert cfg.head_dim == 64
    assert hidden % 128 == 0
    assert cfg.num_attention_heads == hidden // 64
    assert cfg.intermediate_size % 128 == 0
    assert cfg.vocab_size == 33


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_preset_is_mup_correct(name):
    _, hidden = _EXPECTED[name]
    cfg = from_preset(name)
    assert cfg.mup_enabled is True
    assert cfg.mup_base_hidden_size == 512
    assert cfg.mup_output_mult == pytest.approx(512 / hidden)
    assert cfg.tie_word_embeddings is False


def test_from_preset_ffn_is_8_over_3_rounded_256():
    # 300m: round_up_to(int(8*1024/3), 256) == 2816
    assert from_preset("300m").intermediate_size == 2816


def test_from_preset_overrides_pass_through():
    cfg = from_preset("35m", num_hidden_layers=4)
    assert cfg.num_hidden_layers == 4
    assert cfg.hidden_size == 512  # untouched


def test_config_classmethod_matches_module_fn():
    a = AblmConfig.from_preset("150m")
    b = from_preset("150m")
    assert a.to_dict() == b.to_dict()

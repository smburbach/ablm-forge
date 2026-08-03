"""muP field derivation and validation on AblmConfig."""

from __future__ import annotations

import pytest

from ablm import AblmConfig


def test_mup_disabled_by_default_multipliers_are_identity():
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8)
    assert cfg.mup_enabled is False
    assert cfg.mup_output_mult == 1.0
    assert cfg.mup_readout_lr_mult == 1.0
    assert cfg.mup_emb_mult == 1.0


def test_mup_multipliers_derive_from_base_over_width():
    cfg = AblmConfig(
        hidden_size=1024,
        num_attention_heads=16,
        mup_enabled=True,
        mup_base_hidden_size=512,
    )
    assert cfg.mup_output_mult == pytest.approx(512 / 1024)
    assert cfg.mup_readout_lr_mult == pytest.approx(512 / 1024)


def test_mup_enabled_requires_base_hidden_size():
    with pytest.raises(ValueError, match="mup_base_hidden_size"):
        AblmConfig(hidden_size=512, num_attention_heads=8, mup_enabled=True)


def test_mup_enabled_rejects_tied_embeddings():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        AblmConfig(
            hidden_size=512,
            num_attention_heads=8,
            mup_enabled=True,
            mup_base_hidden_size=512,
            tie_word_embeddings=True,
        )


def test_mup_fields_survive_to_dict_round_trip():
    cfg = AblmConfig(
        hidden_size=768,
        num_attention_heads=12,
        mup_enabled=True,
        mup_base_hidden_size=512,
        mup_emb_mult=2.0,
    )
    restored = AblmConfig.from_dict(cfg.to_dict())
    assert restored.mup_enabled is True
    assert restored.mup_base_hidden_size == 512
    assert restored.mup_emb_mult == 2.0
    assert restored.mup_output_mult == pytest.approx(512 / 768)

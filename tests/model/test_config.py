"""Tests for `ablm.model.configuration_ablm` — AblmConfig validation and derivation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ablm.model import AblmConfig
from ablm.model.ffn import round_up_to

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults and derived fields
# ---------------------------------------------------------------------------


def test_defaults_match_architecture_spec():
    cfg = AblmConfig()
    assert cfg.model_type == "ablm"
    assert cfg.vocab_size == 33
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert cfg.max_position_embeddings == 1024
    assert cfg.rope_theta == 10000.0
    # Defaults track ESM-C: Pre-LN, bias-free, SwiGLU, no QK-norm / residual
    # scaling, ESM-style token dropout.
    assert cfg.norm_type == "layernorm"
    assert cfg.norm_bias is False
    assert cfg.norm_strategy == "pre"
    assert cfg.qk_norm is False
    assert cfg.residual_scaling == "none"
    assert cfg.init_scale_output_projections is True
    assert cfg.ffn_activation == "swiglu"
    assert cfg.ffn_bias is False
    assert cfg.token_dropout is False  # ESM-C removed token dropout
    assert cfg.attention_dropout == 0.0
    assert cfg.hidden_dropout == 0.0
    assert cfg.tie_word_embeddings is False
    assert cfg.mlm_head_activation == "gelu"
    assert cfg.classifier_pool == "mean"
    assert cfg.classifier_dropout == 0.0
    assert cfg.pre_head_norm is False
    assert cfg.gradient_checkpointing is False
    assert cfg.pad_token_id == 1
    assert cfg.bos_token_id == 0
    assert cfg.eos_token_id == 2
    assert cfg.unk_token_id == 3
    assert cfg.mask_token_id == 32


def test_head_dim_derived_from_hidden_and_heads():
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8)
    assert cfg.head_dim == 64


def test_intermediate_size_derived_from_swiglu_convention():
    cfg = AblmConfig(hidden_size=768)
    # 8/3 * 768 = 2048; already a multiple of 256.
    assert cfg.intermediate_size == round_up_to(int(8 * 768 / 3), 256)
    assert cfg.intermediate_size == 2048


def test_intermediate_size_rounds_up_to_256():
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8)
    # 8/3 * 512 ≈ 1365 -> round up to nearest 256 -> 1536.
    assert cfg.intermediate_size == 1536


def test_rope_dim_defaults_to_head_dim_with_zero_nope():
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8)
    assert cfg.rope_dim == cfg.head_dim
    assert cfg.nope_dim == 0


def test_explicit_overrides_take_precedence_over_derivations():
    cfg = AblmConfig(
        hidden_size=512,
        num_attention_heads=8,
        head_dim=64,
        intermediate_size=2048,
        rope_dim=32,
        nope_dim=32,
    )
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 2048
    assert cfg.rope_dim == 32
    assert cfg.nope_dim == 32


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def test_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        AblmConfig(hidden_size=100, num_attention_heads=8)


def test_rejects_head_dim_mismatch():
    with pytest.raises(ValueError, match="must equal hidden_size"):
        AblmConfig(hidden_size=512, num_attention_heads=8, head_dim=128)


def test_rejects_rope_plus_nope_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        AblmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=32,
            nope_dim=16,
        )


def test_rejects_odd_rope_dim():
    with pytest.raises(ValueError, match="rope_dim must be even"):
        AblmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=33,
            nope_dim=31,
        )


def test_rejects_negative_rope_dim():
    with pytest.raises(ValueError, match="rope_dim must be >= 0"):
        AblmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=-2,
            nope_dim=66,
        )


@pytest.mark.parametrize(
    "field,bad_value,expected_match",
    [
        ("norm_type", "zorm", "norm_type must be one of"),
        ("norm_strategy", "preprost", "norm_strategy must be one of"),
        ("residual_scaling", "linear", "residual_scaling must be one of"),
        ("ffn_activation", "relu", "ffn_activation must be one of"),
        ("mlm_head_activation", "swiglu", "mlm_head_activation must be one of"),
        ("classifier_pool", "max", "classifier_pool must be one of"),
    ],
)
def test_rejects_unknown_categorical_values(field, bad_value, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        AblmConfig(**{field: bad_value})


def test_non_default_vocab_emits_warning():
    with pytest.warns(UserWarning, match="custom vocabularies are not yet supported"):
        AblmConfig(vocab_size=64)


# ---------------------------------------------------------------------------
# Pass-through / forward-compat with PretrainedConfig kwargs
# ---------------------------------------------------------------------------


def test_pretrained_config_kwargs_forwarded():
    cfg = AblmConfig(
        architectures=["AblmForMaskedLM"],
        _name_or_path="smburbach/ablm-forge",
    )
    # Both attrs come from PretrainedConfig.__init__'s kwargs handling.
    assert cfg.architectures == ["AblmForMaskedLM"]
    assert cfg._name_or_path == "smburbach/ablm-forge"


def test_tie_word_embeddings_forwarded_to_base_class():
    cfg = AblmConfig(tie_word_embeddings=True)
    assert cfg.tie_word_embeddings is True


# ---------------------------------------------------------------------------
# save_pretrained / from_pretrained round-trip
# ---------------------------------------------------------------------------


def test_save_and_from_pretrained_roundtrip_preserves_fields(tmp_path: Path):
    cfg = AblmConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        norm_type="rmsnorm",
        norm_strategy="hybrid",
        post_embed_norm=True,
        tie_word_embeddings=True,
    )
    cfg.save_pretrained(tmp_path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["model_type"] == "ablm"

    restored = AblmConfig.from_pretrained(tmp_path)
    assert restored.hidden_size == 512
    assert restored.num_hidden_layers == 4
    assert restored.num_attention_heads == 8
    assert restored.head_dim == 64
    assert restored.norm_type == "rmsnorm"
    assert restored.norm_strategy == "hybrid"
    assert restored.post_embed_norm is True
    assert restored.tie_word_embeddings is True


def test_save_pretrained_writes_token_ids(tmp_path: Path):
    cfg = AblmConfig()
    cfg.save_pretrained(tmp_path)
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["pad_token_id"] == 1
    assert on_disk["bos_token_id"] == 0
    assert on_disk["eos_token_id"] == 2

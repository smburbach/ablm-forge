"""Parity of the stock-ESM2-350M AblmConfig against transformers' EsmForMaskedLM.

This is the architecture every ablm-sweeps `esm2/14_rerun_v2026-09-17` stage trains
(32L / 960 / 20 heads, GELU FFN 3840, rotary, biases on, tied embeddings, token dropout).
transformers' model has 355,252,514 parameters; forge omits the vestigial contact head
(`Linear(32 * 20 -> 1)` = 641 params, never trained under MLM), so the match is exact minus 641.
ESM initialises with `normal_(0.02)`; forge truncates at ±2σ, which realises 0.879626σ, so the
match uses `initializer_range = 0.02 / 0.879626` and no residual-writer scaling.
"""

from __future__ import annotations

import torch

from ablm import AblmConfig, AblmForMaskedLM

ESM2_350M_PARAMS = 355_252_514
ESM_CONTACT_HEAD_PARAMS = 32 * 20 + 1
EXPECTED_PARAMS = ESM2_350M_PARAMS - ESM_CONTACT_HEAD_PARAMS  # 355,251,873
ESM_INIT_STD = 0.02
TRUNC_NORMAL_2SIGMA_FACTOR = 0.879626


def esm2_350m_config(**overrides: object) -> AblmConfig:
    """The rerun-01 stock ESM2 350M recipe expressed as an AblmConfig."""
    fields: dict[str, object] = dict(
        vocab_size=33,
        pad_token_id=1,
        mask_token_id=32,
        hidden_size=960,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=3840,
        max_position_embeddings=322,
        norm_type="layernorm",
        norm_eps=1e-12,
        norm_bias=True,
        norm_strategy="pre",
        qk_norm=False,
        post_embed_norm=False,
        residual_scaling="none",
        ffn_activation="gelu",
        ffn_bias=True,
        attention_bias=True,
        token_dropout=True,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        tie_word_embeddings=True,
        mlm_head_activation="gelu",
        initializer_range=ESM_INIT_STD / TRUNC_NORMAL_2SIGMA_FACTOR,
        init_scale_output_projections=False,
    )
    fields.update(overrides)
    return AblmConfig(**fields)  # ty: ignore[invalid-argument-type]  # dict[str, object] unpack into typed kwargs


def test_head_dim_is_48():
    assert esm2_350m_config().head_dim == 48


def test_param_count_matches_esm2_350m_minus_contact_head():
    model = AblmForMaskedLM(esm2_350m_config())
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_config_round_trips_through_to_dict():
    cfg = esm2_350m_config()
    assert AblmConfig(**cfg.to_dict()).to_dict() == cfg.to_dict()


def test_realised_init_std_matches_esm():
    torch.manual_seed(0)
    model = AblmForMaskedLM(esm2_350m_config(num_hidden_layers=2))
    block = model.ablm.backbone.layers[0]
    for weight in (
        block.attention.q_proj.weight,
        block.attention.o_proj.weight,
        block.ffn.down_proj.weight,
        model.get_input_embeddings().weight,
    ):
        assert abs(weight.std().item() - ESM_INIT_STD) < 0.0005, weight.shape


def test_forward_produces_finite_logits():
    model = AblmForMaskedLM(esm2_350m_config(num_hidden_layers=2)).eval()
    ids = torch.randint(4, 30, (2, 16))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.shape == (2, 16, 33)
    assert torch.isfinite(logits).all()

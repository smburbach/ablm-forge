"""Verify the default architecture tracks ESM-C, plus token-dropout behavior.

ESM-C (EvolutionaryScale Cambrian) is: Pre-LN, full RoPE, SwiGLU, bias-free
linear layers and layer norms, no QK-norm, no residual scaling, and crucially
**no token dropout** (ESM-2 had it; ESM-C removed it). ABLM's defaults are set to
match; these tests pin that alignment. Token dropout remains available as an
opt-in knob and is exercised below by passing `token_dropout=True` explicitly.
"""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.config import load_config
from ablm.model.embedding import _TOKEN_DROPOUT_MASK_RATIO_TRAIN, AblmEmbedding
from ablm.model.norm import AblmLayerNorm


def _tiny(**kw) -> AblmConfig:
    return AblmConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        **kw,
    )


# ---------------------------------------------------------------------------
# ESM-C default architecture
# ---------------------------------------------------------------------------


def test_default_architecture_matches_esmc():
    cfg = AblmConfig()
    assert cfg.norm_strategy == "pre"  # Pre-LN
    assert cfg.norm_type == "layernorm"
    assert cfg.norm_bias is False  # bias-free layer norms
    assert cfg.ffn_activation == "swiglu"
    assert cfg.ffn_bias is False  # bias-free FFN
    assert cfg.qk_norm is False  # ESM-C has no QK-norm
    assert cfg.residual_scaling == "none"  # ESM-C has no residual scaling
    assert cfg.token_dropout is False  # ESM-C removed ESM-2's token dropout
    # Full RoPE on every head channel.
    assert cfg.rope_dim == cfg.head_dim
    assert cfg.nope_dim == 0


def test_norm_bias_false_makes_all_layernorms_bias_free():
    model = AblmForMaskedLM(_tiny(norm_type="layernorm", norm_bias=False))
    layernorms = [m for m in model.modules() if isinstance(m, AblmLayerNorm)]
    assert layernorms, "expected at least one AblmLayerNorm"
    assert all(ln.bias is None for ln in layernorms)


def test_norm_bias_true_restores_bias():
    model = AblmForMaskedLM(_tiny(norm_type="layernorm", norm_bias=True))
    layernorms = [m for m in model.modules() if isinstance(m, AblmLayerNorm)]
    assert all(ln.bias is not None for ln in layernorms)


@pytest.mark.parametrize("preset", ["esmc_300m", "esmc_600m", "esmc_6b"])
def test_esmc_presets_have_exact_dims_and_head_dim_64(preset):
    cfg = load_config(["--preset", preset]).model
    assert cfg.head_dim == 64
    assert cfg.max_position_embeddings == 2048
    assert cfg.qk_norm is False and cfg.residual_scaling == "none"
    expected = {
        "esmc_300m": (30, 960, 15),
        "esmc_600m": (36, 1152, 18),
        "esmc_6b": (80, 2560, 40),
    }[preset]
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads) == expected


# ---------------------------------------------------------------------------
# Token dropout (ESM-2 / ESM-C)
# ---------------------------------------------------------------------------


def test_token_dropout_zeros_mask_rows_then_rescales():
    cfg = _tiny(token_dropout=True)
    emb = AblmEmbedding(cfg).eval()
    ids = torch.randint(4, 30, (1, 8))
    ids[0, 3] = cfg.mask_token_id
    mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = emb(ids, mask)
    # The masked position is zeroed (then scaled, so still zero).
    assert torch.allclose(out[0, 3], torch.zeros(cfg.hidden_size))
    # Non-masked positions are non-zero.
    assert out[0, 0].abs().sum() > 0


def test_token_dropout_inference_scale_without_masks():
    cfg = _tiny(token_dropout=True)
    emb = AblmEmbedding(cfg).eval()
    ids = torch.randint(4, 30, (1, 8))  # no mask tokens
    mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        raw = emb.embed_tokens(ids)
        out = emb(ids, mask)
    expected_scale = 1.0 - _TOKEN_DROPOUT_MASK_RATIO_TRAIN  # 0.88
    assert torch.allclose(out, raw * expected_scale, atol=1e-6)


def test_token_dropout_off_is_plain_lookup():
    cfg = _tiny(token_dropout=False)
    emb = AblmEmbedding(cfg).eval()
    ids = torch.randint(4, 30, (1, 8))
    ids[0, 2] = cfg.mask_token_id
    with torch.no_grad():
        out = emb(ids, torch.ones(1, 8, dtype=torch.long))
        raw = emb.embed_tokens(ids)
    assert torch.allclose(out, raw)


def test_token_dropout_model_forward_trains():
    """A full model with token_dropout on runs forward+backward with finite loss."""
    model = AblmForMaskedLM(_tiny(token_dropout=True)).train()
    ids = torch.randint(4, 30, (2, 16))
    ids[0, 5] = model.config.mask_token_id
    labels = torch.randint(0, 33, (2, 16))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None

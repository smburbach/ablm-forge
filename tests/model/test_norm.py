"""Tests for `ablm.model.norm` — AblmLayerNorm, AblmRMSNorm, make_norm."""

from __future__ import annotations

import pytest
import torch

from ablm.model.norm import AblmLayerNorm, AblmRMSNorm, make_norm

# ---------------------------------------------------------------------------
# AblmLayerNorm
# ---------------------------------------------------------------------------


def test_layernorm_output_shape_matches_input():
    norm = AblmLayerNorm(8)
    x = torch.randn(2, 4, 8)
    out = norm(x)
    assert out.shape == x.shape


def test_layernorm_zero_mean_unit_var_with_default_affine():
    norm = AblmLayerNorm(64)
    x = torch.randn(3, 5, 64) * 7.0 + 3.0
    out = norm(x)
    assert torch.allclose(out.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_layernorm_weight_and_bias_applied():
    norm = AblmLayerNorm(4)
    with torch.no_grad():
        norm.weight.fill_(2.0)
        norm.bias.fill_(5.0)
    x = torch.randn(2, 4)
    out = norm(x)
    # After normalization, mean should be 5.0 and std should be 2.0 per row.
    assert torch.allclose(out.mean(-1), torch.full((2,), 5.0), atol=1e-4)
    assert torch.allclose(out.std(-1, unbiased=False), torch.full((2,), 2.0), atol=1e-4)


def test_layernorm_no_bias_param_when_disabled():
    norm = AblmLayerNorm(8, bias=False)
    assert norm.bias is None
    x = torch.randn(2, 8)
    out = norm(x)
    # Without bias, per-row mean is zero.
    assert torch.allclose(out.mean(-1), torch.zeros(2), atol=1e-5)


def test_layernorm_preserves_input_dtype_bf16():
    norm = AblmLayerNorm(16)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16


def test_layernorm_preserves_input_dtype_fp16():
    norm = AblmLayerNorm(16)
    x = torch.randn(2, 16, dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16


def test_layernorm_internal_compute_is_fp32():
    """A pathological input that would overflow fp16 variance must succeed."""
    norm = AblmLayerNorm(8)
    # Values too large to square in fp16 (max ~65504, 300**2 = 90000 overflows).
    x = torch.full((1, 8), 300.0, dtype=torch.float16)
    x[0, 0] = -300.0  # break the degenerate constant case
    out = norm(x)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float16


def test_layernorm_matches_torch_layernorm():
    """Numerically equivalent to torch.nn.LayerNorm in fp32."""
    torch_ln = torch.nn.LayerNorm(32, eps=1e-6)
    ablm_ln = AblmLayerNorm(32, eps=1e-6)
    with torch.no_grad():
        ablm_ln.weight.copy_(torch_ln.weight)
        ablm_ln.bias.copy_(torch_ln.bias)
    x = torch.randn(4, 7, 32)
    assert torch.allclose(ablm_ln(x), torch_ln(x), atol=1e-5)


# ---------------------------------------------------------------------------
# AblmRMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_output_shape_matches_input():
    norm = AblmRMSNorm(8)
    x = torch.randn(2, 4, 8)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_rms_with_default_weight():
    norm = AblmRMSNorm(128)
    x = torch.randn(3, 5, 128) * 4.0
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(3, 5), atol=1e-4)


def test_rmsnorm_no_bias():
    norm = AblmRMSNorm(8)
    assert not hasattr(norm, "bias") or getattr(norm, "bias", None) is None


def test_rmsnorm_weight_applied():
    norm = AblmRMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(3.0)
    x = torch.randn(2, 16)
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full((2,), 3.0), atol=1e-4)


def test_rmsnorm_preserves_input_dtype_bf16():
    norm = AblmRMSNorm(16)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16


def test_rmsnorm_internal_compute_is_fp32():
    """Squaring large fp16 values would overflow without fp32 internals."""
    norm = AblmRMSNorm(8)
    x = torch.full((1, 8), 300.0, dtype=torch.float16)
    out = norm(x)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# make_norm factory
# ---------------------------------------------------------------------------


def test_make_norm_layernorm():
    norm = make_norm("layernorm", 32)
    assert isinstance(norm, AblmLayerNorm)


def test_make_norm_rmsnorm():
    norm = make_norm("rmsnorm", 32)
    assert isinstance(norm, AblmRMSNorm)


def test_make_norm_forwards_eps():
    norm = make_norm("layernorm", 16, eps=1e-3)
    assert norm.eps == 1e-3
    norm = make_norm("rmsnorm", 16, eps=1e-3)
    assert norm.eps == 1e-3


def test_make_norm_raises_on_unknown_type():
    with pytest.raises(ValueError, match="Unknown norm_type"):
        make_norm("groupnorm", 16)


def test_make_norm_accepts_tuple_shape():
    norm = make_norm("layernorm", (8,))
    x = torch.randn(2, 4, 8)
    out = norm(x)
    assert out.shape == x.shape

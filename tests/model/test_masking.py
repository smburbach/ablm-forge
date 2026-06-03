"""Tests for `ablm.model.masking` — prepare_attention_mask."""

from __future__ import annotations

import pytest
import torch

from ablm.model.masking import prepare_attention_mask

# ---------------------------------------------------------------------------
# prepare_attention_mask
# ---------------------------------------------------------------------------


def test_prepare_attention_mask_defaults_to_all_ones():
    mask = prepare_attention_mask(None, batch_size=2, seq_len=5, device="cpu")
    assert mask.shape == (2, 5)
    assert mask.dtype == torch.long
    assert mask.device.type == "cpu"
    assert torch.equal(mask, torch.ones(2, 5, dtype=torch.long))


def test_prepare_attention_mask_honors_dtype_for_default():
    mask = prepare_attention_mask(None, batch_size=2, seq_len=3, device="cpu", dtype=torch.bool)
    assert mask.dtype == torch.bool
    assert mask.all()


def test_prepare_attention_mask_returns_caller_mask_as_is():
    supplied = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    out = prepare_attention_mask(supplied, batch_size=2, seq_len=3, device="cpu")
    assert out is supplied


def test_prepare_attention_mask_rejects_bad_shape():
    bad = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="expected"):
        prepare_attention_mask(bad, batch_size=2, seq_len=5, device="cpu")


def test_prepare_attention_mask_rejects_wrong_rank():
    bad = torch.ones(2, 3, 5, dtype=torch.long)
    with pytest.raises(ValueError):
        prepare_attention_mask(bad, batch_size=2, seq_len=3, device="cpu")

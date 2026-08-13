"""Hidden AdamW 2D weights (readout, MLM-head dense, classifier) get muP-scaled LR."""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM, AblmForSequenceClassification
from ablm.training.optim import build_muon_optimizer

pytestmark = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="DistributedMuon reuses torch.optim._muon internals (torch >= 2.11)",
)


def _model(mup: bool) -> AblmForMaskedLM:
    kw = dict(mup_enabled=True, mup_base_hidden_size=256) if mup else {}
    cfg = AblmConfig(hidden_size=512, num_attention_heads=8, num_hidden_layers=2, **kw)
    return AblmForMaskedLM(cfg)


def _group_lrs(opt) -> list[float]:
    return [g["lr"] for g in opt.param_groups]


def _muon_adjust_lr_fn(opt) -> str | None:
    # The Muon child is the first sub-optimizer; its adjust_lr_fn lives on the group.
    return opt.optimizers[0].param_groups[0]["adjust_lr_fn"]


def test_mup_forces_original_muon_lr_rule():
    """muP LR transfer requires Muon's aspect-ratio-only 'original' rule -- not
    match_rms_adamw, which scales ~sqrt(width) and breaks transfer."""
    assert _muon_adjust_lr_fn(build_muon_optimizer(_model(mup=True), lr=1e-3)) == "original"


def test_non_mup_keeps_match_rms_adamw():
    """Off muP, the single-scale Moonshot RMS-match rule is correct (and the default)."""
    assert _muon_adjust_lr_fn(build_muon_optimizer(_model(mup=False), lr=1e-3)) == "match_rms_adamw"


def test_explicit_match_rms_adamw_under_mup_raises():
    with pytest.raises(ValueError, match="match_rms_adamw"):
        build_muon_optimizer(_model(mup=True), lr=1e-3, adjust_lr_fn="match_rms_adamw")


def test_disabled_has_no_extra_mup_group():
    opt = build_muon_optimizer(_model(mup=False), lr=1e-3)
    # 1 Muon group + 2 AdamW groups (decay / no-decay).
    assert len(opt.param_groups) == 3
    assert all(lr == 1e-3 for lr in _group_lrs(opt))


def test_enabled_scales_all_hidden_matrices_but_not_input_embedding():
    model = _model(mup=True)
    readout = model.lm_head.decoder.weight
    mlm_dense = model.lm_head.dense.weight
    input_emb = model.get_input_embeddings().weight
    opt = build_muon_optimizer(model, lr=1e-3)
    # base/hidden = 256/512 = 0.5
    expected_lr = pytest.approx(1e-3 * 0.5)

    def _group_for(param):
        groups = [g for g in opt.param_groups if any(p is param for p in g["params"])]
        assert len(groups) == 1
        return groups[0]

    assert _group_for(readout)["lr"] == expected_lr
    assert _group_for(mlm_dense)["lr"] == expected_lr
    assert _group_for(input_emb)["lr"] == 1e-3
    # and none of these are in a Muon group (still AdamW).


def test_sequence_classification_builds_without_output_embeddings():
    # AblmForSequenceClassification.get_output_embeddings() is None; the mup group
    # must be built from get_input_embeddings() instead, or this raises AttributeError.
    cfg = AblmConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        mup_enabled=True,
        mup_base_hidden_size=256,
    )
    model = AblmForSequenceClassification(cfg)
    opt = build_muon_optimizer(model, lr=1e-3)
    classifier = model.classifier.weight
    groups = [g for g in opt.param_groups if any(p is classifier for p in g["params"])]
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3 * 0.5)

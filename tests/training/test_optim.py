"""Tests for the Muon builder and the CombinedOptimizer facade."""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.optim import CombinedOptimizer, OptimizerSettings, build_muon_optimizer


@pytest.fixture
def tiny_model() -> AblmForMaskedLM:
    cfg = AblmConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return AblmForMaskedLM(cfg)


# ----------------------------------------------------------------------
# Muon partition (2D hidden -> Muon, rest -> AdamW) + CombinedOptimizer
# ----------------------------------------------------------------------


def test_build_muon_returns_combined_optimizer(tiny_model):
    opt = build_muon_optimizer(tiny_model, OptimizerSettings(lr=1e-3))
    assert isinstance(opt, CombinedOptimizer)
    assert isinstance(opt, torch.optim.Optimizer)  # required by LR schedulers
    assert len(opt.optimizers) == 2  # Muon + AdamW


def test_combined_param_groups_cover_all_params_without_duplication(tiny_model):
    opt = build_muon_optimizer(tiny_model, OptimizerSettings())
    in_groups = [p for g in opt.param_groups for p in g["params"]]
    in_group_ids = {id(p) for p in in_groups}
    trainable_ids = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert len(in_groups) == len(in_group_ids)  # no duplicates
    assert in_group_ids == trainable_ids  # full coverage


def test_muon_group_excludes_embeddings_and_output_heads(tiny_model):
    # The Muon child is the first sub-optimizer; its params must be 2D and not
    # embeddings / output projections.
    opt = build_muon_optimizer(tiny_model, OptimizerSettings())
    muon_param_ids = {id(p) for g in opt.optimizers[0].param_groups for p in g["params"]}
    for name, p in tiny_model.named_parameters():
        if id(p) in muon_param_ids:
            assert p.ndim == 2
            assert "embed" not in name and "lm_head" not in name and "decoder" not in name


def test_combined_optimizer_step_updates_params(tiny_model):
    opt = build_muon_optimizer(tiny_model, OptimizerSettings(lr=1e-2, weight_decay=0.0))
    ids = torch.randint(4, 30, (2, 16))
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    opt.zero_grad()
    out = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    out.loss.backward()
    opt.step()
    changed = [
        n for n, p in tiny_model.named_parameters() if not torch.equal(p.detach(), before[n])
    ]
    assert any("layers.0" in n and "weight" in n for n in changed)
    assert len(changed) > 1


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step")
def test_combined_lr_scheduler_mutates_child_groups(tiny_model):
    opt = build_muon_optimizer(tiny_model, OptimizerSettings(lr=1.0, weight_decay=0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5)
    sched.step()
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(0.5)


def test_combined_state_dict_round_trips(tiny_model):
    opt = build_muon_optimizer(tiny_model, OptimizerSettings(lr=1e-2, weight_decay=0.0))
    ids = torch.randint(4, 30, (2, 16))
    opt.zero_grad()
    out = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    out.loss.backward()
    opt.step()

    sd = opt.state_dict()
    assert "optimizers" in sd and len(sd["optimizers"]) == 2

    opt2 = build_muon_optimizer(tiny_model, OptimizerSettings(lr=1e-2, weight_decay=0.0))
    opt2.load_state_dict(sd)
    assert len(opt2.state_dict()["optimizers"]) == 2

"""Tests for the Muon param split, AdamW grouping, and the CombinedOptimizer facade."""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.optim import (
    CombinedOptimizer,
    build_muon_optimizer,
    split_muon_params,
)


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
# Muon param split (2D transformer-body weights -> Muon, rest -> AdamW)
# ----------------------------------------------------------------------


def test_split_is_exhaustive_and_disjoint(tiny_model):
    muon_params, adam_params, adam_names = split_muon_params(tiny_model)
    in_split = {id(p) for p in (*muon_params, *adam_params)}
    trainable = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert in_split == trainable  # exhaustive
    assert len(muon_params) + len(adam_params) == len(in_split)  # disjoint
    assert muon_params


def test_muon_group_is_2d_body_weights_only(tiny_model):
    muon_params, _, _ = split_muon_params(tiny_model)
    muon_ids = {id(p) for p in muon_params}
    for name, p in tiny_model.named_parameters():
        if id(p) in muon_ids:
            assert p.ndim == 2
            assert name.startswith("ablm.backbone.layers.")
            assert "embed" not in name and not name.startswith("lm_head")


# ----------------------------------------------------------------------
# build_muon_optimizer + CombinedOptimizer facade
# ----------------------------------------------------------------------


def test_build_returns_combined_optimizer_with_two_children(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3)
    assert isinstance(opt, CombinedOptimizer)
    assert isinstance(opt, torch.optim.Optimizer)  # required by LR schedulers
    assert len(opt.optimizers) == 2  # Muon + AdamW


def test_combined_param_groups_cover_all_params_without_duplication(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3)
    in_groups = [p for g in opt.param_groups for p in g["params"]]
    in_group_ids = {id(p) for p in in_groups}
    trainable_ids = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert len(in_groups) == len(in_group_ids)  # no duplicates
    assert in_group_ids == trainable_ids  # full coverage


def test_combined_step_updates_params(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-2)
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
    opt = build_muon_optimizer(tiny_model, lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5)
    sched.step()
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(0.5)


def test_combined_state_dict_uses_standard_flat_layout(tiny_model):
    # The standard {"state": {pid: ...}, "param_groups": [...]} layout is what both the
    # normal optimizer.pt path and torch's FSDP distributed-checkpoint path require.
    opt = build_muon_optimizer(tiny_model, lr=1e-2)
    ids = torch.randint(4, 30, (2, 16))
    opt.zero_grad()
    tiny_model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()
    ).loss.backward()
    opt.step()

    sd = opt.state_dict()
    assert set(sd) == {"state", "param_groups"}
    # param-ids are a contiguous 0..N enumeration over the concatenated groups
    pids = [pid for g in sd["param_groups"] for pid in g["params"]]
    assert pids == list(range(len(pids)))
    assert set(sd["state"]).issubset(set(pids))


def test_combined_state_dict_round_trips_values(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-2)
    ids = torch.randint(4, 30, (2, 16))
    opt.zero_grad()
    tiny_model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()
    ).loss.backward()
    opt.step()
    sd = opt.state_dict()

    opt2 = build_muon_optimizer(tiny_model, lr=1e-2)
    opt2.load_state_dict(sd)
    # The Muon child's momentum buffers and the AdamW child's moments came back intact.
    muon_state = opt2.optimizers[0].state
    assert muon_state and all("momentum_buffer" in s for s in muon_state.values())
    adam_state = opt2.optimizers[1].state
    assert adam_state and all("exp_avg" in s for s in adam_state.values())
    sd2 = opt2.state_dict()
    assert set(sd2["state"]) == set(sd["state"])


# ---------------------------------------------------------------------------
# config fidelity
# ---------------------------------------------------------------------------


def _adamw_child(opt):
    return opt.optimizers[1]


def test_split_returns_names_aligned_with_adam_params(tiny_model):
    _, adam_params, adam_names = split_muon_params(tiny_model)
    assert len(adam_names) == len(adam_params)
    by_name = dict(tiny_model.named_parameters())
    for name, p in zip(adam_names, adam_params, strict=True):
        assert by_name[name] is p


def test_betas_and_eps_are_forwarded_to_adamw(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3, betas=(0.85, 0.97), eps=1e-6)
    for group in _adamw_child(opt).param_groups:
        assert group["betas"] == (0.85, 0.97)
        assert group["eps"] == 1e-6


def test_adamw_betas_default_matches_hf_training_arguments(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3)
    assert _adamw_child(opt).param_groups[0]["betas"] == (0.9, 0.999)


def test_muon_weight_decay_defaults_to_shared_weight_decay(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3, weight_decay=0.05)
    assert opt.optimizers[0].param_groups[0]["weight_decay"] == 0.05


def test_muon_weight_decay_override_is_independent(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-3, weight_decay=0.05, muon_weight_decay=0.0)
    assert opt.optimizers[0].param_groups[0]["weight_decay"] == 0.0
    decay_groups = [g for g in _adamw_child(opt).param_groups if g["weight_decay"] > 0]
    assert decay_groups and decay_groups[0]["weight_decay"] == 0.05


def test_decay_parameter_names_control_the_adamw_split(tiny_model):
    _, _, adam_names = split_muon_params(tiny_model)
    chosen = {adam_names[0]}
    opt = build_muon_optimizer(tiny_model, lr=1e-3, weight_decay=0.1, decay_parameter_names=chosen)
    decay, no_decay = _adamw_child(opt).param_groups
    assert decay["weight_decay"] == 0.1
    assert len(decay["params"]) == 1
    assert no_decay["weight_decay"] == 0.0


def test_build_muon_optimizer_applies_betas_and_eps(tiny_model):
    # betas / eps flow to the AdamW child. Non-default values (not 0.999 / 1e-8) so a
    # match with a default can't mask a forwarding regression. The training script passes
    # these from TrainingArguments; build_muon_optimizer is the single wiring point.
    opt = build_muon_optimizer(tiny_model, lr=1e-3, betas=(0.9, 0.98), eps=1e-5)
    for group in _adamw_child(opt).param_groups:
        assert group["betas"] == (0.9, 0.98)
        assert group["eps"] == 1e-5

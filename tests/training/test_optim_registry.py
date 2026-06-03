"""Tests for the optimizer registry, parameter partitioning, and the Muon facade."""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.grouping import is_muon_eligible, is_no_decay_param, partition_parameters
from ablm.training.muon import CombinedOptimizer, build_muon_optimizer
from ablm.training.optim_registry import (
    OptimizerSettings,
    available_optimizers,
    build_optimizer,
    register_custom_optimizer,
    register_hf_optimizer,
    resolve_optimizer,
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
# Registry
# ----------------------------------------------------------------------


def test_builtin_optimizers_registered():
    assert {"adamw", "adamw_fused", "adafactor", "muon"}.issubset(set(available_optimizers()))


def test_hf_native_resolves_to_optim_string():
    spec = resolve_optimizer("adamw_fused")
    assert spec.hf_optim == "adamw_torch_fused"
    assert not spec.is_custom


def test_muon_resolves_to_custom_builder():
    spec = resolve_optimizer("muon")
    assert spec.is_custom
    assert spec.hf_optim is None


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        resolve_optimizer("nope")


def test_build_hf_native_directly_is_error(tiny_model):
    with pytest.raises(ValueError, match="HF-native"):
        build_optimizer("adamw", tiny_model, OptimizerSettings())


def test_double_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_hf_optimizer("adamw", "adamw_torch")
    with pytest.raises(ValueError, match="already registered"):

        @register_custom_optimizer("muon")
        def _dup(model, settings):  # pragma: no cover
            raise NotImplementedError


# ----------------------------------------------------------------------
# Parameter partitioning
# ----------------------------------------------------------------------


def test_no_decay_classifies_biases_norms_embeddings(tiny_model):
    for name, p in tiny_model.named_parameters():
        if p.ndim <= 1 or "embed" in name:
            assert is_no_decay_param(name, p)


def test_partition_covers_all_trainable_params_without_duplication(tiny_model):
    groups = partition_parameters(tiny_model, use_muon=True)
    grouped = (
        groups.muon_params() + groups.adamw_decay_params() + groups.adamw_no_decay_params()
    )
    grouped_ids = {id(p) for p in grouped}
    trainable_ids = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert len(grouped) == len(grouped_ids)  # no duplicates
    assert grouped_ids == trainable_ids  # full coverage


def test_muon_group_excludes_embeddings_and_output_heads(tiny_model):
    groups = partition_parameters(tiny_model, use_muon=True)
    for name, p in groups.muon:
        assert p.ndim == 2
        assert is_muon_eligible(name, p)
        assert "embed" not in name
        assert "lm_head" not in name and "decoder" not in name


def test_partition_without_muon_has_empty_muon_group(tiny_model):
    groups = partition_parameters(tiny_model, use_muon=False)
    assert groups.muon == []


# ----------------------------------------------------------------------
# Muon CombinedOptimizer
# ----------------------------------------------------------------------


def test_build_muon_returns_combined_optimizer(tiny_model):
    opt = build_optimizer("muon", tiny_model, OptimizerSettings(lr=1e-3))
    assert isinstance(opt, CombinedOptimizer)
    assert isinstance(opt, torch.optim.Optimizer)  # required by LR schedulers
    assert len(opt.optimizers) == 2  # Muon + AdamW


def test_combined_param_groups_cover_all_params(tiny_model):
    opt = build_optimizer("muon", tiny_model, OptimizerSettings())
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    trainable = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert in_groups == trainable


def test_combined_optimizer_step_updates_params(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-2, weight_decay=0.0)
    ids = torch.randint(4, 30, (2, 16))
    labels = ids.clone()
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    opt.zero_grad()
    out = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    out.loss.backward()
    opt.step()
    changed = [
        n for n, p in tiny_model.named_parameters() if not torch.equal(p.detach(), before[n])
    ]
    # Both a Muon-eligible (2D hidden) and an AdamW param should have moved.
    assert any("layers.0" in n and "weight" in n for n in changed)
    assert len(changed) > 1


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step")
def test_combined_lr_scheduler_mutates_child_groups(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1.0, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5)
    sched.step()  # apply multiplier
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(0.5)


def test_combined_state_dict_round_trips(tiny_model):
    opt = build_muon_optimizer(tiny_model, lr=1e-2, weight_decay=0.0)
    ids = torch.randint(4, 30, (2, 16))
    opt.zero_grad()
    out = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    out.loss.backward()
    opt.step()

    sd = opt.state_dict()
    assert "optimizers" in sd and len(sd["optimizers"]) == 2

    opt2 = build_muon_optimizer(tiny_model, lr=1e-2, weight_decay=0.0)
    opt2.load_state_dict(sd)
    assert len(opt2.state_dict()["optimizers"]) == 2

"""Tests for `ablm.training.data.metrics` — per-region eval CE/accuracy + RegionEvalMixin."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Trainer, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.data import RegionEvalMixin, compute_metrics, per_token_ce_and_hits

# ---------------------------------------------------------------------------
# per_token_ce_and_hits
# ---------------------------------------------------------------------------


def test_per_token_ce_and_hits_shapes_and_dtypes():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    ce, hit = per_token_ce_and_hits(logits, labels)
    assert ce.shape == (2, 5)
    assert hit.shape == (2, 5)
    assert ce.dtype == torch.float32
    assert hit.dtype == torch.int8


def test_per_token_ce_and_hits_zero_at_ignored_positions():
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[1, -100, 2, -100]])
    ce, hit = per_token_ce_and_hits(logits, labels)
    assert ce[0, 1].item() == 0.0
    assert ce[0, 3].item() == 0.0
    assert hit[0, 1].item() == 0
    assert hit[0, 3].item() == 0


def test_per_token_ce_and_hits_matches_manual_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 5)
    labels = torch.randint(0, 5, (2, 3))
    ce, hit = per_token_ce_and_hits(logits, labels)
    expected_ce = torch.nn.functional.cross_entropy(
        logits.view(-1, 5), labels.view(-1), reduction="none"
    ).view(2, 3)
    assert torch.allclose(ce, expected_ce, atol=1e-6)
    assert torch.equal(hit.bool(), logits.argmax(dim=-1) == labels)


def test_per_token_ce_and_hits_top1_hit_is_correct():
    # logits deterministically favor class 2 everywhere.
    logits = torch.zeros(1, 3, 4)
    logits[..., 2] = 10.0
    labels = torch.tensor([[2, 0, 2]])
    ce, hit = per_token_ce_and_hits(logits, labels)
    assert hit.tolist() == [[1, 0, 1]]


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def _eval_pred(ce, region, hit, label_ids):
    return SimpleNamespace(
        predictions={"ce": np.asarray(ce), "region": np.asarray(region), "hit": np.asarray(hit)},
        label_ids=np.asarray(label_ids),
    )


def test_compute_metrics_returns_all_expected_keys():
    ce = [[1.0, 2.0, 3.0, 4.0]]
    region = [[0, 1, 2, 3]]
    hit = [[1, 0, 1, 0]]
    labels = [[5, 5, 5, 5]]  # all masked (none == -100)
    metrics = compute_metrics(_eval_pred(ce, region, hit, labels))
    expected_keys = {
        "CE_overall",
        "ACC_overall",
        "CE_non_cdr",
        "ACC_non_cdr",
        "CE_cdr1",
        "ACC_cdr1",
        "CE_cdr2",
        "ACC_cdr2",
        "CE_cdr3",
        "ACC_cdr3",
    }
    assert set(metrics) == expected_keys


def test_compute_metrics_ce_overall_equals_mean_over_masked_positions():
    ce = [[1.0, 2.0, 3.0, 4.0, 100.0]]
    region = [[0, 1, 2, 3, 0]]
    hit = [[1, 0, 1, 0, 1]]
    labels = [[5, 5, 5, 5, -100]]  # last position is NOT masked (ignored)
    metrics = compute_metrics(_eval_pred(ce, region, hit, labels))
    assert metrics["CE_overall"] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4)
    assert metrics["ACC_overall"] == pytest.approx((1 + 0 + 1 + 0) / 4)


def test_compute_metrics_partitions_by_cdr_level_aliasing_shm():
    # region 4 (FW+SHM) aliases into non_cdr; region 5 (CDR1+SHM) aliases into cdr1.
    ce = [[1.0, 2.0, 3.0, 4.0]]
    region = [[0, 4, 1, 5]]
    hit = [[1, 1, 0, 0]]
    labels = [[5, 5, 5, 5]]
    metrics = compute_metrics(_eval_pred(ce, region, hit, labels))
    assert metrics["CE_non_cdr"] == pytest.approx((1.0 + 2.0) / 2)
    assert metrics["CE_cdr1"] == pytest.approx((3.0 + 4.0) / 2)
    assert np.isnan(metrics["CE_cdr2"])
    assert np.isnan(metrics["CE_cdr3"])


def test_compute_metrics_four_levels_sum_to_overall_when_all_regions_valid():
    # region 4 (FW+SHM) aliases into non_cdr -> that level covers 2 of the 5 positions.
    ce = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    region = [[0, 1, 2, 3, 4]]
    hit = [[1, 1, 1, 1, 1]]
    labels = [[5, 5, 5, 5, 5]]
    metrics = compute_metrics(_eval_pred(ce, region, hit, labels))
    weighted_sum = (
        metrics["CE_non_cdr"] * 2 + metrics["CE_cdr1"] + metrics["CE_cdr2"] + metrics["CE_cdr3"]
    ) / 5
    assert weighted_sum == pytest.approx(metrics["CE_overall"])


def test_compute_metrics_returns_nan_when_no_positions_masked():
    ce = [[1.0, 2.0]]
    region = [[0, 1]]
    hit = [[1, 0]]
    labels = [[-100, -100]]
    metrics = compute_metrics(_eval_pred(ce, region, hit, labels))
    assert np.isnan(metrics["CE_overall"])
    assert np.isnan(metrics["ACC_overall"])


# ---------------------------------------------------------------------------
# RegionEvalMixin: get_eval_dataloader collator swap (isolated from real Trainer)
# ---------------------------------------------------------------------------


class _FakeTrainerBase:
    """Minimal stand-in exposing just what RegionEvalMixin's MRO needs."""

    def __init__(self, data_collator=None):
        self.data_collator = data_collator

    def get_eval_dataloader(self, eval_dataset=None):
        # Return the collator in effect *during* the call, so the swap is observable.
        return self.data_collator


class _MixedFake(RegionEvalMixin, _FakeTrainerBase):
    pass


def test_get_eval_dataloader_uses_eval_collator_and_restores_original():
    orig, eval_collator = object(), object()
    obj = _MixedFake(data_collator=orig, eval_data_collator=eval_collator)
    used = obj.get_eval_dataloader()
    assert used is eval_collator
    assert obj.data_collator is orig  # restored after the call


def test_get_eval_dataloader_falls_through_when_no_eval_collator_set():
    orig = object()
    obj = _MixedFake(data_collator=orig)
    used = obj.get_eval_dataloader()
    assert used is orig


# ---------------------------------------------------------------------------
# RegionEvalMixin composed with the real transformers.Trainer
# ---------------------------------------------------------------------------


class _RegionTrainer(RegionEvalMixin, Trainer):
    pass


@pytest.fixture
def tiny_model() -> AblmForMaskedLM:
    cfg = AblmConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return AblmForMaskedLM(cfg)


def test_region_eval_mixin_composes_with_trainer(tiny_model: AblmForMaskedLM, tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer = _RegionTrainer(model=tiny_model, args=args)
    assert isinstance(trainer, Trainer)
    assert trainer.eval_data_collator is None  # eval_data_collator kwarg consumed, not forwarded


def test_region_eval_mixin_prediction_step_reduces_logits_and_carries_region(
    tiny_model: AblmForMaskedLM, tmp_path
):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer = _RegionTrainer(model=tiny_model, args=args)

    input_ids = torch.randint(4, 30, (2, 6))
    labels = input_ids.clone()
    labels[:, 0] = -100
    region_mask = torch.zeros(2, 6, dtype=torch.long)
    region_mask[:, 0] = -1
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "region_mask": region_mask,
    }

    loss, outputs, out_labels = trainer.prediction_step(
        tiny_model, inputs, prediction_loss_only=False
    )
    assert loss is not None
    assert set(outputs) == {"ce", "region", "hit"}
    assert outputs["ce"].shape == (2, 6)
    assert outputs["ce"].dtype == torch.float32
    assert outputs["hit"].dtype == torch.int8
    assert outputs["region"].dtype == torch.int8
    assert torch.equal(outputs["region"].cpu(), region_mask.to(torch.int8))
    assert torch.equal(out_labels.cpu(), labels)
    # region_mask must not leak through to the wrapped model call.
    assert "region_mask" not in inputs


def test_region_eval_mixin_strips_region_mask_in_training(tiny_model: AblmForMaskedLM, tmp_path):
    """Training compute_loss must drop region_mask before model.forward: the collator carries
    it on every batch and the model does not accept it. Without the strip this raises the
    `TypeError: forward() got an unexpected keyword argument 'region_mask'` the anchor smoke hit."""
    # use_cpu: compute_loss (called directly) doesn't move inputs to device like the training
    # loop's _prepare_inputs does, so keep model + inputs both on CPU for this unit test.
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    trainer = _RegionTrainer(model=tiny_model, args=args)

    input_ids = torch.randint(4, 30, (2, 6))
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "region_mask": torch.zeros(2, 6, dtype=torch.long),
    }
    loss = trainer.compute_loss(tiny_model, inputs)
    assert torch.isfinite(loss).item()
    assert "region_mask" not in inputs  # stripped before reaching model.forward


def test_region_eval_mixin_prediction_loss_only_skips_reduction(
    tiny_model: AblmForMaskedLM, tmp_path
):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer = _RegionTrainer(model=tiny_model, args=args)

    input_ids = torch.randint(4, 30, (1, 5))
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "region_mask": torch.zeros(1, 5, dtype=torch.long),
    }
    loss, logits, labels = trainer.prediction_step(tiny_model, inputs, prediction_loss_only=True)
    assert loss is not None
    # untouched by per_token_ce_and_hits: logits stay whatever the base Trainer returned
    assert not isinstance(logits, dict)

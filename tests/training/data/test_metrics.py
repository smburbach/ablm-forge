"""Tests for `ablm.training.data.metrics` — per-region eval CE/accuracy + RegionEvalMixin."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Trainer, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.data import (
    MaskingStatsMixin,
    RegionEvalMixin,
    compute_metrics,
    masking_stats,
    per_token_ce_and_hits,
)

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
        f"{m}_{s}"
        for m in ("CE", "ACC")
        for s in ("overall", "non_cdr", "cdr1", "cdr2", "cdr3", "templated", "non_templated")
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


# ---------------------------------------------------------------------------
# seq_mutated split, masking_stats, side-channel stripping, MaskingStatsMixin
# ---------------------------------------------------------------------------


def test_compute_metrics_adds_mutation_split_when_present():
    ce = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    region = np.array([[0, 1], [4, 7]], dtype=np.int8)
    hit = np.array([[1, 0], [0, 1]], dtype=np.int8)
    labels = np.array([[5, 5], [5, 5]])
    mutated = np.array([[0, 0], [1, 1]], dtype=np.int8)
    pred = SimpleNamespace(
        predictions={"ce": ce, "region": region, "hit": hit, "mutated": mutated},
        label_ids=labels,
    )
    m = compute_metrics(pred)
    assert m["CE_seq_unmutated"] == pytest.approx(1.5)
    assert m["CE_seq_mutated"] == pytest.approx(3.5)
    assert m["CE_templated"] == pytest.approx(1.5)
    assert m["CE_non_templated"] == pytest.approx(3.5)


def test_masking_stats_on_a_known_batch():
    labels = torch.tensor([[-100, 7, -100, 7, -100, -100], [-100, -100, -100, 7, 7, -100]])
    region = torch.tensor([[-1, 0, 0, 1, 5, -1], [-1, 0, 4, 4, 3, -1]])
    s = masking_stats(labels, region, p=0.5)
    assert s["mask_rate"] == pytest.approx(4 / 8)
    assert s["rate_fw_templated"] == pytest.approx(1 / 3)
    assert s["rate_cdr_templated"] == pytest.approx(2 / 2)
    assert s["rate_fw_shm"] == pytest.approx(1 / 2)
    assert s["rate_cdr_shm"] == pytest.approx(0 / 1)
    assert s["count_z_sd"] == pytest.approx(0.0)


def test_region_eval_mixin_strips_side_channels_in_training(tiny_model: AblmForMaskedLM, tmp_path):
    """Forge's forward has no **kwargs: region_mask and seq_mutated must never reach it."""

    class T(RegionEvalMixin, Trainer):
        pass

    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True)
    trainer = T(model=tiny_model, args=args)
    ids = torch.randint(4, 30, (2, 8))
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "region_mask": torch.zeros_like(ids),
        "seq_mutated": torch.tensor([0, 1]),
    }
    loss = trainer.compute_loss(tiny_model, inputs)
    assert torch.isfinite(loss)


def test_region_eval_mixin_prediction_step_carries_mutated(tiny_model: AblmForMaskedLM, tmp_path):
    class T(RegionEvalMixin, Trainer):
        pass

    trainer = T(model=tiny_model, args=TrainingArguments(output_dir=str(tmp_path), report_to=[]))
    ids = torch.randint(4, 30, (2, 8))
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "region_mask": torch.zeros_like(ids),
        "seq_mutated": torch.tensor([0, 1]),
    }
    _, out, _ = trainer.prediction_step(tiny_model, inputs, prediction_loss_only=False)
    assert out["mutated"].shape == ids.shape
    assert out["mutated"][0].tolist() == [0] * 8 and out["mutated"][1].tolist() == [1] * 8


def test_masking_stats_mixin_logs_train_mask_keys(tiny_model: AblmForMaskedLM, tmp_path):
    from datasets import Dataset

    from ablm.model.tokenization_ablm import AblmTokenizerFast
    from ablm.training.data import RegionAwareCollator, add_region_mask

    tokenizer = AblmTokenizerFast()
    rows = [
        add_region_mask(
            {
                "s:0": "M" * 12,
                "c:0": "0" * 12,
                "n:0": "0" * 12,
                "s:1": "A" * 8,
                "c:1": "0" * 8,
                "n:1": "0" * 8,
            },
            tokenizer,
            seq_col="s",
            cdr_col="c",
            nt_col="n",
        )
        for _ in range(8)
    ]
    ds = Dataset.from_list(rows)

    class T(MaskingStatsMixin, RegionEvalMixin, Trainer):
        pass

    trainer = T(
        model=tiny_model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            report_to=[],
            max_steps=2,
            logging_steps=1,
            per_device_train_batch_size=4,
            remove_unused_columns=False,
        ),
        train_dataset=ds,
        data_collator=RegionAwareCollator(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=1
        ),
    )
    trainer.train()
    logged = [e for e in trainer.state.log_history if "mask/mask_rate" in e]
    assert logged, trainer.state.log_history
    assert 0.0 < logged[-1]["mask/mask_rate"] < 0.5
    assert "mask/rate_fw_templated" in logged[-1]

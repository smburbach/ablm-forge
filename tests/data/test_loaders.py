"""Tests for the 🤗 datasets-based streaming loader and the MLM collator."""

from __future__ import annotations

import pytest
import torch
from transformers import DataCollatorForLanguageModeling

from ablm.config import DataConfig
from ablm.data.loaders import build_collator, build_train_dataset


def _data(path, **kw) -> DataConfig:
    return DataConfig(train=str(path), **kw)


def test_build_train_dataset_yields_tokenized_examples(training_parquet):
    ds = build_train_dataset(_data(training_parquet), max_length=64, seed=0)
    ex = next(iter(ds))
    assert {"input_ids", "attention_mask", "special_tokens_mask"} <= set(ex)
    assert "sequence" not in ex  # original parquet columns are dropped by .map
    assert len(ex["input_ids"]) <= 64  # truncated to max_length


def test_build_train_dataset_requires_data():
    with pytest.raises(ValueError, match="No training data"):
        build_train_dataset(DataConfig(train=None), max_length=64, seed=0)


def test_build_train_dataset_interleaves_multiple_sources(training_parquet):
    cfg = DataConfig(
        train={
            "a": {"path": str(training_parquet), "fraction": 0.5},
            "b": {"path": str(training_parquet), "fraction": 0.5},
        }
    )
    ds = build_train_dataset(cfg, max_length=64, seed=0)
    examples = [ex for ex, _ in zip(ds, range(5), strict=False)]
    assert len(examples) == 5
    assert all("input_ids" in e for e in examples)


def test_build_collator_masks_and_pads(training_parquet):
    cfg = _data(training_parquet)
    ds = build_train_dataset(cfg, max_length=64, seed=0)
    collator = build_collator(cfg)
    assert isinstance(collator, DataCollatorForLanguageModeling)

    batch = collator([ex for ex, _ in zip(ds, range(4), strict=False)])
    assert {"input_ids", "attention_mask", "labels"} <= set(batch)
    assert batch["input_ids"].shape[0] == 4
    # Dynamic padding: a rectangular batch.
    assert batch["input_ids"].ndim == 2
    # Unmasked positions are ignored in the loss (-100).
    assert bool((batch["labels"] == -100).any())
    # Masked targets are real token ids copied from input.
    assert bool((batch["labels"] != -100).any())


def test_build_collator_uses_configured_probabilities(training_parquet):
    collator = build_collator(
        DataConfig(
            train=str(training_parquet),
            mask_prob=0.3,
            mask_token_prob=0.7,
            random_token_prob=0.2,
        )
    )
    assert collator.mlm_probability == pytest.approx(0.3)
    assert collator.mask_replace_prob == pytest.approx(0.7)
    assert collator.random_replace_prob == pytest.approx(0.2)


def test_collated_batch_runs_through_model(training_parquet):
    """The dataset + collator output is directly consumable by AblmForMaskedLM."""
    from ablm import AblmConfig, AblmForMaskedLM

    cfg = _data(training_parquet)
    ds = build_train_dataset(cfg, max_length=64, seed=0)
    batch = build_collator(cfg)([ex for ex, _ in zip(ds, range(2), strict=False)])

    model = AblmForMaskedLM(
        AblmConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    )
    out = model(**batch)
    assert torch.isfinite(out.loss)

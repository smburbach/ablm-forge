"""Tests for the 🤗 datasets-based streaming training loader."""

from __future__ import annotations

import pytest
import torch
from transformers import DataCollatorForLanguageModeling

from ablm.data import get_tokenizer
from ablm.data.loaders import build_train_dataset


def test_build_train_dataset_yields_tokenized_examples(training_parquet):
    ds = build_train_dataset(str(training_parquet), max_length=64, seed=0)
    ex = next(iter(ds))
    assert {"input_ids", "attention_mask", "special_tokens_mask"} <= set(ex)
    assert "sequence" not in ex  # original parquet columns are dropped by .map
    assert len(ex["input_ids"]) <= 64  # truncated to max_length


def test_build_train_dataset_requires_data():
    with pytest.raises(ValueError, match="No training data"):
        build_train_dataset([], max_length=64, seed=0)


def test_build_train_dataset_interleaves_multiple_sources(training_parquet):
    ds = build_train_dataset(
        [str(training_parquet), str(training_parquet)],
        probabilities=[0.5, 0.5],
        max_length=64,
        seed=0,
    )
    examples = [ex for ex, _ in zip(ds, range(5), strict=False)]
    assert len(examples) == 5
    assert all("input_ids" in e for e in examples)


def test_build_train_dataset_rejects_mismatched_probabilities(training_parquet):
    with pytest.raises(ValueError, match="probabilities has"):
        build_train_dataset([str(training_parquet)] * 2, probabilities=[1.0], max_length=64, seed=0)


def test_collated_batch_runs_through_model(training_parquet):
    """The dataset output pads/masks via the HF collator and runs through the model."""
    from ablm import AblmConfig, AblmForMaskedLM

    ds = build_train_dataset(str(training_parquet), max_length=64, seed=0)
    collator = DataCollatorForLanguageModeling(tokenizer=get_tokenizer(), mlm=True)
    batch = collator([ex for ex, _ in zip(ds, range(2), strict=False)])
    assert {"input_ids", "attention_mask", "labels"} <= set(batch)
    assert bool((batch["labels"] == -100).any())  # unmasked positions ignored

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

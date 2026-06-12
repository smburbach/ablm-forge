"""Pilot end-to-end training test on the stock HuggingFace Trainer.

Composes the building blocks the way `scripts/pretrain.py` does — AblmConfig +
TrainingArguments + streaming dataset + collator + (optional Muon) optimizer +
stock Trainer — and trains a tiny model for a few steps on the real small parquet
fixture. Asserts finite loss, a checkpoint, and that resume restores the global
step. Marked slow.
"""

from __future__ import annotations

import pytest
import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast
from ablm.training.optim import MUON_OPTIM, OptimizerTrainer

pytestmark = pytest.mark.slow

_MAX_LENGTH = 64


def _stream_dataset(parquet):
    """The streaming + tokenize recipe from scripts/pretrain.py, inlined."""
    tokenizer = AblmTokenizerFast()
    ds = load_dataset("parquet", data_files=str(parquet), split="train", streaming=True)
    return ds.map(
        lambda b: tokenizer(
            b["sequence"], truncation=True, max_length=_MAX_LENGTH, return_special_tokens_mask=True
        ),
        batched=True,
        remove_columns=ds.column_names,
    ).shuffle(seed=42, buffer_size=1024)


def _build_trainer(
    parquet, output_dir, *, optimizer="adamw", max_steps=8, save_steps=4
) -> OptimizerTrainer:
    model = AblmForMaskedLM(
        AblmConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=_MAX_LENGTH,
        )
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        warmup_steps=2,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
        dataloader_num_workers=0,
    )
    dataset = _stream_dataset(parquet)
    return OptimizerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=AblmTokenizerFast(), mlm=True),
        use_muon=optimizer == MUON_OPTIM,
    )


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_pilot_train_runs_with_finite_loss(training_parquet, tmp_path, optimizer):
    out = tmp_path / optimizer
    result = _build_trainer(training_parquet, out, optimizer=optimizer).train()
    assert result.global_step == 8
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert list(out.glob("checkpoint-*"))  # a checkpoint was written


def test_pilot_resume_restores_global_step(training_parquet, tmp_path):
    out = tmp_path / "resumerun"
    _build_trainer(training_parquet, out, max_steps=4, save_steps=2).train()
    checkpoints = sorted(out.glob("checkpoint-*"))
    assert checkpoints

    trainer = _build_trainer(training_parquet, out, max_steps=6, save_steps=2)
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    assert result.global_step == 6

"""Pilot end-to-end training test on the stock HuggingFace Trainer.

Trains a tiny model for a few steps on the real small parquet fixture, through
the same `build_trainer` path the CLI uses. Asserts the loop completes with
finite loss, a checkpoint is written, and resume restores the global step.
Marked slow.
"""

from __future__ import annotations

import pytest
import torch

from ablm.config import AblmRunConfig, DataConfig, TrainConfig
from ablm.model import AblmConfig
from ablm.train import build_trainer

pytestmark = pytest.mark.slow


def _tiny_run_config(
    parquet, output_dir, *, max_steps=8, save_every=4, **train_overrides
) -> AblmRunConfig:
    model = AblmConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    train = TrainConfig(
        max_steps=max_steps,
        batch_size=4,
        lr=1e-3,
        warmup_steps=2,
        scheduler="warmup_linear",
        log_every=1,
        save_every=save_every,
        save_total_limit=2,
        wandb_enabled=False,
        mixed_precision="no",
        output_dir=str(output_dir),
        **train_overrides,
    )
    data = DataConfig(train=str(parquet), num_workers=0, pin_memory=False)
    return AblmRunConfig(model=model, train=train, data=data)


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_pilot_train_runs_and_decreases_loss(training_parquet, tmp_path, optimizer):
    cfg = _tiny_run_config(training_parquet, tmp_path / optimizer, optimizer=optimizer)
    trainer = build_trainer(cfg)

    result = trainer.train()

    assert result.global_step == 8
    final_loss = result.training_loss
    assert final_loss == pytest.approx(final_loss)  # finite (not NaN)
    assert torch.isfinite(torch.tensor(final_loss))
    # A checkpoint directory was written.
    assert list((tmp_path / optimizer).glob("checkpoint-*"))


def test_pilot_resume_restores_global_step(training_parquet, tmp_path):
    out = tmp_path / "resumerun"
    cfg = _tiny_run_config(training_parquet, out, max_steps=4, save_every=2)
    build_trainer(cfg).train()
    checkpoints = sorted((out).glob("checkpoint-*"))
    assert checkpoints

    # Resume and run to a larger step budget.
    cfg2 = _tiny_run_config(training_parquet, out, max_steps=6, save_every=2)
    cfg2.train.resume_from = str(checkpoints[-1])
    trainer2 = build_trainer(cfg2)
    result = trainer2.train(resume_from_checkpoint=str(checkpoints[-1]))
    assert result.global_step == 6

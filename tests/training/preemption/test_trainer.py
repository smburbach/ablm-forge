"""Tests for `ablm.training.preemption.trainer` on a tiny CPU model."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.preemption import PreemptionSafeTrainer
from ablm.training.preemption.trainer import S5cmdSyncCallback, SigtermShieldedWorkerInit

if TYPE_CHECKING:
    from pathlib import Path

    from tests.training.preemption.conftest import FakeS5cmd


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


@pytest.fixture
def tiny_dataset() -> Dataset:
    rows = [{"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]} for _ in range(8)]
    return Dataset.from_list(rows)


@pytest.fixture
def collator() -> DataCollatorForLanguageModeling:
    from ablm import AblmTokenizerFast

    return DataCollatorForLanguageModeling(AblmTokenizerFast(), mlm=True)


@pytest.fixture(autouse=True)
def _off_slurm(restore_sigterm: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.delenv("JOB_WORK_DIR", raising=False)


def _args(tmp_path: Path, **kwargs: object) -> TrainingArguments:
    return TrainingArguments(output_dir=str(tmp_path / "out"), report_to=[], use_cpu=True, **kwargs)


def test_sigterm_shielded_worker_init_ignores_sigterm_and_chains() -> None:
    seen: list[int] = []
    SigtermShieldedWorkerInit(seen.append)(3)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    assert seen == [3]


def test_mixin_adds_sync_callback(tiny_model: AblmForMaskedLM, tmp_path: Path) -> None:
    trainer = PreemptionSafeTrainer(
        model=tiny_model, args=_args(tmp_path, enable_jit_checkpoint=False)
    )
    callbacks = trainer.callback_handler.callbacks
    assert any(isinstance(cb, S5cmdSyncCallback) for cb in callbacks)
    assert trainer.sigterm_received is False


def test_mixin_requires_srun_step_under_slurm(
    tiny_model: AblmForMaskedLM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(RuntimeError, match="srun"):
        PreemptionSafeTrainer(model=tiny_model, args=_args(tmp_path, enable_jit_checkpoint=True))


def test_dataloader_workers_are_shielded_under_jit(
    tiny_model: AblmForMaskedLM,
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, enable_jit_checkpoint=True, dataloader_num_workers=1)
    trainer = PreemptionSafeTrainer(
        model=tiny_model, args=args, train_dataset=tiny_dataset, data_collator=collator
    )
    loader = trainer.get_train_dataloader()
    inner = getattr(loader, "base_dataloader", loader)
    assert isinstance(inner.worker_init_fn, SigtermShieldedWorkerInit)


def test_train_fresh_then_resume_off_cluster(
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
    fake_s5cmd: FakeS5cmd,
) -> None:
    def build() -> PreemptionSafeTrainer:
        torch_cfg = AblmConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
        return PreemptionSafeTrainer(
            model=AblmForMaskedLM(torch_cfg),
            args=_args(tmp_path, max_steps=2, save_steps=1, save_strategy="steps"),
            train_dataset=tiny_dataset,
            data_collator=collator,
        )

    build().train()
    assert (tmp_path / "out" / "checkpoint-2").is_dir()

    resumed = build()
    resumed.train()
    assert resumed.state.global_step == 2
    assert fake_s5cmd.calls() == []

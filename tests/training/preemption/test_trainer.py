"""Tests for `ablm.training.preemption.trainer` on a tiny CPU model."""

from __future__ import annotations

import signal
from functools import partial
from typing import TYPE_CHECKING

import pytest
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from transformers.trainer_utils import seed_worker

from ablm import AblmConfig, AblmForMaskedLM
from ablm.training.preemption import PreemptionSafeMixin, PreemptionSafeTrainer
from ablm.training.preemption import trainer as preemption_trainer
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

    def forbidden_finish(self: object, exit_code: int = 0) -> None:
        raise RuntimeError(f"finish({exit_code}) called")

    monkeypatch.setattr(PreemptionSafeMixin, "finish", forbidden_finish)
    monkeypatch.setattr(PreemptionSafeTrainer, "finish", forbidden_finish, raising=False)


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


def _loader_init_fn(tiny_model, tiny_dataset, collator, tmp_path, jit: bool):
    args = _args(tmp_path, enable_jit_checkpoint=jit, dataloader_num_workers=1)
    trainer = PreemptionSafeTrainer(
        model=tiny_model, args=args, train_dataset=tiny_dataset, data_collator=collator
    )
    loader = trainer.get_train_dataloader()
    return getattr(loader, "base_dataloader", loader).worker_init_fn


def test_dataloader_workers_are_shielded_under_jit(
    tiny_model: AblmForMaskedLM,
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
) -> None:
    init_fn = _loader_init_fn(tiny_model, tiny_dataset, collator, tmp_path, jit=True)
    assert isinstance(init_fn, SigtermShieldedWorkerInit)
    assert isinstance(init_fn.inner, partial)
    assert init_fn.inner.func is seed_worker


def test_dataloader_workers_are_not_shielded_without_jit(
    tiny_model: AblmForMaskedLM,
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
) -> None:
    init_fn = _loader_init_fn(tiny_model, tiny_dataset, collator, tmp_path, jit=False)
    assert not isinstance(init_fn, SigtermShieldedWorkerInit)


def _build_trainer(
    tiny_dataset: Dataset, collator: DataCollatorForLanguageModeling, tmp_path: Path
) -> PreemptionSafeTrainer:
    cfg = AblmConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return PreemptionSafeTrainer(
        model=AblmForMaskedLM(cfg),
        args=_args(tmp_path, max_steps=2, save_steps=1, save_strategy="steps"),
        train_dataset=tiny_dataset,
        data_collator=collator,
    )


def _count_training_steps(trainer: PreemptionSafeTrainer) -> list[int]:
    steps: list[int] = []
    original = trainer.training_step

    def counting(*args: object, **kwargs: object) -> object:
        steps.append(1)
        return original(*args, **kwargs)

    trainer.training_step = counting  # ty: ignore[invalid-assignment]  # test spy on a bound method
    return steps


def test_train_fresh_then_resume_off_cluster(
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
    fake_s5cmd: FakeS5cmd,
) -> None:
    _build_trainer(tiny_dataset, collator, tmp_path).train()
    state_file = tmp_path / "out" / "checkpoint-2" / "trainer_state.json"
    before = (state_file.stat().st_mtime_ns, state_file.read_bytes())

    resumed = _build_trainer(tiny_dataset, collator, tmp_path)
    steps = _count_training_steps(resumed)
    resumed.train()

    assert resumed.state.global_step == 2
    assert steps == []
    assert (state_file.stat().st_mtime_ns, state_file.read_bytes()) == before
    assert fake_s5cmd.calls() == []


def test_train_restarts_when_resume_checkpoint_is_none(
    tiny_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_trainer(tiny_dataset, collator, tmp_path).train()
    monkeypatch.setattr(preemption_trainer, "resume_checkpoint", lambda args: None)

    restarted = _build_trainer(tiny_dataset, collator, tmp_path)
    steps = _count_training_steps(restarted)
    restarted.train()

    assert len(steps) == 2

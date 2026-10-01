"""Tests for `ablm.training.preemption.object_storage` against a fake `s5cmd`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from transformers import TrainingArguments

from ablm.training.preemption.object_storage import (
    S3_ENDPOINT,
    discard_torn_checkpoints,
    latest_complete_checkpoint,
    mirror_to_object_storage,
    on_cluster,
    restore_from_object_storage,
    resume_checkpoint,
    s3_job_uri,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.training.preemption.conftest import FakeS5cmd

SENTINEL = "checkpoint-is-incomplete.txt"


def _checkpoint(parent: Path, step: int, torn: bool = False) -> Path:
    path = parent / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}")
    if torn:
        (path / SENTINEL).write_text("")
    return path


def test_s3_job_uri_uses_user_and_job_dir(cluster_env: Path) -> None:
    assert s3_job_uri("brineylab-eu") == "s3://brineylab-eu/u/job1"


def test_on_cluster_false_off_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_WORK_DIR", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert on_cluster() is False


def test_on_cluster_true_with_work_dir(cluster_env: Path) -> None:
    assert on_cluster() is True


def test_on_cluster_raises_inside_slurm_without_work_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOB_WORK_DIR", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    with pytest.raises(RuntimeError, match=r"\.env"):
        on_cluster()


def test_mirror_skips_off_cluster(fake_s5cmd: FakeS5cmd, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_WORK_DIR", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert mirror_to_object_storage() is True
    assert fake_s5cmd.calls() == []


def test_mirror_first_bucket_first_try(fake_s5cmd: FakeS5cmd, cluster_env: Path) -> None:
    (cluster_env / "model.bin").write_text("x")
    (cluster_env / ".hidden").write_text("x")
    assert mirror_to_object_storage() is True
    calls = fake_s5cmd.calls()
    assert calls == [
        [
            "s5cmd",
            "--endpoint-url",
            S3_ENDPOINT,
            "sync",
            "--exclude",
            "*/.*",
            "--exclude",
            ".*",
            "--delete",
            f"{cluster_env}/",
            "s3://brineylab-eu/u/job1/",
        ]
    ]
    target = fake_s5cmd.bucket("brineylab-eu") / "u" / "job1"
    assert (target / "model.bin").exists()
    assert not (target / ".hidden").exists()


def test_mirror_retries_same_bucket_once(fake_s5cmd: FakeS5cmd, cluster_env: Path) -> None:
    fake_s5cmd.set_exits("1,0")
    assert mirror_to_object_storage() is True
    calls = fake_s5cmd.calls()
    assert len(calls) == 2
    assert all("s3://brineylab-eu/u/job1/" in call for call in calls)


def test_mirror_falls_back_to_second_bucket(fake_s5cmd: FakeS5cmd, cluster_env: Path) -> None:
    fake_s5cmd.set_exits("1,1,0")
    assert mirror_to_object_storage() is True
    calls = fake_s5cmd.calls()
    assert len(calls) == 3
    assert "s3://brineylab-us-east/u/job1/" in calls[2]


def test_mirror_returns_false_when_every_bucket_refuses(
    fake_s5cmd: FakeS5cmd, cluster_env: Path
) -> None:
    fake_s5cmd.set_exits("1,1,1,1")
    assert mirror_to_object_storage() is False
    assert len(fake_s5cmd.calls()) == 4


def test_discard_torn_checkpoints_removes_only_incomplete(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 100)
    _checkpoint(tmp_path, 200, torn=True)
    discard_torn_checkpoints(str(tmp_path))
    assert (tmp_path / "checkpoint-100").is_dir()
    assert not (tmp_path / "checkpoint-200").exists()


def test_latest_complete_checkpoint_missing_dir_is_none(tmp_path: Path) -> None:
    assert latest_complete_checkpoint(str(tmp_path / "absent")) is None


def test_latest_complete_checkpoint_skips_torn(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 100)
    _checkpoint(tmp_path, 200, torn=True)
    assert latest_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-100")


def test_restore_reads_every_bucket_newest_in_second(
    fake_s5cmd: FakeS5cmd, cluster_env: Path
) -> None:
    _checkpoint(fake_s5cmd.bucket("brineylab-eu") / "u" / "job1", 100)
    _checkpoint(fake_s5cmd.bucket("brineylab-us-east") / "u" / "job1", 200)
    restore_from_object_storage(str(cluster_env))
    assert latest_complete_checkpoint(str(cluster_env)) == str(cluster_env / "checkpoint-200")
    assert (cluster_env / "checkpoint-100").is_dir()
    assert (cluster_env / "checkpoint-200").is_dir()
    assert fake_s5cmd.calls() == [
        [
            "s5cmd",
            "--endpoint-url",
            S3_ENDPOINT,
            "sync",
            f"s3://{bucket}/u/job1/*",
            f"{cluster_env}/",
        ]
        for bucket in ("brineylab-eu", "brineylab-us-east")
    ]


def test_restore_discards_torn_before_next_bucket(fake_s5cmd: FakeS5cmd, cluster_env: Path) -> None:
    eu = fake_s5cmd.bucket("brineylab-eu") / "u" / "job1"
    _checkpoint(eu, 100)
    _checkpoint(eu, 200, torn=True)
    _checkpoint(fake_s5cmd.bucket("brineylab-us-east") / "u" / "job1", 200)
    restore_from_object_storage(str(cluster_env))
    assert latest_complete_checkpoint(str(cluster_env)) == str(cluster_env / "checkpoint-200")
    assert not (cluster_env / "checkpoint-200" / SENTINEL).exists()


def test_resume_checkpoint_fresh_run_does_not_restore(
    fake_s5cmd: FakeS5cmd, cluster_env: Path, tmp_path: Path
) -> None:
    args = TrainingArguments(output_dir=str(tmp_path / "out"), report_to=[])
    assert resume_checkpoint(args) is None
    assert fake_s5cmd.calls() == []


def test_resume_checkpoint_requeued_restores_then_resumes(
    fake_s5cmd: FakeS5cmd, cluster_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    _checkpoint(fake_s5cmd.bucket("brineylab-eu") / "u" / "job1", 50)
    args = TrainingArguments(output_dir=str(cluster_env), report_to=[])
    assert resume_checkpoint(args) == str(cluster_env / "checkpoint-50")
    assert len(fake_s5cmd.calls()) == 2

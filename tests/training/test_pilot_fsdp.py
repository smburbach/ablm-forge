"""Pilot FSDP2 + torchrun end-to-end test.

Launches the example `scripts/pretrain.py` under `torchrun` with `--fsdp` and
asserts the run completes (exit 0) and writes a sharded checkpoint — both the model
shard (`pytorch_model_fsdp_0`) and the optimizer state (`optimizer_0`). The
single-process variant exercises the full launch + FSDP2 wiring on one GPU; the
multi-GPU variant runs the real sharded path when >= 2 GPUs are present.

Both optimizer arms are covered. Muon under FSDP is the load-bearing case: HF
forbids a pre-built `optimizers=` tuple under FSDP (so it must go through
`OptimizerTrainer.create_optimizer`), and `CombinedOptimizer` has to serialize in
the standard flat layout for torch's distributed-checkpoint optimizer save to work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "pretrain.py"
_FIXTURE = _REPO / "tests" / "fixtures" / "training" / "test_sequences.parquet"


def _run_torchrun(
    nproc: int, output_dir: Path, *, optimizer: str = "adamw"
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        str(_SCRIPT),
        "--data",
        str(_FIXTURE),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        "4",
        "--batch-size",
        "4",
        "--warmup-steps",
        "1",
        "--hidden-size",
        "32",
        "--num-layers",
        "2",
        "--num-heads",
        "4",
        "--max-length",
        "64",
        "--save-steps",
        "2",
        "--optimizer",
        optimizer,
        "--bf16",
        "--gradient-checkpointing",
        "--fsdp",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_REPO)


def _assert_sharded_checkpoint(output_dir: Path) -> None:
    assert list(output_dir.glob("checkpoint-*/pytorch_model_fsdp_0")), (
        f"no sharded model checkpoint under {output_dir}: {list(output_dir.glob('*'))}"
    )
    # The optimizer state must save too (this is what crashed before the standard-layout fix).
    assert list(output_dir.glob("checkpoint-*/optimizer_0")), (
        f"no sharded optimizer state under {output_dir}"
    )


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2/torchrun pilot needs CUDA")
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_pilot_fsdp2_single_process(tmp_path, optimizer):
    output_dir = tmp_path / optimizer
    result = _run_torchrun(1, output_dir, optimizer=optimizer)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    _assert_sharded_checkpoint(output_dir)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs for real sharding")
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_pilot_fsdp2_multi_gpu(tmp_path, optimizer):
    output_dir = tmp_path / optimizer
    result = _run_torchrun(2, output_dir, optimizer=optimizer)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    _assert_sharded_checkpoint(output_dir)

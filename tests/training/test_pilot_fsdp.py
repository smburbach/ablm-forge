"""Pilot FSDP2 + torchrun end-to-end test.

Launches the example `scripts/pretrain.py` under `torchrun` with `--fsdp` and
asserts the run completes (exit 0) and writes a sharded checkpoint. The
single-process variant exercises the full launch + FSDP2 wiring on one GPU; the
multi-GPU variant runs the real sharded path when >= 2 GPUs are present.
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


def _run_torchrun(nproc: int, output_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        str(_SCRIPT),
        "--data", str(_FIXTURE),
        "--output-dir", str(output_dir),
        "--max-steps", "4",
        "--batch-size", "4",
        "--warmup-steps", "1",
        "--hidden-size", "32",
        "--num-layers", "2",
        "--num-heads", "4",
        "--max-length", "64",
        "--save-steps", "2",
        "--bf16",
        "--gradient-checkpointing",
        "--fsdp",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_REPO)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2/torchrun pilot needs CUDA")
def test_pilot_fsdp2_single_process(tmp_path):
    output_dir = tmp_path / "out"
    result = _run_torchrun(1, output_dir)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    shards = list(output_dir.glob("checkpoint-*/pytorch_model_fsdp_0"))
    assert shards, f"no sharded checkpoint under {output_dir}: {list(output_dir.glob('*'))}"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs for real sharding")
def test_pilot_fsdp2_multi_gpu(tmp_path):
    output_dir = tmp_path / "out"
    result = _run_torchrun(2, output_dir)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    assert list(output_dir.glob("checkpoint-*/pytorch_model_fsdp_0"))

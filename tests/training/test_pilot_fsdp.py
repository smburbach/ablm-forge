"""Pilot FSDP2 + torchrun end-to-end test.

Launches `ablm.train` under `torchrun` with an FSDP2 config and asserts the run
completes (exit 0) and writes a sharded checkpoint. The single-process variant
exercises the full launch + FSDP2 wiring on one GPU; the multi-GPU variant runs
the real sharded path when >= 2 GPUs are present (skipped otherwise).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "tests" / "fixtures" / "training" / "test_sequences.parquet"


def _write_config(tmp_path: Path, output_dir: Path) -> Path:
    cfg = textwrap.dedent(f"""
        model:
          hidden_size: 32
          num_hidden_layers: 2
          num_attention_heads: 4
          intermediate_size: 64
          max_position_embeddings: 64
        train:
          max_steps: 4
          batch_size: 4
          warmup_steps: 1
          lr: 1.0e-3
          log_every: 1
          save_every: 2
          wandb_enabled: false
          mixed_precision: bf16
          gradient_checkpointing: true
          fsdp: "full_shard auto_wrap"
          output_dir: {output_dir}
        data:
          train: {_FIXTURE}
          num_workers: 0
          pin_memory: false
    """).strip()
    path = tmp_path / "fsdp_run.yaml"
    path.write_text(cfg)
    return path


def _run_torchrun(nproc: int, config_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "-m",
        "ablm.train",
        "--config",
        str(config_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_REPO)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2/torchrun pilot needs CUDA")
def test_pilot_fsdp2_single_process(tmp_path):
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, output_dir)
    result = _run_torchrun(1, config_path)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    # A sharded FSDP checkpoint was written.
    shards = list(output_dir.glob("checkpoint-*/pytorch_model_fsdp_0"))
    assert shards, f"no sharded checkpoint under {output_dir}: {list(output_dir.glob('*'))}"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="training fixture not present")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs for real sharding")
def test_pilot_fsdp2_multi_gpu(tmp_path):
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, output_dir)
    result = _run_torchrun(2, config_path)
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr[-3000:]}"
    assert list(output_dir.glob("checkpoint-*/pytorch_model_fsdp_0"))

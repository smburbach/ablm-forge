"""Slurm preemption safety for a stock HF `Trainer`.

`PreemptionSafeMixin` (and the ready-made `PreemptionSafeTrainer`) wires up the CoreWeave
preemption path: HF's `enable_jit_checkpoint` saves on SIGTERM, each save is mirrored to
object storage with `s5cmd`, a re-queued job restores before resuming, DataLoader workers
are shielded from SIGTERM, and `finish()` uploads, cleans up and exits. Build the trainer,
call `train()`, then `finish()`; the sweep script must launch training as an `srun` step.

This is a tracked port of coreweave-docs `model-training/single-run/jit/preemption.py` at
`d748711`, plus an off-cluster guard (no `JOB_WORK_DIR`: upload and cleanup are skipped so a
local smoke run does not raise). Upstream changes are re-ported, not hand-edited here.
"""

from __future__ import annotations

from .trainer import PreemptionSafeMixin, PreemptionSafeTrainer

__all__ = ["PreemptionSafeMixin", "PreemptionSafeTrainer"]

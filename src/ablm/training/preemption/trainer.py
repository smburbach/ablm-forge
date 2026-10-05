"""Trainer pieces for Slurm preemption: sync callback, worker shield, mixin."""

from __future__ import annotations

import os
import shutil
import signal
import sys
import traceback
from typing import TYPE_CHECKING, Any, NoReturn

from transformers import Trainer, TrainerCallback

from .object_storage import mirror_to_object_storage, on_cluster, resume_checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import TrainerControl, TrainerState, TrainingArguments


def _finalize_wandb() -> None:
    """Close the W&B run: os._exit skips atexit, so wandb never flushes on its own."""
    try:
        import wandb
    except ImportError:  # this module does not require wandb
        return
    if wandb.run is not None:
        wandb.finish()


class S5cmdSyncCallback(TrainerCallback):
    """Mirror each saved checkpoint to object storage before the work dir is freed."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Mirror the work dir after a checkpoint save, on the world-zero process only."""
        if state.is_world_process_zero:
            mirror_to_object_storage()
        return control


class SigtermShieldedWorkerInit:
    """Ignore SIGTERM in DataLoader workers, from startup rather than first collate.

    A worker fetches a whole batch before collating, and a signal in that window kills
    it and aborts the step. Chains the inner fn -- Trainer's seed_worker. Upstream
    (transformers 5.17) still does nothing about workers, so this is still needed.
    """

    def __init__(self, inner: Callable[[int], None] | None) -> None:
        self.inner = inner

    def __call__(self, worker_id: int) -> None:
        """Ignore SIGTERM, then run the wrapped init fn if any."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if self.inner is not None:
            self.inner(worker_id)


class PreemptionSafeMixin:
    """Survive Slurm preemption, for a Trainer built with enable_jit_checkpoint.

    Mix in ahead of Trainer. The parts are useless alone and most fail silently, so they
    live together and drop as a whole.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_callback(S5cmdSyncCallback())  # ty: ignore[unresolved-attribute]  # Trainer method
        self.sigterm_received = False
        if not self.args.enable_jit_checkpoint:  # ty: ignore[unresolved-attribute]  # Trainer attr
            return

        # the srun requirement lives outside this class, so check it loudly
        if os.environ.get("SLURM_JOB_ID") and "SLURM_STEP_ID" not in os.environ:
            raise RuntimeError(
                "enable_jit_checkpoint needs training to run as an srun job step, "
                "otherwise Slurm's preemption signal never reaches it. Launch with "
                "`srun ... accelerate launch pretraining.py` as train.sh does."
            )

        # chain rather than replace: Trainer clears its own flag before saving, so the
        # signal is the only reliable sign JIT (not e.g. early stopping) ended training
        previous = signal.getsignal(signal.SIGTERM)
        if callable(previous):  # SIG_DFL/SIG_IGN: nothing to chain

            def note_sigterm(signum: int, frame: Any) -> None:
                self.sigterm_received = True
                previous(signum, frame)  # ty: ignore[call-top-callable]  # callable() narrows the handler union to Top[Callable]

            signal.signal(signal.SIGTERM, note_sigterm)

    def _get_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Shield workers on every loader Trainer builds -- train, eval and test."""
        dataloader = super()._get_dataloader(*args, **kwargs)  # ty: ignore[unresolved-attribute]  # Trainer method
        if self.args.enable_jit_checkpoint:  # ty: ignore[unresolved-attribute]  # Trainer attr; only under JIT: shielded workers never reap
            # Accelerate's adapter proxies attribute reads, so assigning on it no-ops
            inner = getattr(dataloader, "base_dataloader", dataloader)
            if getattr(inner, "num_workers", 0) > 0:
                inner.worker_init_fn = SigtermShieldedWorkerInit(inner.worker_init_fn)
        return dataloader

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train, restoring on a re-queue and exiting on preemption or a crash.

        Both live here rather than at the call site so neither can be forgotten.
        """
        if not args and "resume_from_checkpoint" not in kwargs:
            kwargs["resume_from_checkpoint"] = resume_checkpoint(self.args)  # ty: ignore[unresolved-attribute]  # Trainer attr

        try:
            output = super().train(*args, **kwargs)  # ty: ignore[unresolved-attribute]  # Trainer method
        except Exception:  # noqa: BLE001 - breadth is the point: any death must sync
            traceback.print_exc()
            self.finish(1)

        if self.sigterm_received:
            print(
                f"Interrupted at step {self.state.global_step} of "  # ty: ignore[unresolved-attribute]  # Trainer attr
                f"{self.state.max_steps}: checkpoint saved, exiting for re-queue.",  # ty: ignore[unresolved-attribute]  # Trainer attr
                flush=True,
            )
            self.finish()
        return output

    def finish(self, exit_code: int = 0) -> NoReturn:
        """Upload the work dir, free it if that succeeded, and end the process.

        W&B is closed between the two: after the upload so its network flush cannot
        delay the checkpoint, before the removal because WANDB_DIR is inside the work
        dir. os._exit because shielded workers cannot be reaped.
        """
        if self.is_world_process_zero() and not on_cluster():  # ty: ignore[unresolved-attribute]  # Trainer method
            _finalize_wandb()
        elif self.is_world_process_zero():  # ty: ignore[unresolved-attribute]  # Trainer method
            work_dir = os.environ["JOB_WORK_DIR"]
            mirrored = mirror_to_object_storage()
            _finalize_wandb()
            if mirrored:
                shutil.rmtree(work_dir, ignore_errors=True)
                if os.path.isdir(work_dir):
                    # another rank can recreate it (bytecode cache) as we walk it
                    print(f"Note: {work_dir} not fully removed.", flush=True)
            else:
                print(
                    f"WARNING: no bucket accepted the upload. Keeping {work_dir} on "
                    f"{os.uname().nodename} -- it is the only copy and dies with the "
                    "worker pod. Recover it with `srun --nodelist=<node> cp -a ...`.",
                    flush=True,
                )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


class PreemptionSafeTrainer(PreemptionSafeMixin, Trainer):
    """Trainer with preemption safety. Mixin first, so its __init__ and train() win."""

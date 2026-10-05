"""Object-storage mirror and restore of the Slurm work dir, via `s5cmd`."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from transformers.trainer_utils import get_last_checkpoint

if TYPE_CHECKING:
    from transformers import TrainingArguments

S3_ENDPOINT = "http://cwlota.com"
# Tried in order; the second is the escape hatch when writes are suspended for one AZ.
S3_BUCKETS = ("brineylab-eu", "brineylab-us-east")


def s3_job_uri(bucket: str) -> str:
    """This job's prefix. Env is read at call time so the module imports off-cluster;
    JOB_DIR is requeue-stable, which is what makes resume work.

    Args:
        bucket: Bucket name.

    Returns:
        `s3://<bucket>/<SLURM_JOB_USER>/<JOB_DIR>`.
    """
    return f"s3://{bucket}/{os.environ['SLURM_JOB_USER']}/{os.environ['JOB_DIR']}"


def on_cluster() -> bool:
    """Whether .env has been sourced; raises in a Slurm job where it was not.

    Returns:
        True when `JOB_WORK_DIR` is set, False off-cluster.

    Raises:
        RuntimeError: Inside a Slurm job with `JOB_WORK_DIR` unset.
    """
    if "JOB_WORK_DIR" in os.environ:
        return True
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "JOB_WORK_DIR is unset inside a Slurm job: `source /mnt/home/$USER/.env` is "
            "missing from the batch script, so checkpoints would never reach object storage."
        )
    return False


def mirror_to_object_storage() -> bool:
    """Mirror the work dir to the first bucket that takes it. Never raises.

    Returns:
        True if a bucket accepted the sync (or off-cluster, where it is skipped).
    """
    if not on_cluster():
        print("not on the cluster (no JOB_WORK_DIR): skipping the object-storage sync")
        return True
    work_dir = os.environ["JOB_WORK_DIR"]
    for bucket in S3_BUCKETS:
        for attempt in (1, 2):
            completed = subprocess.run(
                # trailing slash: sync contents, not the dir itself (avoids double-nesting)
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
                    f"{work_dir}/",
                    f"{s3_job_uri(bucket)}/",
                ],
                check=False,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            if completed.returncode == 0:
                if bucket != S3_BUCKETS[0]:
                    print(f"Uploaded to {bucket} after {S3_BUCKETS[0]} failed.", flush=True)
                return True
            # a preemption kills s5cmd mid-transfer, which also exits 1 saying nothing,
            # so retry here -- sync is incremental -- rather than pay to re-upload the
            # whole work dir to another region on what may be our own death
            print(
                f"WARNING: sync to {bucket} exited {completed.returncode} "
                f"(attempt {attempt} of 2).",
                flush=True,
            )
    return False


def discard_torn_checkpoints(output_dir: str) -> None:
    """Drop checkpoints a cut-short save left behind: get_last_checkpoint ignores the
    sentinel, and a later sync cannot replace a directory that still carries one.

    Args:
        output_dir: Directory holding `checkpoint-*` subdirectories.
    """
    for sentinel in glob.glob(
        os.path.join(output_dir, "checkpoint-*", "checkpoint-is-incomplete.txt")
    ):
        incomplete = os.path.dirname(sentinel)
        print(f"Discarding incomplete checkpoint {incomplete}", flush=True)
        shutil.rmtree(incomplete, ignore_errors=True)


def restore_from_object_storage(output_dir: str) -> None:
    """Pull every bucket down: an upload that fell back lands the newest state in the
    second one, so stopping at the first can resume an interval behind.

    Args:
        output_dir: Local output dir, cleaned of torn checkpoints after each bucket.
    """
    work_dir = os.environ["JOB_WORK_DIR"]
    for bucket in S3_BUCKETS:
        job_uri = s3_job_uri(bucket)
        print(f"Restoring checkpoints from {job_uri}", flush=True)
        subprocess.run(
            ["s5cmd", "--endpoint-url", S3_ENDPOINT, "sync", f"{job_uri}/*", f"{work_dir}/"],
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        # before the next bucket syncs, so its good copy is not merged into a torn one
        discard_torn_checkpoints(output_dir)


def latest_complete_checkpoint(output_dir: str) -> str | None:
    """Latest complete checkpoint under output_dir, or None for a fresh start.

    Args:
        output_dir: Directory holding `checkpoint-*` subdirectories; may not exist.

    Returns:
        Path of the newest complete checkpoint, or None.
    """
    if not os.path.isdir(output_dir):  # get_last_checkpoint raises on a missing dir
        return None

    discard_torn_checkpoints(output_dir)
    return get_last_checkpoint(output_dir)


def resume_checkpoint(training_args: TrainingArguments) -> str | None:
    """The checkpoint a re-queued job should resume from, or None for a fresh run.

    Args:
        training_args: Arguments of the trainer about to train.

    Returns:
        Path of the checkpoint to resume from, or None.
    """
    requeued = int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
    with training_args.main_process_first(desc="checkpoint restore"):
        if requeued and training_args.process_index == 0:
            restore_from_object_storage(training_args.output_dir)  # ty: ignore[invalid-argument-type]  # HF types output_dir Optional; never None after __post_init__
        last_ckpt = latest_complete_checkpoint(training_args.output_dir)  # ty: ignore[invalid-argument-type]  # HF types output_dir Optional; never None after __post_init__

    if last_ckpt:
        print(f"Resuming from {last_ckpt}", flush=True)
    elif requeued:
        # ambiguous: nothing saved yet, or the restore failed -- s5cmd exits 0 for both
        print(
            "WARNING: re-queued but restored no checkpoint, so this run starts from "
            "step 0. If the previous attempt saved anything, the next upload will "
            f"overwrite it. Check {os.environ['JOB_DIR']} in "
            f"{' and '.join(S3_BUCKETS)} before letting it run.",
            flush=True,
        )
    return last_ckpt

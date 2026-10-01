"""Fixtures for the preemption tests: a fake `s5cmd` that mirrors buckets onto local dirs."""

from __future__ import annotations

import os
import stat
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_FAKE_S5CMD = """#!{python}
import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["FAKE_S5CMD_ROOT"])
argv = sys.argv[1:]
with open(os.environ["FAKE_S5CMD_LOG"], "a") as log:
    log.write("\\t".join(["s5cmd", *argv]) + "\\n")

args = argv[2:] if argv[:1] == ["--endpoint-url"] else argv
assert args[0] == "sync", args
excludes = []
positional = []
i = 1
while i < len(args):
    if args[i] == "--exclude":
        excludes.append(args[i + 1])
        i += 2
    elif args[i].startswith("--"):
        i += 1
    else:
        positional.append(args[i])
        i += 1
src, dst = positional


def local(path):
    if path.startswith("s3://"):
        return root / path[len("s3://"):]
    return Path(path)


counter = Path(os.environ["FAKE_S5CMD_COUNTER"])
done = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(done + 1))
codes = [c for c in os.environ.get("FAKE_S5CMD_EXITS", "").split(",") if c]
if done < len(codes) and int(codes[done]) != 0:
    sys.exit(int(codes[done]))

src_dir = local(src[:-2] if src.endswith("/*") else src.rstrip("/"))
dst_dir = local(dst.rstrip("/"))
if src_dir.is_dir():
    dst_dir.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(".*") if excludes else None
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=ignore)
"""


class FakeS5cmd:
    """Handle on the fake `s5cmd`: its call log, bucket roots and scripted exit codes."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self._monkeypatch = monkeypatch

    def calls(self) -> list[list[str]]:
        """Return the argv of every call so far, in order."""
        log = self.root / "s5cmd.log"
        if not log.exists():
            return []
        return [line.split("\t") for line in log.read_text().splitlines()]

    def bucket(self, name: str) -> Path:
        """Return the local directory standing in for bucket `name`."""
        return self.root / "buckets" / name

    def set_exits(self, codes: str) -> None:
        """Script the exit codes of upcoming calls, e.g. `"1,0"`; exhausted means 0."""
        self._monkeypatch.setenv("FAKE_S5CMD_EXITS", codes)
        (self.root / "exits.count").unlink(missing_ok=True)


@pytest.fixture
def fake_s5cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeS5cmd:
    root = tmp_path / "fake_s5cmd"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "s5cmd"
    script.write_text(_FAKE_S5CMD.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IRWXU | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_S5CMD_ROOT", str(root / "buckets"))
    monkeypatch.setenv("FAKE_S5CMD_LOG", str(root / "s5cmd.log"))
    monkeypatch.setenv("FAKE_S5CMD_COUNTER", str(root / "exits.count"))
    monkeypatch.setenv("FAKE_S5CMD_EXITS", "")
    return FakeS5cmd(root, monkeypatch)


@pytest.fixture
def cluster_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set the cluster env vars; return the work dir."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setenv("SLURM_JOB_USER", "u")
    monkeypatch.setenv("JOB_DIR", "job1")
    monkeypatch.setenv("JOB_WORK_DIR", str(work_dir))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    return work_dir


@pytest.fixture
def restore_sigterm() -> Iterator[None]:
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)

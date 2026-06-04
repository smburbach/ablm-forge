"""Import-hygiene guards for the ``ablm.data`` package.

``ablm.data`` (and every submodule) must import cleanly without pulling in
``ablm.eval`` — the eval harness depends on ``ablm.data``, not the reverse.
Checks run in a fresh subprocess so they are not masked by other tests in the
session having already imported ``ablm.eval``.
"""

from __future__ import annotations

import subprocess
import sys

_SUBMODULES = [
    "ablm.data",
    "ablm.data.tokenizer",
    "ablm.data.loaders",
]


def _import_in_subprocess(module: str) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter and report any ``ablm.eval`` leak."""
    code = (
        "import sys\n"
        f"import {module}\n"
        "assert 'ablm.eval' not in sys.modules, "
        f"'{module} transitively imported ablm.eval'\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_data_package_imports_without_eval() -> None:
    """``import ablm.data`` succeeds and does not import ``ablm.eval``."""
    result = _import_in_subprocess("ablm.data")
    assert result.returncode == 0, result.stderr


def test_submodules_import_without_eval() -> None:
    """Every ``ablm.data`` submodule imports cleanly and eval-free."""
    for module in _SUBMODULES:
        result = _import_in_subprocess(module)
        assert result.returncode == 0, f"{module}: {result.stderr}"

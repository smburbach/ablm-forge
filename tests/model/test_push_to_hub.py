"""Custom-code copy + remote-loading round-trip for the ABLM auto-class hook.

These exercise the ``register_for_auto_class`` file-copy step: saving a
model must drop the source ``.py`` files next to ``config.json`` so that a fresh
interpreter can reload it via ``trust_remote_code=True`` without ``ablm``
installed in the registry.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import ablm  # noqa: F401  (import triggers auto-class + custom-code registration)
from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def saved_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a tiny model + tokenizer and return the directory."""
    tmpdir = tmp_path_factory.mktemp("ablm_remote")
    config = AblmConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    AblmForMaskedLM(config).save_pretrained(tmpdir)
    AblmTokenizerFast().save_pretrained(tmpdir)
    return tmpdir


def test_custom_code_files_copied(saved_model_dir: Path) -> None:
    """The public source files plus their helper imports land beside config.json."""
    names = {p.name for p in saved_model_dir.iterdir()}

    # Public custom-code modules HF copies via register_for_auto_class.
    assert "modeling_ablm.py" in names
    assert "configuration_ablm.py" in names
    assert "tokenization_ablm.py" in names

    # Helper modules imported by modeling_ablm.py must come along too, or the
    # remote import fails.
    for helper in ("attention.py", "transformer.py", "norm.py", "rope.py", "ffn.py"):
        assert helper in names, f"missing helper module {helper}"

    # auto_map entries must be present in the written config.
    config_text = (saved_model_dir / "config.json").read_text()
    assert "auto_map" in config_text
    assert "modeling_ablm.AblmForMaskedLM" in config_text


@pytest.mark.slow
def test_remote_reload_in_subprocess(saved_model_dir: Path) -> None:
    """A fresh interpreter reloads the model via trust_remote_code (no `import ablm`).

    Run in a subprocess so the in-process auto-class registration from
    ``import ablm`` cannot mask a regression in the custom-code copy step.
    """
    script = (
        "from transformers import AutoModelForMaskedLM, AutoTokenizer; "
        f"m = AutoModelForMaskedLM.from_pretrained({str(saved_model_dir)!r}, "
        "trust_remote_code=True); "
        f"t = AutoTokenizer.from_pretrained({str(saved_model_dir)!r}, "
        "trust_remote_code=True); "
        "assert type(m).__name__ == 'AblmForMaskedLM'; "
        "assert t('MEEPQ').input_ids == [0, 20, 9, 9, 14, 16, 2]"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

"""Save/load round-trip through the in-process Auto* registration (BALM-style).

`import ablm` registers the ABLM classes with the Auto* factories, so a saved
checkpoint reloads via `AutoModel*.from_pretrained` **with the package installed**
— no `trust_remote_code`, no `auto_map`, no copied source files. These tests lock
that contract: the round-trip works, and no custom-code artifacts are written
(which is what previously forced the model package to stay flat).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import ablm  # noqa: F401  (import triggers Auto* registration)
from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def saved_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a tiny model + tokenizer and return the directory."""
    tmpdir = tmp_path_factory.mktemp("ablm_register")
    config = AblmConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    AblmForMaskedLM(config).save_pretrained(tmpdir)
    AblmTokenizerFast().save_pretrained(tmpdir)
    return tmpdir


def test_reload_via_auto_class(saved_model_dir: Path) -> None:
    """AutoModelForMaskedLM resolves the registered ABLM class and runs a forward."""
    model = AutoModelForMaskedLM.from_pretrained(saved_model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
    assert type(model).__name__ == "AblmForMaskedLM"
    assert isinstance(model, AblmForMaskedLM)

    batch = tokenizer(["MEEPQ"], return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    assert out.logits.shape == (1, batch.input_ids.shape[1], model.config.vocab_size)


def test_no_custom_code_artifacts_written(saved_model_dir: Path) -> None:
    """Register-only loading must NOT copy source files or write an auto_map.

    Their presence would mean the trust_remote_code file-copy path is back — the
    thing that forces the model package flat. Their absence keeps `layers/` legal.
    """
    names = {p.name for p in saved_model_dir.iterdir()}
    for src in ("modeling_ablm.py", "configuration_ablm.py", "tokenization_ablm.py"):
        assert src not in names, f"unexpected copied source file {src}"

    config_text = (saved_model_dir / "config.json").read_text()
    assert "auto_map" not in config_text

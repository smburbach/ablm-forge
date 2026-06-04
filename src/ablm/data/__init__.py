"""ABLM data tooling.

`build_train_dataset` streams/tokenizes/mixes parquet sources via 🤗 `datasets`;
pad/mask its output with `transformers.DataCollatorForLanguageModeling` in your
training script. `get_tokenizer` returns the shared ESM-C tokenizer.
"""

from __future__ import annotations

from ablm.data.loaders import build_train_dataset
from ablm.data.tokenizer import get_tokenizer

__all__ = [
    "build_train_dataset",
    "get_tokenizer",
]

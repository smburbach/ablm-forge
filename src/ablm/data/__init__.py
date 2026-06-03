"""ABLM data tooling.

Streaming MLM data built on 🤗 `datasets` + `transformers`: `build_train_dataset`
streams/tokenizes/mixes parquet sources, `build_collator` is the stock
masked-LM collator. The public surface re-exported below feeds the HF Trainer.
"""

from __future__ import annotations

from ablm.data.config import parse_train_configs
from ablm.data.loaders import build_collator, build_train_dataset
from ablm.data.tokenizer import get_tokenizer

__all__ = [
    "build_collator",
    "build_train_dataset",
    "get_tokenizer",
    "parse_train_configs",
]

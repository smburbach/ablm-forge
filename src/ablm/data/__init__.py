"""ABLM data tooling.

`build_train_dataset` streams/tokenizes/mixes parquet sources via 🤗 `datasets`;
pad/mask its output with `transformers.DataCollatorForLanguageModeling` (with an
`ablm.AblmTokenizerFast`) in your training script.
"""

from __future__ import annotations

from ablm.data.loaders import build_train_dataset

__all__ = ["build_train_dataset"]

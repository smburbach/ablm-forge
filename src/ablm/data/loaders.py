"""Streaming MLM training data, built on 🤗 `datasets`.

`build_train_dataset` streams protein sequences from parquet via
`datasets.load_dataset(streaming=True)`, tokenizes them with a `.map`, optionally
mixes several sources with `interleave_datasets`, shuffles with a buffer, and
shards per node. Distributed and DataLoader-worker sharding are handled by
`datasets` + the HF Trainer.

Pair the result with `transformers.DataCollatorForLanguageModeling` (masks +
pads) in your training script — see `scripts/pretrain.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node

from ablm.model import AblmTokenizerFast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import IterableDataset

_PARQUET_GLOB = "*.parquet"


def _data_files(path: str) -> str:
    """Resolve a parquet file or a directory of shards to a `data_files` value."""
    p = Path(path)
    return str(p / _PARQUET_GLOB) if p.is_dir() else str(p)


def build_train_dataset(
    train: str | Sequence[str],
    *,
    max_length: int,
    seed: int,
    shuffle_buffer_size: int = 10_000,
    probabilities: Sequence[float] | None = None,
) -> IterableDataset:
    """Build the streaming, tokenized MLM training dataset.

    Args:
        train: A parquet path/dir, or a sequence of them to mix.
        max_length: Tokenization truncation length.
        seed: Shuffle / interleave seed.
        shuffle_buffer_size: Buffer for `IterableDataset.shuffle`.
        probabilities: Per-source sampling weights when `train` is a sequence
            (normalized to sum to 1; defaults to equal weights). Ignored for a
            single source.

    The result is shuffled with a buffer and sharded across distributed ranks
    (`RANK` / `WORLD_SIZE`, set by torchrun); DataLoader workers are sharded by
    `datasets` automatically. Yields tokenized examples (`input_ids`,
    `attention_mask`, `special_tokens_mask`); pad/mask them with
    `transformers.DataCollatorForLanguageModeling`.
    """
    paths = [train] if isinstance(train, str) else list(train)
    if not paths:
        raise ValueError("No training data: pass a parquet path or a sequence of paths.")

    tokenizer = AblmTokenizerFast()

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    parts: list[IterableDataset] = []
    for path in paths:
        ds = load_dataset("parquet", data_files=_data_files(path), split="train", streaming=True)
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        parts.append(ds)

    if len(parts) == 1:
        dataset = parts[0]
    else:
        if probabilities is None:
            probabilities = [1.0 / len(parts)] * len(parts)
        elif len(probabilities) != len(parts):
            raise ValueError(
                f"probabilities has {len(probabilities)} entries but there are "
                f"{len(parts)} sources."
            )
        total = float(sum(probabilities))
        if total <= 0:
            raise ValueError(f"probabilities must sum to > 0, got {total}.")
        weights = [p / total for p in probabilities]
        dataset = interleave_datasets(
            parts, probabilities=weights, seed=seed, stopping_strategy="all_exhausted"
        )

    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dataset = split_dataset_by_node(
            dataset, rank=int(os.environ.get("RANK", "0")), world_size=world_size
        )
    return dataset

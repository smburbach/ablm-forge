"""Streaming MLM data pipeline, built on 🤗 `datasets` + `transformers`.

`build_train_dataset` streams protein sequences from parquet via
`datasets.load_dataset(streaming=True)`, tokenizes them with a `.map`, mixes
multiple sources with `interleave_datasets`, shuffles with a buffer, and shards
per node. `build_collator` returns the stock
`transformers.DataCollatorForLanguageModeling`, which applies BERT-style
masked-LM corruption (the scheme ESM-2 / ESM-C use). Distributed and DataLoader
worker sharding are handled by `datasets` + the HF Trainer — no custom striping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import DataCollatorForLanguageModeling

from ablm.data.config import parse_train_configs
from ablm.data.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from datasets import IterableDataset

    from ablm.config import DataConfig

_PARQUET_GLOB = "*.parquet"


def _data_files(path: str) -> str:
    """Resolve a parquet file or a directory of shards to a `data_files` value."""
    p = Path(path)
    return str(p / _PARQUET_GLOB) if p.is_dir() else str(p)


def build_train_dataset(data: DataConfig, *, max_length: int, seed: int) -> IterableDataset:
    """Build the streaming, tokenized MLM training dataset from `data.train`.

    A single path streams one source; a `{name: {path, fraction}}` mapping mixes
    several via `interleave_datasets` weighted by the (normalized) fractions. The
    result is shuffled with a buffer and sharded across distributed ranks
    (`RANK` / `WORLD_SIZE`, set by torchrun); DataLoader workers are sharded by
    `datasets` automatically.

    Yields tokenized examples (`input_ids`, `attention_mask`, `special_tokens_mask`);
    the collator from `build_collator` masks and pads them.
    """
    entries = parse_train_configs(data.train)
    if not entries:
        raise ValueError(
            "No training data configured. Set `data.train` to a parquet path or a "
            "{name: {path, fraction}} mapping."
        )

    tokenizer = get_tokenizer()

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    parts: list[IterableDataset] = []
    for entry in entries:
        ds = load_dataset(
            "parquet", data_files=_data_files(entry.path), split="train", streaming=True
        )
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        parts.append(ds)

    if len(parts) == 1:
        dataset = parts[0]
    else:
        dataset = interleave_datasets(
            parts,
            probabilities=[entry.fraction for entry in entries],
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    dataset = dataset.shuffle(seed=seed, buffer_size=data.shuffle_buffer_size)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dataset = split_dataset_by_node(
            dataset, rank=int(os.environ.get("RANK", "0")), world_size=world_size
        )
    return dataset


def build_collator(data: DataConfig) -> DataCollatorForLanguageModeling:
    """Build the masked-LM collator (BERT 80/10/10 corruption, dynamic padding)."""
    return DataCollatorForLanguageModeling(
        tokenizer=get_tokenizer(),
        mlm=True,
        mlm_probability=data.mask_prob,
        mask_replace_prob=data.mask_token_prob,
        random_replace_prob=data.random_token_prob,
    )

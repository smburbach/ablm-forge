"""ABLM data tooling.

Tokenization access, collation, and the dataset builders for MLM pretraining.
The public surface re-exported below feeds the HuggingFace Trainer.
"""

from __future__ import annotations

from ablm.data.config import parse_train_configs
from ablm.data.sequence.collate import MLMCollator, tokenize_and_pad
from ablm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from ablm.data.tokenizer import get_tokenizer

__all__ = [
    "InterleavedDataset",
    "MLMCollator",
    "ShardedProteinDataset",
    "get_tokenizer",
    "parse_train_configs",
    "tokenize_and_pad",
]

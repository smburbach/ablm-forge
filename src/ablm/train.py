"""Training entry point: build the model + data + stock `transformers.Trainer`.

Launch with torchrun (FSDP2 config lives in the YAML / TrainingArguments):

    torchrun --standalone --nproc_per_node=8 -m ablm.train --config run.yaml

No `Trainer` subclass: HF-native optimizers are selected via
`TrainingArguments.optim`; custom optimizers (e.g. Muon) are built from the
registry and handed to the stock Trainer through its `optimizers=` tuple.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from ablm.config import (
    DataConfig,
    build_training_arguments,
    load_config,
    optimizer_settings,
)
from ablm.data.config import parse_train_configs
from ablm.data.sequence.collate import MLMCollator
from ablm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from ablm.data.tokenizer import get_tokenizer
from ablm.training.optim_registry import build_optimizer, resolve_optimizer

if TYPE_CHECKING:
    from torch.utils.data import Dataset, IterableDataset
    from transformers import Trainer

    from ablm.config import AblmRunConfig

logger = logging.getLogger("ablm.train")

# Eval-dataset types this base repo can build as MLM eval sets. Other types
# (structure, proteingym, ...) belong to a downstream eval harness, out of scope.
_MLM_EVAL_TYPES = frozenset({"sequence"})


def build_train_dataset(data: DataConfig, *, seed: int) -> IterableDataset:
    """Build the training `IterableDataset` from `data.train` (single or interleaved)."""
    entries = parse_train_configs(data.train)
    if not entries:
        raise ValueError(
            "No training data configured. Set `data.train` to a parquet path or a "
            "{name: {path, fraction}} mapping."
        )
    datasets = [
        ShardedProteinDataset(
            entry.path,
            shuffle_shards=data.shuffle_shards,
            shuffle_rows=data.shuffle_rows,
            seed=seed,
            load_masking_weights=data.weighted_masking,
        )
        for entry in entries
    ]
    if len(datasets) == 1:
        return datasets[0]
    fractions = [entry.fraction for entry in entries]
    return InterleavedDataset(datasets, fractions, seed=seed)


def build_eval_datasets(data: DataConfig, *, seed: int) -> dict[str, Dataset] | None:
    """Build a name -> dataset map for sequence-type eval datasets, if any."""
    if not data.eval or not isinstance(data.eval, dict):
        return None
    eval_sets: dict[str, Dataset] = {}
    for name, spec in data.eval.items():
        if spec is None:
            continue
        eval_type = spec.get("type") if isinstance(spec, dict) else None
        path = spec.get("path") if isinstance(spec, dict) else spec
        if eval_type not in _MLM_EVAL_TYPES:
            logger.warning("Skipping eval dataset %r of unsupported type %r.", name, eval_type)
            continue
        eval_sets[str(name)] = ShardedProteinDataset(
            path, shuffle_shards=False, shuffle_rows=False, seed=seed
        )
    return eval_sets or None


def build_collator(data: DataConfig, max_length: int, *, deterministic: bool) -> MLMCollator:
    """Build an `MLMCollator` from the data config."""
    return MLMCollator(
        get_tokenizer(),
        max_length=max_length,
        mask_prob=data.mask_prob,
        mask_token_prob=data.mask_token_prob,
        random_token_prob=data.random_token_prob,
        weighted_masking=data.weighted_masking,
        deterministic=deterministic,
    )


def build_trainer(cfg: AblmRunConfig) -> Trainer:
    """Assemble a stock `transformers.Trainer` from a resolved `AblmRunConfig`."""
    from transformers import Trainer

    from ablm.model import AblmForMaskedLM

    model = AblmForMaskedLM(cfg.model)

    training_args = build_training_arguments(cfg.train, cfg.data)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    max_length = cfg.model.max_position_embeddings
    train_dataset = build_train_dataset(cfg.data, seed=cfg.train.seed)
    eval_datasets = build_eval_datasets(cfg.data, seed=cfg.train.seed)
    train_collator = build_collator(cfg.data, max_length, deterministic=False)

    # Custom optimizers (Muon) are built here and handed to the stock Trainer;
    # HF-native optimizers are selected purely via training_args.optim.
    spec = resolve_optimizer(cfg.train.optimizer)
    optimizers = (None, None)
    if spec.is_custom:
        optimizer = build_optimizer(cfg.train.optimizer, model, optimizer_settings(cfg.train))
        optimizers = (optimizer, None)  # Trainer creates the scheduler from args

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=train_collator,
        optimizers=optimizers,
    )


def main(argv: list[str] | None = None) -> None:
    """Load config, build the trainer, train, and save the final model."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(argv if argv is not None else sys.argv[1:])
    if cfg.train.wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", cfg.train.wandb_project)

    trainer = build_trainer(cfg)
    trainer.train(resume_from_checkpoint=cfg.train.resume_from)
    trainer.save_model()


if __name__ == "__main__":
    main()

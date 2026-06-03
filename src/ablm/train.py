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

from ablm.config import build_training_arguments, load_config, optimizer_settings
from ablm.data.loaders import build_collator, build_train_dataset
from ablm.training.optim import build_optimizer, resolve_optimizer

if TYPE_CHECKING:
    from transformers import Trainer

    from ablm.config import AblmRunConfig


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
    train_dataset = build_train_dataset(cfg.data, max_length=max_length, seed=cfg.train.seed)
    train_collator = build_collator(cfg.data)

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
        # datasets.IterableDataset is accepted at runtime; not in the HF stub union.
        train_dataset=train_dataset,  # ty: ignore[invalid-argument-type]
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

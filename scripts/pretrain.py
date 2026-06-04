"""Example MLM pretraining script for ablm-forge.

ablm-forge is a library, not a framework: there is no config system, CLI, or data
module. You compose the building blocks — `AblmConfig`, a 🤗 `datasets` stream, a
`DataCollatorForLanguageModeling`, an optimizer, and the stock
`transformers.Trainer` — in a script like this one and launch it. Copy and edit
it for your runs.

    # single GPU
    python scripts/pretrain.py --data /data/train.parquet --output-dir out

    # multi-GPU + FSDP2
    torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
        --data /data/train/ --output-dir out --fsdp --bf16 --gradient-checkpointing

`--data` is a parquet file/dir with `sequence_id` + `sequence` columns (shard
into multiple parquet files for `--num-workers > 1`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast
from ablm.training.optim import OptimizerSettings, build_muon_optimizer

# HF-native optimizers are just TrainingArguments.optim strings.
_HF_OPTIM = {"adamw": "adamw_torch", "adamw_fused": "adamw_torch_fused", "adafactor": "adafactor"}


def build_dataset(data: str, *, max_length: int, seed: int, shuffle_buffer: int):
    """Stream + tokenize parquet into an MLM `datasets.IterableDataset`.

    Single-source and single-node. Edit to add `datasets.interleave_datasets` for
    mixing. To scale to multiple processes/nodes, shard per rank with
    `datasets.distributed.split_dataset_by_node` (and set the Trainer's
    `accelerator_config={"dispatch_batches": False}`).
    """
    data_files = f"{data}/*.parquet" if Path(data).is_dir() else data
    ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    tokenizer = AblmTokenizerFast()
    ds = ds.map(
        lambda b: tokenizer(
            b["sequence"], truncation=True, max_length=max_length, return_special_tokens_mask=True
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds.shuffle(seed=seed, buffer_size=shuffle_buffer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="parquet file or directory of shards")
    p.add_argument("--output-dir", required=True)
    # Model (defaults ~ ESM-C 600M-ish; shrink for quick runs)
    p.add_argument("--hidden-size", type=int, default=1152)
    p.add_argument("--num-layers", type=int, default=36)
    p.add_argument("--num-heads", type=int, default=18)
    p.add_argument("--max-length", type=int, default=1024)
    # Training
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--optimizer", choices=[*_HF_OPTIM, "muon"], default="adamw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=10_000)
    # Hardware / memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fsdp", action="store_true", help="enable FSDP2 full_shard")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = AblmForMaskedLM(
        AblmConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            max_position_embeddings=args.max_length,
        )
    )

    dataset = build_dataset(
        args.data, max_length=args.max_length, seed=args.seed, shuffle_buffer=args.shuffle_buffer
    )
    collator = DataCollatorForLanguageModeling(tokenizer=AblmTokenizerFast(), mlm=True)

    # FSDP2: shard on the AblmBlock; route activation checkpointing into fsdp_config
    # (the Trainer's gradient_checkpointing adds a redundant all-gather under FSDP).
    fsdp = ""
    fsdp_config = None
    grad_ckpt_arg = args.gradient_checkpointing
    if args.fsdp:
        fsdp = "full_shard auto_wrap"
        fsdp_config = {
            "fsdp_version": 2,
            "transformer_layer_cls_to_wrap": ["AblmBlock"],
            "activation_checkpointing": args.gradient_checkpointing,
        }
        grad_ckpt_arg = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim=_HF_OPTIM.get(args.optimizer, "adamw_torch"),
        bf16=args.bf16,
        gradient_checkpointing=grad_ckpt_arg,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        dataloader_num_workers=args.num_workers,
        save_steps=args.save_steps,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    # HF-native optimizers come from training_args.optim; Muon is built here and
    # passed through the Trainer's optimizers= tuple (scheduler stays from args).
    optimizers = (None, None)
    if args.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model, OptimizerSettings(lr=args.lr, weight_decay=args.weight_decay)
        )
        optimizers = (optimizer, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=optimizers,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

# ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder wired to the stock HuggingFace
`Trainer`, launched via `torchrun` + FSDP2, with SDPA-based attention and an
optional Muon optimizer.

It's a **library, not a framework**: there is no config system or CLI. You
compose the building blocks (`AblmConfig`, `build_train_dataset`, a
`DataCollatorForLanguageModeling`, an optimizer, `transformers.Trainer`) in a
training script. `scripts/pretrain.py` is a complete, copy-and-edit example.

## Reference architecture

Defaults track **ESM-C** (EvolutionaryScale Cambrian): Pre-LN, full RoPE,
SwiGLU, bias-free linear layers + layer norms, no QK-norm, no residual scaling,
no token dropout, and the bit-for-bit ESM-C 33-token tokenizer. Everything beyond
ESM-C (`qk_norm`, `residual_scaling`, `norm_strategy`, partial RoPE, ESM-2-style
`token_dropout`) is an opt-in `AblmConfig` knob. ESM-C sizes are head_dim-64 at
30L/960, 36L/1152, 80L/2560 (300M / 600M / 6B).

## Install

```bash
uv venv && uv pip install -e ".[dev,train]"
```

## Train

Edit `scripts/pretrain.py` (or write your own), then:

```bash
# single GPU
python scripts/pretrain.py --data /data/train.parquet --output-dir out
# multi-GPU + FSDP2
torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
    --data /data/train/ --output-dir out --fsdp --bf16 --gradient-checkpointing
```

`--data` is a parquet file or directory of shards with `sequence_id` + `sequence`
columns (shard into multiple parquet files for `--num-workers > 1`).

A minimal script is just:

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast
from ablm.data import build_train_dataset

model = AblmForMaskedLM(AblmConfig())          # architecture knobs here
ds = build_train_dataset("train.parquet", max_length=1024, seed=42)
collator = DataCollatorForLanguageModeling(tokenizer=AblmTokenizerFast(), mlm=True)
args = TrainingArguments(output_dir="out", max_steps=100_000, optim="adamw_torch", bf16=True)
Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
```

## Optimizers, schedulers, attention

- **Attention** — just `F.scaled_dot_product_attention`, which auto-selects the
  fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime. A
  manual fp32-softmax path runs only when you request `output_attentions=True`.
  Nothing to configure.
- **Optimizer** — HF-native ones are `TrainingArguments(optim="adamw_torch" | …)`.
  Muon (2D-hidden Muon + AdamW for the rest) is built with
  `ablm.training.optim.build_muon_optimizer(model, OptimizerSettings(...))` and
  passed via `Trainer(..., optimizers=(opt, None))`. No `Trainer` subclass.
- **LR schedule** — `TrainingArguments.lr_scheduler_type` (`linear`, `cosine`,
  `cosine_with_min_lr`, `warmup_stable_decay`, …).

> Note: Muon's Newton-Schulz step assumes full 2D weights; under FSDP2 sharding
> it is mathematically approximate. Validate Muon under single-GPU / DDP first.

## Layout

- `src/ablm/model/` — the encoder, heads, and `AblmConfig`, registered with the
  HuggingFace Auto* classes. Attention is SDPA + a manual-softmax fallback.
- `src/ablm/data/` — 🤗 `datasets` streaming loader (`build_train_dataset`);
  pad/mask with HF `DataCollatorForLanguageModeling` + `AblmTokenizerFast`.
- `src/ablm/training/optim.py` — Muon `CombinedOptimizer` + `build_muon_optimizer`.
- `scripts/pretrain.py` — example training entry point (torchrun-launchable).

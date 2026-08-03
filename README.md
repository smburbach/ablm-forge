# ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder wired to the stock HuggingFace
`Trainer`, launched via `accelerate --multi_gpu` (DDP), with SDPA-based attention
and Muon as the recommended production optimizer.

It's a **library, not a framework**: no config system, CLI, or data module. You
compose the building blocks (`AblmConfig`, a 🤗 `datasets` stream, a
`DataCollatorForLanguageModeling`, an optimizer, `transformers.Trainer`) in a
training script. `scripts/pretrain.py` is a complete, copy-and-edit example.

## Reference architecture

Defaults track **ESM-C** (EvolutionaryScale Cambrian): Pre-LN, full RoPE,
SwiGLU, **bias-free** linear layers and layer norms (`norm_bias=false`,
`ffn_bias=false`, `attention_bias=false`), no QK-norm, no residual scaling,
no token dropout, and the bit-for-bit ESM-C 33-token tokenizer. Everything beyond
ESM-C (`qk_norm`, `residual_scaling`, `norm_strategy`, partial RoPE,
`token_dropout`, `attention_bias`) is an opt-in `AblmConfig` knob. ESM-C sizes
are head_dim-64 at 30L/960, 36L/1152, 80L/2560 (300M / 600M / 6B). FFN variants
are `swiglu` / `geglu` / `reglu` (gated) and `gelu_mlp` (non-gated, ESM-2 style),
selected via `ffn_activation`.

## Install

```bash
uv venv && uv pip install -e ".[dev,train]"
```

## Train

Edit `scripts/pretrain.py` (or write your own), then:

```bash
# single GPU
python scripts/pretrain.py --data /data/train.parquet --output-dir out
# multi-GPU (DDP)
accelerate launch --multi_gpu --mixed_precision bf16 scripts/pretrain.py \
    --data /data/train/ --output-dir out --bf16 --gradient-checkpointing
```

`--data` is a parquet file or directory of shards with `sequence_id` + `sequence`
columns (shard into multiple parquet files for `--num-workers > 1`).

A minimal script is just:

```python
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast

tok = AblmTokenizerFast()
ds = load_dataset("parquet", data_files="train.parquet", split="train", streaming=True)
ds = ds.map(
    lambda b: tok(b["sequence"], truncation=True, max_length=1024, return_special_tokens_mask=True),
    batched=True, remove_columns=ds.column_names,
).shuffle(seed=42, buffer_size=10_000)

model = AblmForMaskedLM(AblmConfig())          # architecture knobs here
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True)
args = TrainingArguments(output_dir="out", max_steps=100_000, optim="adamw_torch", bf16=True)
Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
```

## Optimizers, schedulers, attention

- **Attention** — just `F.scaled_dot_product_attention`, which auto-selects the
  fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime. A
  manual fp32-softmax path runs only when you request `output_attentions=True`.
  Nothing to configure.
- **Optimizer** — HF-native ones are `TrainingArguments(optim="adamw_torch" | …)`.
  Muon (2D-hidden `DistributedMuon` + AdamW for the rest) is built with
  `ablm.training.optim.build_muon_optimizer(model, ...)` and handed to the stock
  `Trainer` via `optimizers=(opt, None)`.

  Muon is the recommended optimizer for production runs: on a 350M AbLM it reached
  lower eval loss than AdamW reproducibly (the largest single architectural effect
  measured, -0.0058 eval/loss) and is LR-robust where AdamW degrades above ~1e-4.
  AdamW remains the default for iteration.
- **LR schedule** — `TrainingArguments.lr_scheduler_type` (`linear`, `cosine`,
  `cosine_with_min_lr`, `warmup_stable_decay`, …).

> Note: `DistributedMuon` shards the Newton-Schulz orthogonalization across the
> **DDP** group (numerically identical to `torch.optim.Muon`, less redundant
> compute); on a single GPU it degrades to plain Muon. It assumes replicated
> (DDP) params and is not FSDP-aware — Newton-Schulz can't orthogonalize a shard,
> so any sharded (FSDP) setup uses AdamW.

## Layout

- `src/ablm/model/` — the encoder, heads, `AblmConfig`, and `AblmTokenizerFast`,
  registered with the HuggingFace Auto* classes. Attention is SDPA + a
  manual-softmax fallback.
- `src/ablm/training/optim.py` — `DistributedMuon`, `CombinedOptimizer`, and
  `build_muon_optimizer`.
- `scripts/pretrain.py` — example training script (data loading + Trainer wiring),
  launchable single-GPU or under `accelerate --multi_gpu`.

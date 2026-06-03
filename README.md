# ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder wired to
the stock HuggingFace `Trainer`, launched via `torchrun` + FSDP2, with
SDPA-based attention and a pluggable optimizer registry.

> Status: under construction. See `docs`/the plan for the build roadmap.

## Reference architecture

Defaults track **ESM-C** (EvolutionaryScale Cambrian): Pre-LN, full RoPE,
SwiGLU, bias-free linear layers + layer norms, no QK-norm, no residual scaling,
no token dropout, and the bit-for-bit ESM-C 33-token tokenizer. Exact-size
presets: `esmc_300m`, `esmc_600m`, `esmc_6b`. Everything beyond ESM-C
(`qk_norm`, `residual_scaling`, `norm_strategy`, partial RoPE, Canon convs, and
ESM-2-style `token_dropout`) is an opt-in experiment knob.

## Install

```bash
uv venv && uv pip install -e ".[dev,train]"
```

## Train

```bash
# single GPU
ablm train --config run.yaml
# CLI dotlist + preset overrides
ablm train --preset 170M model.num_hidden_layers=24 train.lr=2e-4
# multi-GPU + FSDP2 (set train.fsdp: "full_shard auto_wrap" in the YAML)
torchrun --standalone --nproc_per_node=8 -m ablm.train --config run.yaml
# inspect what's registered
ablm info
```

## Optimizers, schedulers, attention

- **Attention** — just `F.scaled_dot_product_attention`, which auto-selects the
  fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime. A
  manual fp32-softmax path runs only when you request `output_attentions=True`
  (SDPA can't return attention weights). Nothing to configure.
- **Optimizer** — set `train.optimizer` to `adamw`, `adamw_fused`, `adafactor`,
  or `muon`. Add one by adding an entry to the `OPTIMIZERS` dict in
  `ablm/training/optim.py` — either an HF `optim` string or a
  `builder(model, settings)` for an optimizer HF doesn't ship. No `Trainer`
  subclass is involved.
- **LR schedule** — `train.scheduler` ∈ `warmup_linear`, `warmup_cosine`,
  `wsd_linear`, `wsd_cosine`, mapped onto HF's native `lr_scheduler_type`.

> Note: Muon's Newton-Schulz step assumes full 2D weights; under FSDP2 sharding
> it is mathematically approximate. Validate Muon under single-GPU / DDP first.

## Layout

- `src/ablm/model/` — the encoder, heads, and config (`AblmConfig`,
  `AblmForMaskedLM`, …), registered with the HuggingFace Auto* classes.
- `src/ablm/model/attention.py` — SDPA attention (+ a manual-softmax fallback
  for `output_attentions`).
- `src/ablm/training/` — optimizer registry (no `Trainer` subclass).
- `src/ablm/data/` — tokenizer + MLM dataset/collator.
- `src/ablm/train.py` — `torchrun -m ablm.train --config <yaml>` entry point.

# Agent Instructions for ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder wired to the stock HuggingFace
`Trainer`, launched via `torchrun` + FSDP2, with SDPA-based attention and an
optional Muon optimizer. It is a **library, not a framework**: no config system,
no CLI — you compose the pieces in a training script (`scripts/pretrain.py` is
the example).

This file is the single source of truth for agent and contributor instructions.
Make all future updates here, not in `CLAUDE.md` (which points back to this file).

## Build & Test Commands

```bash
# install (editable, with dev + train extras)
uv pip install -e ".[dev,train]"

# run all tests / fast only / with coverage
pytest
pytest -m "not slow"
pytest --cov=ablm

# lint / format / type check
ruff check src/
ruff format src/
ty check src/                          # ty (Astral), not mypy; must be clean
```

> **Type checking uses `ty`, not mypy.** Framework-boundary diagnostics are
> suppressed inline with documented `# ty: ignore[<rule>]` comments.

## Reference architecture

The default config tracks **ESM-C** (EvolutionaryScale Cambrian): Pre-LN, full
RoPE, SwiGLU, **bias-free** linear layers and layer norms (`norm_bias=false`,
`ffn_bias=false`), **no QK-norm**, **no residual scaling**, and **no token
dropout** (`token_dropout=false` — ESM-2 had it; ESM-C removed it as redundant
under Pre-LN. The ESM-2 behavior is implemented and available via
`token_dropout=true`). ESM-C sizes are head_dim-64 at 30L/960, 36L/1152,
80L/2560 (300M / 600M / 6B) — set them directly on `AblmConfig`. The tokenizer is
bit-for-bit ESM-C (33-token vocab). Tests `tests/model/test_esm_alignment.py` pin
this alignment — keep them green when touching defaults. The architecture is a
superset: `qk_norm`,
`residual_scaling`, `norm_strategy`, partial RoPE, and `token_dropout` are opt-in
knobs for experiments. (Exact ESM-2 parity would additionally need a plain GELU
MLP FFN, which is not yet implemented — only SwiGLU is.)

> Attention is just `F.scaled_dot_product_attention`, which auto-selects the
> fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime — no
> kernel registry, no torch.compile needed. A manual fp32-softmax path runs only
> for `output_attentions=True` (SDPA can't return weights).

## Core design rules (do not violate)

- **No config system, no CLI.** Configuration lives in Python: `AblmConfig`
  (a `PretrainedConfig`) for the model, `transformers.TrainingArguments` for
  training, composed in a script (`scripts/pretrain.py`). Don't add OmegaConf /
  YAML config trees / a `train` CLI / presets.
- **No `Trainer` subclass, no custom trainer loop.** Use stock
  `transformers.Trainer` directly. HF-native optimizers via
  `TrainingArguments.optim`; Muon via `build_muon_optimizer` + the `optimizers=`
  tuple. Schedules via `lr_scheduler_type`.
- **Attention is SDPA + a manual fallback** in `ablm/model/attention.py`. Don't
  reintroduce a kernel registry / explicit flash-attn integration: SDPA already
  auto-selects the fused backend. (Keep attention in one file, not a subpackage —
  HF copies only depth-1 relative imports for `trust_remote_code`.)
- **All public model classes in `modeling_ablm.py`** for the same
  `trust_remote_code` reason; internal blocks live in their own modules.
- **MoE is out of scope.** Do not reintroduce it.

## Architecture

- **src layout**: all package code under `src/ablm/`.
- **Build system**: hatchling (`pyproject.toml` only).
- **Testing**: pytest in `tests/`.

## Project Structure

```
src/ablm/
├── model/
│   ├── outputs.py norm.py masking.py rope.py embedding.py ffn.py
│   ├── attention.py            # AblmAttention: SDPA + manual-softmax fallback
│   ├── transformer.py          # AblmBlock + AblmStack (FSDP wrap unit: AblmBlock)
│   ├── configuration_ablm.py   # AblmConfig
│   ├── tokenization_ablm.py    # AblmTokenizerFast (33-token ESM-C vocab)
│   └── modeling_ablm.py        # all public Ablm* model classes
└── training/
    └── optim.py                # Muon CombinedOptimizer + build_muon_optimizer
scripts/pretrain.py             # example training script: data loading + Trainer wiring
tests/                          # pytest, mirrors src/
```

Data loading (stream parquet via 🤗 `datasets` + tokenize + shuffle) is *not* in
the package — it's a handful of standard `datasets` calls in the training script
(`scripts/pretrain.py`), so each run owns and can edit it. It's single-node;
`split_dataset_by_node` would be added there only when scaling to multiple
processes/nodes.

## Launching training

There's no entry point in the package — copy/edit `scripts/pretrain.py`:

```bash
# single GPU
python scripts/pretrain.py --data train.parquet --output-dir out
# multi-GPU + FSDP2
torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
    --data train/ --output-dir out --fsdp --bf16 --gradient-checkpointing
```

## Code Style

- Python 3.11+, modern typing (`X | Y`, `Self`), `from __future__ import annotations`.
- Type hints on all signatures; Google-style docstrings on public APIs.
- Ruff lint + format; line length 100. No wildcard imports, no bare `except:`.

## Testing

- Mirror source layout. `pytest.fixture` for setup, `@pytest.mark.parametrize`
  for input variation, `@pytest.mark.slow` for multi-step / GPU runs.
- Prefer real data (the parquet fixture under `tests/fixtures/training/`).
- The pilot suite (`tests/training/test_pilot_train.py`,
  `test_pilot_fsdp.py`) trains a tiny model end-to-end by composing the Trainer
  directly (the `scripts/pretrain.py` flow); `test_pilot_fsdp.py` runs the script
  under torchrun. Keep these green — they prove the HF-Trainer + FSDP2 wiring.

## What Not To Do

- Don't add `# type: ignore` / `# ty: ignore` without a specific rule code.
- Don't use `os.path` — use `pathlib.Path`.
- Don't put logic in `__init__.py`.
- Don't reintroduce a custom trainer, a config/CLI system, an attention
  subpackage, or MoE.

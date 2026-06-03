# Agent Instructions for ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder (ported from `oplm`) wired to
the stock HuggingFace `Trainer`, launched via `torchrun` + FSDP2, with
SDPA-based attention and a pluggable optimizer registry.

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
`token_dropout=true`). Exact-size presets: `esmc_300m`,
`esmc_600m`, `esmc_6b`. The tokenizer is bit-for-bit ESM-C (33-token vocab).
Tests `tests/model/test_esm_alignment.py` pin this alignment — keep them green
when touching defaults. The architecture is a superset: `qk_norm`,
`residual_scaling`, `norm_strategy`, partial RoPE, and Canon convs are opt-in
knobs for experiments. (Exact ESM-2 parity would additionally need a plain GELU
MLP FFN, which is not yet implemented — only SwiGLU is.)

> Attention is just `F.scaled_dot_product_attention`, which auto-selects the
> fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime — no
> kernel registry, no torch.compile needed. A manual fp32-softmax path runs only
> for `output_attentions=True` (SDPA can't return weights).

## Core design rules (do not violate)

- **No `Trainer` subclass, no custom trainer loop.** Use stock
  `transformers.Trainer` directly. Optimizer choice flows through HF's native
  hooks: `TrainingArguments.optim`, `lr_scheduler_type`, and the
  `optimizers=` / `optimizer_cls_and_kwargs` constructor args. New optimizers go
  in the registry (`ablm/training/optim_registry.py`), never the Trainer.
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
│   ├── outputs.py norm.py masking.py rope.py embedding.py ffn.py conv.py
│   ├── attention.py            # AblmAttention: SDPA + manual-softmax fallback
│   ├── transformer.py          # AblmBlock + AblmStack (FSDP wrap unit: AblmBlock)
│   ├── configuration_ablm.py   # AblmConfig
│   ├── tokenization_ablm.py    # AblmTokenizerFast (33-token ESM-C vocab)
│   └── modeling_ablm.py        # all public Ablm* model classes
├── training/
│   ├── optim_registry.py       # name -> HF optim string | custom (cls/builder)
│   ├── muon.py                 # CombinedOptimizer (Muon 2D + AdamW rest)
│   └── grouping.py             # name-aware parameter partition helpers
├── data/                       # tokenizer + MLM dataset/collator
├── config.py                   # OmegaConf -> AblmConfig + TrainingArguments + ...
├── train.py                    # torchrun entry: build + transformers.Trainer
└── cli.py                      # `ablm train ...`, `ablm info`
tests/                          # pytest, mirrors src/
```

## Launching training

```bash
# single GPU
ablm train --config run.yaml
# multi-GPU + FSDP2 (set train.fsdp: "full_shard auto_wrap" in the YAML)
torchrun --standalone --nproc_per_node=8 -m ablm.train --config run.yaml
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
  `test_pilot_fsdp.py`) trains a tiny model end-to-end through the real
  `build_trainer` path — keep these green; they prove the HF-Trainer + FSDP2 wiring.

## What Not To Do

- Don't add `# type: ignore` / `# ty: ignore` without a specific rule code.
- Don't use `os.path` — use `pathlib.Path`.
- Don't put logic in `__init__.py`.
- Don't reintroduce a custom trainer, an attention subpackage, or MoE.

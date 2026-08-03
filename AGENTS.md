# Agent Instructions for ablm-forge

Lab base model-architecture repo for antibody/protein language-model
experiments. An ESM-style bidirectional encoder wired to the stock HuggingFace
`Trainer`, launched via `torchrun` + FSDP2, with SDPA-based attention and Muon
as the recommended production optimizer. It is a **library, not a framework**:
no config system, no CLI — you compose the pieces in a training script
(`scripts/pretrain.py` is the example).

Muon is the recommended optimizer for production runs: on a 350M AbLM it reached
lower eval loss than AdamW reproducibly (the largest single architectural effect
measured, -0.0058 eval/loss) and is LR-robust where AdamW degrades above ~1e-4.
AdamW remains the default for iteration and the safe choice under FSDP.

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
`ffn_bias=false`, `attention_bias=false`), **no QK-norm**, **no residual
scaling**, and **no token dropout** (`token_dropout=false` — ESM-2 had it; ESM-C
removed it as redundant under Pre-LN. The ESM-2 behavior is implemented and
available via `token_dropout=true`). ESM-C sizes are head_dim-64 at 30L/960,
36L/1152, 80L/2560 (300M / 600M / 6B) — set them directly on `AblmConfig`. The
tokenizer is bit-for-bit ESM-C (33-token vocab). Tests
`tests/model/test_esm_alignment.py` pin this alignment — keep them green when
touching defaults. The architecture is a superset: `qk_norm`, `residual_scaling`,
`norm_strategy`, partial RoPE, `token_dropout`, and `attention_bias` are opt-in
knobs for experiments. FFN variants are `swiglu` / `geglu` / `reglu` (gated, 3
matrices, gate activation from `ACT2FN`) and `gelu_mlp` (non-gated, 2 matrices,
ESM-2 style). Add a gated variant by registering it in `_FFN_VARIANTS` in
`ffn.py` — no new class needed. When `intermediate_size` is left `None` the
derivation is variant-aware: 4x hidden for `gelu_mlp`, ~8/3 x hidden for the
gated variants.

> Attention is just `F.scaled_dot_product_attention`, which auto-selects the
> fastest fused backend (FlashAttention / cuDNN / mem-efficient) at runtime — no
> kernel registry, no torch.compile needed. A manual fp32-softmax path runs only
> for `output_attentions=True` (SDPA can't return weights).

- **Default presets** live in `model/presets.py` (`from_preset("300m")`,
  `AblmConfig.from_preset(...)`): a muP-correct, kernel-optimized ladder
  (35m/150m/300m/600m/6b, d0=512). muP is opt-in (`mup_enabled`), so non-preset
  configs are unaffected.

## Core design rules (do not violate)

- **No config system, no CLI.** Configuration lives in Python: `AblmConfig`
  (a `PretrainedConfig`) for the model, `transformers.TrainingArguments` for
  training, composed in a script (`scripts/pretrain.py`). Don't add OmegaConf /
  YAML config trees / a `train` CLI. The in-code muP preset ladder in
  `model/presets.py` (see above) is the sanctioned exception — a plain Python
  registry, not a config system; don't add external/YAML preset files.
- **No custom trainer loop; subclass `Trainer` only to build Muon.** Use stock
  `transformers.Trainer`. HF-native optimizers via `TrainingArguments.optim`;
  schedules via `lr_scheduler_type`. The *one* sanctioned subclass is
  `OptimizerTrainer`, which overrides only `create_optimizer` to build Muon — this
  is mandatory, not stylistic: HF forbids a pre-built `optimizers=` tuple once FSDP
  is enabled, and `optimizer_cls_and_kwargs` can't express the name-based
  Muon/AdamW split. Don't override `training_step`/`compute_loss`/the loop, and
  don't add other subclasses.
- **Attention is SDPA + a manual fallback** in `ablm/model/layers/attention.py`.
  Don't reintroduce a kernel registry / explicit flash-attn integration: SDPA
  already auto-selects the fused backend.
- **All public model classes in `modeling_ablm.py`** (standard HF convention);
  the architecture building blocks live in the `model/layers/` subpackage.
- **Loading is register-based** (BALM-style): `import ablm` registers the classes
  with the Auto* factories, so checkpoints reload via `AutoModel*.from_pretrained`
  **with ablm-forge installed** — no `register_for_auto_class` / `auto_map` /
  `trust_remote_code`. That's what lets the model package use subpackages; don't
  reintroduce the file-copy path (it forces a flat package and can't follow
  subpackage-relative imports).
- **MoE is out of scope.** Do not reintroduce it.

## Architecture

- **src layout**: all package code under `src/ablm/`.
- **Build system**: hatchling (`pyproject.toml` only).
- **Testing**: pytest in `tests/`.

## Project Structure

```
src/ablm/
├── model/
│   ├── configuration_ablm.py   # AblmConfig
│   ├── tokenization_ablm.py    # AblmTokenizerFast (33-token ESM-C vocab)
│   ├── modeling_ablm.py        # all public Ablm* model classes
│   └── layers/                 # architecture building blocks (the screen surface)
│       ├── norm.py masking.py rope.py embedding.py ffn.py
│       ├── attention.py        # AblmAttention: SDPA + manual-softmax fallback
│       └── transformer.py      # AblmBlock + AblmStack (FSDP wrap unit: AblmBlock)
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

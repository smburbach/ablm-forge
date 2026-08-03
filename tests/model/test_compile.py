"""Compile smoke test — the training path must trace and run under torch.compile.

Prod trains with ``torch_compile=True`` (inductor), and the architecture screen
runs on the same compiled path for a fair comparison, so a graph break or an
aot_autograd failure would only surface at launch. Nothing else in the suite
exercises compile. This pins the contract cheaply on CPU: the ``aot_eager``
backend runs dynamo tracing *and* aot_autograd partitioning — the layer where
the plain-float residual-scale bug lived (``alpha`` is a buffer to guard it) —
without a GPU or inductor, and ``fullgraph=True`` fails on any graph break.
"""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM

_B, _T, _VOCAB = 2, 16, 33


def _anchor_shaped_config(**overrides) -> AblmConfig:
    """Tiny but architecturally faithful: the anchor's gated FFN, sqrt-depth
    residual scaling and QK-norm are the pieces most likely to trip tracing."""
    base = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        ffn_activation="swiglu",
        residual_scaling="sqrt_num_layers",
        qk_norm=True,
    )
    base.update(overrides)
    return AblmConfig(**base)


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Isolate each test from another's compile cache."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _train_one_compiled_step(config: AblmConfig) -> None:
    torch.manual_seed(0)
    model = AblmForMaskedLM(config).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)

    input_ids = torch.randint(4, _VOCAB, (_B, _T))
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    labels = input_ids.clone()

    out = compiled(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss).item()

    out.loss.backward()  # aot_eager builds and runs the backward graph here
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_anchor_masked_lm_compiles_and_trains_one_step() -> None:
    """The anchor architecture must trace as a single graph and take a step."""
    _train_one_compiled_step(_anchor_shaped_config())


@pytest.mark.slow
@pytest.mark.parametrize("ffn_activation", ["swiglu", "geglu", "reglu", "gelu_mlp"])
def test_every_ffn_variant_compiles(ffn_activation: str) -> None:
    """Each FFN variant the screen sweeps must compile without a graph break."""
    _train_one_compiled_step(_anchor_shaped_config(ffn_activation=ffn_activation))


@pytest.mark.slow
@pytest.mark.parametrize("norm_strategy", ["pre", "sandwich", "hybrid", "post_sdpa"])
def test_every_norm_strategy_compiles(norm_strategy: str) -> None:
    """Each norm placement the screen sweeps must compile without a graph break."""
    _train_one_compiled_step(_anchor_shaped_config(norm_strategy=norm_strategy))

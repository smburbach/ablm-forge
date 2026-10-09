"""The opt-in ``flash_attention_2`` path on real padded antibody batches (eval-25k, v2026-09-29).

SDPA cannot reach its flash backend with forge's key-padding mask, so the default path runs
memory-efficient attention. ``attn_implementation="flash_attention_2"`` routes through HF's
``flash_attention_forward`` (unpad -> ``flash_attn_varlen_func`` -> re-pad), the kernel stock
ESM2 trains with. Needs CUDA and ``flash_attn``; run in the cluster container.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset
from torch.profiler import ProfilerActivity, profile
from transformers import Trainer, TrainingArguments

from ablm import AblmConfig, AblmForMaskedLM
from ablm.model.tokenization_ablm import AblmTokenizerFast
from ablm.training.data import RegionAwareCollator, add_region_mask

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flash_attn") is None, reason="needs flash_attn"),
]

FIXTURE = Path(__file__).parents[1] / "fixtures" / "training" / "eval-25k_v2026-09-29.parquet"
N_ROWS = 64


def _config(attn_implementation: str, dropout: float = 0.0) -> AblmConfig:
    return AblmConfig(
        hidden_size=960,
        num_hidden_layers=2,
        num_attention_heads=20,
        intermediate_size=3840,
        max_position_embeddings=322,
        norm_type="layernorm",
        norm_eps=1e-12,
        ffn_activation="gelu",
        ffn_bias=True,
        attention_bias=True,
        token_dropout=True,
        hidden_dropout=dropout,
        attention_dropout=dropout,
        attn_implementation=attn_implementation,
    )


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    columns = [
        f"{c}:{i}" for c in ("sequence_aa", "cdr_mask_aa", "nongermline_mask_aa") for i in (0, 1)
    ]
    table = pq.read_table(FIXTURE, columns=[*columns, "v_mutation_count_aa:0"]).slice(0, N_ROWS)
    tokenizer = AblmTokenizerFast()
    return [
        add_region_mask(
            row,
            tokenizer,
            seq_col="sequence_aa",
            cdr_col="cdr_mask_aa",
            nt_col="nongermline_mask_aa",
            mutation_col="v_mutation_count_aa",
        )
        for row in table.to_pylist()
    ]


@pytest.fixture(scope="module")
def batch(rows: list[dict]) -> dict[str, torch.Tensor]:
    collator = RegionAwareCollator(
        tokenizer=AblmTokenizerFast(),
        mlm=True,
        mlm_probability=0.15,
        seed=12345,
        pad_to_multiple_of=64,
    )
    b = collator(rows)
    assert (b["attention_mask"] == 0).any(), "fixture batch must contain padding"
    return {k: b[k].cuda() for k in ("input_ids", "attention_mask", "labels")}


def _loss_and_grads(model: AblmForMaskedLM, batch: dict) -> tuple[torch.Tensor, dict]:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    return loss.detach().float(), {n: p.grad.float().clone() for n, p in model.named_parameters()}


def test_flash_attention_2_launches_flash_kernels(batch: dict) -> None:
    model = AblmForMaskedLM(_config("flash_attention_2", dropout=0.1)).cuda().train()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(**batch).loss.backward()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert any("flash_fwd" in k for k in kernels)
    assert any("flash_bwd" in k for k in kernels)
    assert not any("fmha_cutlass" in k for k in kernels)


def test_flash_attention_2_matches_sdpa_on_padded_batch(batch: dict) -> None:
    torch.manual_seed(0)
    sdpa = AblmForMaskedLM(_config("sdpa")).cuda().train()
    flash = AblmForMaskedLM(_config("flash_attention_2")).cuda().train()
    flash.load_state_dict(sdpa.state_dict())

    loss_s, grads_s = _loss_and_grads(sdpa, batch)
    loss_f, grads_f = _loss_and_grads(flash, batch)

    assert torch.allclose(loss_s, loss_f, rtol=5e-3)
    for name, g in grads_s.items():
        cos = torch.nn.functional.cosine_similarity(g.flatten(), grads_f[name].flatten(), dim=0)
        assert cos > 0.99, f"{name}: grad cosine {cos:.4f}"


def test_flash_attention_2_rejects_fp32_inputs(batch: dict) -> None:
    model = AblmForMaskedLM(_config("flash_attention_2")).cuda()
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        model(**batch)


def test_flash_attention_2_checkpoint_loads_under_sdpa(tmp_path: Path) -> None:
    AblmForMaskedLM(_config("flash_attention_2")).save_pretrained(tmp_path)
    model, info = AblmForMaskedLM.from_pretrained(
        tmp_path, attn_implementation="sdpa", output_loading_info=True
    )
    assert model.config._attn_implementation == "sdpa"
    assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))


def test_flash_attention_2_pilot_train_and_eval(rows: list[dict], tmp_path: Path) -> None:
    ds = Dataset.from_list(rows)
    region_collator = RegionAwareCollator(
        tokenizer=AblmTokenizerFast(), mlm=True, mlm_probability=0.15, seed=1, pad_to_multiple_of=64
    )

    def collator(examples: list[dict]) -> dict:
        b = region_collator(examples)
        return {k: b[k] for k in ("input_ids", "attention_mask", "labels")}

    trainer = Trainer(
        model=AblmForMaskedLM(_config("flash_attention_2", dropout=0.1)),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=6,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=1e-4,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            save_strategy="no",
            dataloader_num_workers=0,
        ),
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collator,
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    assert result.global_step == 6
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))

"""Parity of the ESM-2-config-matched AblmConfig against the phase-02 reference.

The reference is `esm2/12_sota_convergence/02_data_reg` run
`sota_data_reg_350M_AR32_unif-tdF-do0.1-s42` (W&B `m0lvqo30`), whose model was built
by ablm-sweeps' monkeypatch builder and has 333,147,364 parameters (reproduced
exactly 2026-07-31). forge legitimately omits ESM's vestigial contact head
(`esm.contact_head.regression`, a `Linear(450 -> 1)` = 451 params) which never
receives gradient during MLM training, so the config-matched model must land at
exactly 333,147,364 - 451.
"""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM

ESM_REFERENCE_PARAMS = 333_147_364
ESM_CONTACT_HEAD_PARAMS = 451
EXPECTED_PARAMS = ESM_REFERENCE_PARAMS - ESM_CONTACT_HEAD_PARAMS  # 333,146,913


def anchor_config(**overrides: object) -> AblmConfig:
    """The phase-02 AR32 pinned cell expressed as an AblmConfig."""
    fields: dict[str, object] = dict(
        vocab_size=33,
        hidden_size=960,
        num_hidden_layers=30,
        num_attention_heads=15,
        intermediate_size=2560,
        max_position_embeddings=322,
        norm_type="layernorm",
        norm_eps=1e-12,
        norm_bias=True,
        norm_strategy="pre",
        qk_norm=False,
        post_embed_norm=False,
        residual_scaling="none",
        ffn_activation="swiglu",
        ffn_bias=True,
        attention_bias=True,
        token_dropout=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        tie_word_embeddings=True,
        mlm_head_activation="gelu",
        pad_token_id=1,
        mask_token_id=32,
    )
    fields.update(overrides)
    return AblmConfig(**fields)  # ty: ignore[invalid-argument-type]


def test_anchor_config_head_dim_is_64():
    assert anchor_config().head_dim == 64


def test_anchor_config_param_count_matches_esm_reference_minus_contact_head():
    model = AblmForMaskedLM(anchor_config())
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_attention_bias_accounts_for_115200_of_the_gap():
    with_bias = AblmForMaskedLM(anchor_config())
    without = AblmForMaskedLM(anchor_config(attention_bias=False))
    n_with = sum(p.numel() for p in with_bias.parameters())
    n_without = sum(p.numel() for p in without.parameters())
    assert n_with - n_without == 4 * 960 * 30  # q/k/v/o bias per layer


def test_anchor_config_round_trips_through_to_dict():
    cfg = anchor_config()
    restored = AblmConfig(**cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()


def test_anchor_model_forward_produces_finite_logits():
    model = AblmForMaskedLM(anchor_config(num_hidden_layers=2)).eval()
    ids = torch.randint(4, 30, (2, 16))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.shape == (2, 16, 33)
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_anchor_shaped_model_trains_a_few_steps(tiny_training_parquet):
    """Pilot-scale run with the anchor's field set (few layers) through HF Trainer."""
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from ablm import AblmTokenizerFast

    tok = AblmTokenizerFast()
    ds = load_dataset("parquet", data_files=str(tiny_training_parquet), split="train")
    ds = ds.map(
        lambda b: tok(b["sequence"], truncation=True, max_length=64),
        batched=True,
        remove_columns=ds.column_names,
    )
    model = AblmForMaskedLM(
        anchor_config(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=1, intermediate_size=128
        )
    )
    args = TrainingArguments(
        output_dir="/tmp/anchor_pilot",
        max_steps=3,
        per_device_train_batch_size=2,
        eval_strategy="steps",
        eval_steps=3,
        report_to=[],
        logging_steps=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=True),
    )
    trainer.train()
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] > 0

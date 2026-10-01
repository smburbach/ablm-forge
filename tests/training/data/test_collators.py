"""Tests for `ablm.training.data.collators`.

Invariants: the side channels survive padding; a seeded collator ignores the global RNG (the
bug that made `eval_mask_seed` inert in ablm-sweeps); `RegionAwareCollator` masks exactly like
the stock HF collator on the same batch; `WeightedMaskingCollator` never selects region -1 and
realises the count distribution its `CountMode` promises.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import DataCollatorForLanguageModeling

from ablm.model.tokenization_ablm import AblmTokenizerFast
from ablm.training.data import (
    CountMode,
    RegionAwareCollator,
    WeightedMaskingCollator,
    add_region_mask,
    pair_mask,
)

# [CLS](1) + 20 heavy residues + <cls>(1) + 20 light residues + [EOS](1) -> separator at 21
_SEPARATOR_INDEX = 21


@pytest.fixture(scope="session")
def tokenizer() -> AblmTokenizerFast:
    return AblmTokenizerFast()


def _make_example(
    tokenizer: AblmTokenizerFast,
    heavy_seq: str,
    heavy_cdr: str,
    heavy_nt: str,
    light_seq: str,
    light_cdr: str,
    light_nt: str,
    mutations: tuple[int, int] | None = None,
) -> dict[str, Any]:
    example = {
        "seq:0": heavy_seq,
        "cdr:0": heavy_cdr,
        "nt:0": heavy_nt,
        "seq:1": light_seq,
        "cdr:1": light_cdr,
        "nt:1": light_nt,
    }
    kwargs = {}
    if mutations is not None:
        example["mut:0"], example["mut:1"] = mutations
        kwargs["mutation_col"] = "mut"
    return add_region_mask(example, tokenizer, seq_col="seq", cdr_col="cdr", nt_col="nt", **kwargs)


@pytest.fixture
def paired_example(tokenizer: AblmTokenizerFast) -> dict[str, Any]:
    """20 heavy residues (10-14 are CDR1) + 20 light residues (all framework)."""
    return _make_example(
        tokenizer,
        "M" * 10 + "E" * 5 + "M" * 5,
        "0" * 10 + "1" * 5 + "0" * 5,
        "0" * 20,
        "A" * 20,
        "0" * 20,
        "0" * 20,
    )


@pytest.fixture
def paired_examples(tokenizer: AblmTokenizerFast) -> list[dict[str, Any]]:
    """Four variable-length examples, two mutated, exercising padding and seq_mutated."""
    return [
        _make_example(
            tokenizer, "M" * 12, "0" * 12, "0" * 12, "A" * 8, "0" * 8, "0" * 8, mutations=(0, 0)
        ),
        _make_example(
            tokenizer,
            "M" * 20 + "E" * 6,
            "0" * 20 + "3" * 6,
            "0" * 20 + "1" * 6,
            "A" * 14,
            "0" * 14,
            "0" * 14,
            mutations=(3, 0),
        ),
        _make_example(
            tokenizer, "V" * 9, "0" * 9, "1" * 9, "S" * 11, "0" * 11, "0" * 11, mutations=(1, 2)
        ),
        _make_example(
            tokenizer,
            "M" * 16,
            "0" * 10 + "2" * 6,
            "0" * 16,
            "A" * 16,
            "0" * 16,
            "0" * 16,
            mutations=(0, 5),
        ),
    ]


# --- pair_mask / add_region_mask ---------------------------------------------------------


def test_pair_mask_wraps_and_separates_regions() -> None:
    assert pair_mask([1, 2, 3], [4, 5]) == [-1, 1, 2, 3, -1, 4, 5, -1]


def test_pair_mask_custom_ignore_index() -> None:
    assert pair_mask([1], [2], ignore_index=-2) == [-2, 1, -2, 2, -2]


def test_add_region_mask_is_aligned_and_length_matched(paired_example: dict[str, Any]) -> None:
    assert len(paired_example["region_mask"]) == len(paired_example["input_ids"])
    assert paired_example["region_mask"][0] == -1
    assert paired_example["region_mask"][_SEPARATOR_INDEX] == -1
    assert paired_example["region_mask"][-1] == -1
    assert "seq_mutated" not in paired_example


def test_add_region_mask_encodes_cdr_and_shm_regions(tokenizer: AblmTokenizerFast) -> None:
    example = _make_example(tokenizer, "MMEE", "0011", "0000", "AA", "00", "01")
    assert example["region_mask"] == [-1, 0, 0, 1, 1, -1, 0, 4, -1]


def test_add_region_mask_seq_mutated_is_heavy_chain_only(tokenizer: AblmTokenizerFast) -> None:
    assert (
        _make_example(tokenizer, "MM", "00", "00", "A", "0", "0", mutations=(0, 7))["seq_mutated"]
        == 0
    )
    assert (
        _make_example(tokenizer, "MM", "00", "00", "A", "0", "0", mutations=(2, 0))["seq_mutated"]
        == 1
    )


def test_add_region_mask_mismatched_lengths_raise(tokenizer: AblmTokenizerFast) -> None:
    example = {"seq:0": "MM", "cdr:0": "0", "nt:0": "00", "seq:1": "A", "cdr:1": "0", "nt:1": "0"}
    with pytest.raises(ValueError):
        add_region_mask(example, tokenizer, seq_col="seq", cdr_col="cdr", nt_col="nt")


# --- RegionAwareCollator ----------------------------------------------------------------


def test_region_aware_keeps_side_channels_through_padding(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]]
) -> None:
    collator = RegionAwareCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=1)
    batch = collator(paired_examples)
    pad = batch["input_ids"] == tokenizer.pad_token_id
    assert pad.any()
    assert batch["region_mask"].shape == batch["input_ids"].shape
    assert (batch["region_mask"][pad] == -1).all()
    assert (batch["labels"][pad] == -100).all()
    assert batch["seq_mutated"].tolist() == [0, 1, 1, 0]


def test_region_aware_without_mutation_col_has_no_seq_mutated(
    tokenizer: AblmTokenizerFast, paired_example: dict[str, Any]
) -> None:
    collator = RegionAwareCollator(tokenizer=tokenizer, mlm=True, seed=1)
    assert "seq_mutated" not in collator([paired_example, paired_example])


def test_region_aware_masks_exactly_like_the_stock_collator(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]]
) -> None:
    """Same seed, same examples minus the side channels -> identical input_ids and labels."""
    ours = RegionAwareCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=7, pad_to_multiple_of=8
    )
    stock = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=7, pad_to_multiple_of=8
    )
    plain = [
        {k: v for k, v in ex.items() if k not in ("region_mask", "seq_mutated")}
        for ex in paired_examples
    ]
    a = ours(paired_examples)
    b = stock(plain)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])


def test_region_aware_mlm_false_uses_input_ids_as_labels(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]]
) -> None:
    collator = RegionAwareCollator(tokenizer=tokenizer, mlm=False)
    batch = collator(paired_examples)
    pad = batch["input_ids"] == tokenizer.pad_token_id
    assert (batch["labels"][pad] == -100).all()
    assert torch.equal(batch["labels"][~pad], batch["input_ids"][~pad])
    assert "region_mask" in batch


# --- seeding (both collators) -------------------------------------------------------------


def _collators(
    tokenizer: AblmTokenizerFast, seed: int | None
) -> list[RegionAwareCollator | WeightedMaskingCollator]:
    return [
        RegionAwareCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=seed),
        *[
            WeightedMaskingCollator(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                seed=seed,
                cdr_ratios=3.0,
                count_mode=mode,
            )
            for mode in CountMode
        ],
    ]


@pytest.mark.parametrize("which", range(4))
def test_seeded_collators_ignore_global_rng(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]], which: int
) -> None:
    torch.manual_seed(8)
    a = _collators(tokenizer, seed=12345)[which](paired_examples)
    torch.manual_seed(16)
    b = _collators(tokenizer, seed=12345)[which](paired_examples)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])


@pytest.mark.parametrize("which", range(4))
def test_different_seeds_give_different_masks(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]], which: int
) -> None:
    a = _collators(tokenizer, seed=1)[which](paired_examples)
    b = _collators(tokenizer, seed=2)[which](paired_examples)
    assert not torch.equal(a["labels"], b["labels"])


@pytest.mark.parametrize("which", range(4))
def test_unseeded_collators_follow_global_rng(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]], which: int
) -> None:
    torch.manual_seed(3)
    a = _collators(tokenizer, seed=None)[which](paired_examples)
    torch.manual_seed(3)
    b = _collators(tokenizer, seed=None)[which](paired_examples)
    assert torch.equal(a["labels"], b["labels"])


# --- WeightedMaskingCollator ----------------------------------------------------------------


def test_weighted_rejects_bad_cdr_ratios_length(tokenizer: AblmTokenizerFast) -> None:
    with pytest.raises(ValueError, match="cdr_ratios"):
        WeightedMaskingCollator(tokenizer=tokenizer, cdr_ratios=[1.0, 2.0])


def test_weighted_region_weights_are_additive(tokenizer: AblmTokenizerFast) -> None:
    collator = WeightedMaskingCollator(tokenizer=tokenizer, cdr_ratios=3.0, nt_ratio=2.0)
    assert collator.region_weights.tolist() == [1.0, 3.0, 3.0, 3.0, 2.0, 4.0, 4.0, 4.0]


def test_weighted_count_mode_accepts_strings(tokenizer: AblmTokenizerFast) -> None:
    assert (
        WeightedMaskingCollator(tokenizer=tokenizer, count_mode="exact").count_mode
        is CountMode.EXACT
    )


@pytest.mark.parametrize("mode", list(CountMode))
def test_weighted_never_selects_region_minus_one(
    tokenizer: AblmTokenizerFast, paired_example: dict[str, Any], mode: CountMode
) -> None:
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.9,
        cdr_ratios=5.0,
        nt_ratio=5.0,
        count_mode=mode,
        seed=1,
    )
    for _ in range(20):
        batch = collator([paired_example, paired_example])
        assert (batch["labels"][:, [0, _SEPARATOR_INDEX, -1]] == -100).all()
        assert (
            batch["input_ids"][:, _SEPARATOR_INDEX] == tokenizer.convert_tokens_to_ids("<cls>")
        ).all()


@pytest.mark.parametrize("mlm_probability", [0.15, 0.4])
def test_exact_count_is_exact(
    tokenizer: AblmTokenizerFast, paired_example: dict[str, Any], mlm_probability: float
) -> None:
    n_valid = sum(1 for r in paired_example["region_mask"] if r >= 0)
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        cdr_ratios=2.0,
        nt_ratio=3.0,
        count_mode=CountMode.EXACT,
        seed=5,
    )
    for _ in range(10):
        n_masked = (collator([paired_example] * 4)["labels"] != -100).sum(dim=-1)
        assert (n_masked == round(n_valid * mlm_probability)).all()


@pytest.mark.parametrize("mode", [CountMode.BINOMIAL, CountMode.BERNOULLI])
def test_variable_count_modes_vary_and_average_to_np(
    tokenizer: AblmTokenizerFast, paired_example: dict[str, Any], mode: CountMode
) -> None:
    n_valid = sum(1 for r in paired_example["region_mask"] if r >= 0)
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.3, cdr_ratios=1.0, count_mode=mode, seed=5
    )
    counts = torch.cat(
        [(collator([paired_example] * 8)["labels"] != -100).sum(dim=-1) for _ in range(50)]
    ).float()
    assert counts.std() > 1.0
    assert abs(counts.mean().item() - n_valid * 0.3) < 0.5


def test_binomial_mode_masks_at_least_one_token_on_short_sequences(
    tokenizer: AblmTokenizerFast,
) -> None:
    tiny = _make_example(tokenizer, "MVE", "000", "000", "A", "0", "0")
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.05, count_mode=CountMode.BINOMIAL, seed=1
    )
    for _ in range(50):
        assert ((collator([tiny] * 8)["labels"] != -100).sum(dim=-1) >= 1).all()


def test_bernoulli_mode_realises_weight_ratio(tokenizer: AblmTokenizerFast) -> None:
    """CDR masking rate / framework masking rate equals the weight ratio under BERNOULLI."""
    example = _make_example(
        tokenizer, "M" * 60 + "E" * 60, "0" * 60 + "1" * 60, "0" * 120, "A" * 60, "0" * 60, "0" * 60
    )
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        cdr_ratios=3.0,
        count_mode=CountMode.BERNOULLI,
        seed=9,
    )
    region = torch.tensor(example["region_mask"])
    hits = torch.zeros(len(region))
    n = 400
    for _ in range(n):
        hits += (collator([example])["labels"][0] != -100).float()
    cdr_rate = hits[region == 1].mean() / n
    fw_rate = hits[region == 0].mean() / n
    assert cdr_rate / fw_rate == pytest.approx(3.0, abs=0.25)


def test_replacement_split_is_roughly_80_10_10(
    tokenizer: AblmTokenizerFast, paired_example: dict[str, Any]
) -> None:
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.5, count_mode=CountMode.EXACT, seed=3
    )
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    original = torch.tensor(paired_example["input_ids"])
    n_mask = n_random = n_total = 0
    for _ in range(60):
        batch = collator([paired_example])
        masked = batch["labels"][0] != -100
        n_total += masked.sum().item()
        is_mask = (batch["input_ids"][0] == mask_id) & masked
        n_mask += is_mask.sum().item()
        n_random += ((batch["input_ids"][0] != original) & masked & ~is_mask).sum().item()
    assert n_mask / n_total == pytest.approx(0.8, abs=0.05)
    assert n_random / n_total == pytest.approx(0.1, abs=0.05)


# --- seeding: stream continuity and weight validation ---------------------------------------


@pytest.mark.parametrize("which", range(4))
def test_consecutive_batches_from_one_seeded_collator_differ(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]], which: int
) -> None:
    collator = _collators(tokenizer, seed=12345)[which]
    a = collator(paired_examples)
    b = collator(paired_examples)
    assert not torch.equal(a["labels"], b["labels"])


def test_region_aware_matches_stock_over_consecutive_calls(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]]
) -> None:
    ours = RegionAwareCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=7, pad_to_multiple_of=8
    )
    stock = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=7, pad_to_multiple_of=8
    )
    plain = [
        {k: v for k, v in ex.items() if k not in ("region_mask", "seq_mutated")}
        for ex in paired_examples
    ]
    for _ in range(3):
        a, b = ours(paired_examples), stock(plain)
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])


@pytest.mark.parametrize("which", range(4))
def test_unseeded_collators_change_with_the_global_seed(
    tokenizer: AblmTokenizerFast, paired_examples: list[dict[str, Any]], which: int
) -> None:
    torch.manual_seed(3)
    a = _collators(tokenizer, seed=None)[which](paired_examples)
    torch.manual_seed(4)
    b = _collators(tokenizer, seed=None)[which](paired_examples)
    assert not torch.equal(a["labels"], b["labels"])


def test_bernoulli_mean_count_is_n_valid_p_at_non_uniform_weights(
    tokenizer: AblmTokenizerFast,
) -> None:
    example = _make_example(
        tokenizer,
        "M" * 60 + "E" * 60,
        "0" * 60 + "1" * 60,
        "0" * 120,
        "A" * 60,
        "0" * 60,
        "0" * 60,
    )
    n_valid = sum(1 for r in example["region_mask"] if r >= 0)
    collator = WeightedMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        cdr_ratios=3.0,
        count_mode=CountMode.BERNOULLI,
        seed=11,
    )
    counts = torch.cat(
        [(collator([example] * 8)["labels"] != -100).sum(dim=-1) for _ in range(50)]
    ).float()
    assert abs(counts.mean().item() - n_valid * 0.15) < 0.6


def test_weighted_rejects_negative_region_weights(tokenizer: AblmTokenizerFast) -> None:
    with pytest.raises(ValueError, match="region weight"):
        WeightedMaskingCollator(tokenizer=tokenizer, cdr_ratios=0.4, nt_ratio=0.5)

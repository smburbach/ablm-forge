"""Tests for `ablm.training.data.collators` — region-weighted CDR masking collator.

Load-bearing invariants under test: exact per-sequence masked-token count,
separator (`region_mask == -1`) never masked, and CDR-region weighting biasing
Gumbel-top-k selection relative to the uniform (`cdr_ratios=1.0`) arm.
"""

from __future__ import annotations

import pytest
import torch

from ablm.model.tokenization_ablm import AblmTokenizerFast
from ablm.training.data import PreferentialMaskingCollator, add_region_mask, pair_mask

# Index of the interior `<cls>` separator for the fixed-length example built below:
# [CLS](1) + 20 heavy residues + <cls>(1) + 20 light residues + [EOS](1) -> index 21.
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
) -> dict:
    example = {
        "seq:0": heavy_seq,
        "cdr:0": heavy_cdr,
        "nt:0": heavy_nt,
        "seq:1": light_seq,
        "cdr:1": light_cdr,
        "nt:1": light_nt,
    }
    return add_region_mask(example, tokenizer, cdr_col="cdr", nt_col="nt", seq_col="seq")


@pytest.fixture
def paired_example(tokenizer: AblmTokenizerFast) -> dict:
    """20 heavy residues (positions 10-14 are CDR1) + 20 light residues (all FW)."""
    heavy_seq = "M" * 10 + "E" * 5 + "M" * 5
    heavy_cdr = "0" * 10 + "1" * 5 + "0" * 5
    heavy_nt = "0" * 20
    light_seq = "A" * 20
    light_cdr = "0" * 20
    light_nt = "0" * 20
    return _make_example(tokenizer, heavy_seq, heavy_cdr, heavy_nt, light_seq, light_cdr, light_nt)


# ---------------------------------------------------------------------------
# pair_mask / add_region_mask
# ---------------------------------------------------------------------------


def test_pair_mask_wraps_and_separates_regions():
    assert pair_mask([1, 2, 3], [4, 5]) == [-1, 1, 2, 3, -1, 4, 5, -1]


def test_pair_mask_custom_ignore_index():
    assert pair_mask([1], [2], ignore_index=-2) == [-2, 1, -2, 2, -2]


def test_add_region_mask_is_aligned_and_length_matched(paired_example: dict):
    assert len(paired_example["region_mask"]) == len(paired_example["input_ids"])
    # outer CLS/EOS and the interior separator are all -1.
    assert paired_example["region_mask"][0] == -1
    assert paired_example["region_mask"][_SEPARATOR_INDEX] == -1
    assert paired_example["region_mask"][-1] == -1


def test_add_region_mask_encodes_cdr_and_shm_regions(tokenizer: AblmTokenizerFast):
    example = _make_example(tokenizer, "MMEE", "0011", "0000", "AA", "00", "01")
    # heavy: M(FW=0) M(FW=0) E(CDR1=1) E(CDR1=1); light: A(FW=0) A(FW=0+4*1=4, SHM)
    assert example["region_mask"] == [-1, 0, 0, 1, 1, -1, 0, 4, -1]


def test_add_region_mask_mismatched_lengths_assert():
    tokenizer = AblmTokenizerFast()
    example = {
        "seq:0": "MM",
        "cdr:0": "0",  # too short vs seq:0
        "nt:0": "00",
        "seq:1": "A",
        "cdr:1": "0",
        "nt:1": "0",
    }
    with pytest.raises((AssertionError, ValueError)):
        add_region_mask(example, tokenizer, cdr_col="cdr", nt_col="nt", seq_col="seq")


# ---------------------------------------------------------------------------
# PreferentialMaskingCollator construction (transformers 5.3.0 compatibility)
# ---------------------------------------------------------------------------


def test_collator_constructs_under_installed_transformers(tokenizer: AblmTokenizerFast):
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, cdr_ratios=1.0, nt_ratio=1.0
    )
    assert collator.mlm is True
    assert collator.mlm_probability == pytest.approx(0.15)
    assert collator.region_multipliers.shape == (8,)


def test_collator_scalar_cdr_ratios_broadcast_to_three(tokenizer: AblmTokenizerFast):
    collator = PreferentialMaskingCollator(tokenizer=tokenizer, cdr_ratios=3.0, nt_ratio=2.0)
    assert collator.cdr_ratios == [3.0, 3.0, 3.0]
    # region 5 (CDR1+SHM) is additive: cdr + nt - 1
    assert collator.region_multipliers[5].item() == pytest.approx(3.0 + 2.0 - 1.0)


def test_collator_rejects_bad_cdr_ratios_length(tokenizer: AblmTokenizerFast):
    with pytest.raises(ValueError, match="cdr_ratios"):
        PreferentialMaskingCollator(tokenizer=tokenizer, cdr_ratios=[1.0, 2.0])


# ---------------------------------------------------------------------------
# Exact-count masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mlm_probability", [0.15, 0.4])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_exact_count_masking(
    tokenizer: AblmTokenizerFast, paired_example: dict, mlm_probability: float, batch_size: int
):
    n_valid = sum(1 for r in paired_example["region_mask"] if r >= 0)
    expected = round(n_valid * mlm_probability)

    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        cdr_ratios=1.0,
        nt_ratio=1.0,
    )
    torch.manual_seed(0)
    batch = collator([paired_example] * batch_size)
    n_masked = (batch["labels"] != -100).sum(dim=-1)
    assert torch.equal(n_masked, torch.full((batch_size,), expected, dtype=n_masked.dtype))


def test_exact_count_masking_holds_across_seeds(tokenizer: AblmTokenizerFast, paired_example: dict):
    n_valid = sum(1 for r in paired_example["region_mask"] if r >= 0)
    expected = round(n_valid * 0.25)
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.25, cdr_ratios=2.0, nt_ratio=3.0
    )
    for seed in range(10):
        torch.manual_seed(seed)
        batch = collator([paired_example, paired_example])
        n_masked = (batch["labels"] != -100).sum(dim=-1)
        assert (n_masked == expected).all(), f"seed={seed}: {n_masked=} expected={expected}"


# ---------------------------------------------------------------------------
# Separator protection (region_mask == -1 is never masked)
# ---------------------------------------------------------------------------


def test_separator_position_never_masked_across_seeds(
    tokenizer: AblmTokenizerFast, paired_example: dict
):
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.9, cdr_ratios=5.0, nt_ratio=5.0
    )
    for seed in range(20):
        torch.manual_seed(seed)
        batch = collator([paired_example, paired_example])
        assert (batch["labels"][:, _SEPARATOR_INDEX] == -100).all(), (
            f"seed={seed}: separator was masked"
        )
        # the input_ids at the separator must also be untouched (never <mask> / random-swapped)
        assert (
            batch["input_ids"][:, _SEPARATOR_INDEX] == tokenizer.convert_tokens_to_ids("<cls>")
        ).all()


def test_outer_specials_never_masked(tokenizer: AblmTokenizerFast, paired_example: dict):
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.9, cdr_ratios=1.0, nt_ratio=1.0
    )
    torch.manual_seed(0)
    batch = collator([paired_example])
    assert batch["labels"][0, 0].item() == -100  # outer CLS
    assert batch["labels"][0, -1].item() == -100  # outer EOS


# ---------------------------------------------------------------------------
# Uniform vs weighted selection
# ---------------------------------------------------------------------------


def _region_selection_rate(
    tokenizer: AblmTokenizerFast,
    example: dict,
    cdr_ratios: float,
    nt_ratio: float,
    mlm_probability: float,
    n_trials: int,
) -> tuple[float, float]:
    """Fraction of trials each FW (region 0) / CDR1 (region 1) position gets masked, averaged."""
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        cdr_ratios=cdr_ratios,
        nt_ratio=nt_ratio,
    )
    region_mask = torch.tensor(example["region_mask"])
    fw_positions = (region_mask == 0).nonzero(as_tuple=True)[0]
    cdr_positions = (region_mask == 1).nonzero(as_tuple=True)[0]

    fw_hits = 0
    cdr_hits = 0
    for seed in range(n_trials):
        torch.manual_seed(seed)
        batch = collator([example])
        masked = batch["labels"][0] != -100
        fw_hits += masked[fw_positions].sum().item()
        cdr_hits += masked[cdr_positions].sum().item()

    fw_rate = fw_hits / (len(fw_positions) * n_trials)
    cdr_rate = cdr_hits / (len(cdr_positions) * n_trials)
    return fw_rate, cdr_rate


def test_uniform_ratios_select_fw_and_cdr_at_similar_rates(
    tokenizer: AblmTokenizerFast, paired_example: dict
):
    fw_rate, cdr_rate = _region_selection_rate(
        tokenizer, paired_example, cdr_ratios=1.0, nt_ratio=1.0, mlm_probability=0.3, n_trials=200
    )
    assert cdr_rate == pytest.approx(fw_rate, abs=0.08)
    assert cdr_rate == pytest.approx(0.3, abs=0.08)


def test_weighted_ratios_bias_selection_toward_cdr(
    tokenizer: AblmTokenizerFast, paired_example: dict
):
    fw_rate, cdr_rate = _region_selection_rate(
        tokenizer, paired_example, cdr_ratios=3.0, nt_ratio=1.0, mlm_probability=0.3, n_trials=200
    )
    assert cdr_rate > fw_rate + 0.1


# ---------------------------------------------------------------------------
# 80/10/10 split
# ---------------------------------------------------------------------------


def test_replacement_split_is_roughly_80_10_10(tokenizer: AblmTokenizerFast, paired_example: dict):
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.5, cdr_ratios=1.0, nt_ratio=1.0
    )
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    n_mask_token = 0
    n_changed_not_mask = 0
    n_total_masked = 0
    original_ids = torch.tensor(paired_example["input_ids"])
    for seed in range(60):
        torch.manual_seed(seed)
        batch = collator([paired_example])
        masked = batch["labels"][0] != -100
        n_total_masked += masked.sum().item()
        is_mask_token = (batch["input_ids"][0] == mask_id) & masked
        n_mask_token += is_mask_token.sum().item()
        changed = (batch["input_ids"][0] != original_ids) & masked & ~is_mask_token
        n_changed_not_mask += changed.sum().item()

    mask_frac = n_mask_token / n_total_masked
    random_frac = n_changed_not_mask / n_total_masked
    assert mask_frac == pytest.approx(0.8, abs=0.05)
    assert random_frac == pytest.approx(0.1, abs=0.05)


# ---------------------------------------------------------------------------
# torch_call: mlm=False path + padding across variable-length examples
# ---------------------------------------------------------------------------


def test_mlm_false_uses_input_ids_as_labels_with_pad_ignored(tokenizer: AblmTokenizerFast):
    short = _make_example(tokenizer, "MM", "00", "00", "A", "0", "0")
    long = _make_example(tokenizer, "MMMM", "0000", "0000", "AA", "00", "00")
    collator = PreferentialMaskingCollator(tokenizer=tokenizer, mlm=False)
    batch = collator([short, long])
    pad_id = tokenizer.pad_token_id
    assert (batch["labels"][batch["input_ids"] == pad_id] == -100).all()
    non_pad = batch["input_ids"] != pad_id
    assert torch.equal(batch["labels"][non_pad], batch["input_ids"][non_pad])


def test_padding_fills_region_mask_with_negative_one(tokenizer: AblmTokenizerFast):
    short = _make_example(tokenizer, "MM", "00", "00", "A", "0", "0")
    long = _make_example(tokenizer, "MMMM", "0000", "0000", "AA", "00", "00")
    collator = PreferentialMaskingCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, cdr_ratios=1.0, nt_ratio=1.0
    )
    torch.manual_seed(0)
    batch = collator([short, long])
    pad_positions = batch["input_ids"] == tokenizer.pad_token_id
    assert pad_positions.any()
    assert (batch["labels"][pad_positions] == -100).all()

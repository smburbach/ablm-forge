"""Realised masking statistics on 2,048 real paired antibodies (eval-25k, v2026-09-29).

Pins the 2026-09-28 measurements from ablm-sweeps (mean of 4 collations, same data):
stock uniform mask rate 0.1505 / count_z_sd 0.992; exact 0.1497 / 0.053 / CDR-FW 2.69;
binomial top-k 0.1500 / 1.013 / 2.68; weighted Bernoulli 0.1500 / 0.973 / 2.98.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from ablm.model.tokenization_ablm import AblmTokenizerFast
from ablm.training.data import (
    TIERS,
    CountMode,
    RegionAwareCollator,
    WeightedMaskingCollator,
    add_region_mask,
    masking_stats,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "training" / "eval-25k_v2026-09-29.parquet"
N_ROWS = 2048
BATCH = 256
P = 0.15


@pytest.fixture(scope="module")
def tokenized_rows() -> list[dict]:
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    table = pq.read_table(
        FIXTURE,
        columns=[
            "sequence_aa:0",
            "sequence_aa:1",
            "cdr_mask_aa:0",
            "cdr_mask_aa:1",
            "nongermline_mask_aa:0",
            "nongermline_mask_aa:1",
            "v_mutation_count_aa:0",
        ],
    ).slice(0, N_ROWS)
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


def _aggregate(collator, rows):
    labels, regions = [], []
    for i in range(0, len(rows), BATCH):
        batch = collator(rows[i : i + BATCH])
        labels.append(batch["labels"])
        regions.append(batch["region_mask"])
    width = max(t.shape[1] for t in labels)

    def pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        return torch.nn.functional.pad(t, (0, width - t.shape[1]), value=fill)

    return masking_stats(
        torch.cat([pad(t, -100) for t in labels]), torch.cat([pad(t, -1) for t in regions]), P
    )


def _cdr_fw_ratio(stats):
    return stats["rate_cdr_templated"] / stats["rate_fw_templated"]


def test_stock_uniform(tokenized_rows):
    s = _aggregate(
        RegionAwareCollator(tokenizer=AblmTokenizerFast(), mlm=True, mlm_probability=P, seed=12345),
        tokenized_rows,
    )
    assert s["mask_rate"] == pytest.approx(0.150, abs=0.004)
    assert s["count_z_sd"] == pytest.approx(1.0, abs=0.08)
    assert _cdr_fw_ratio(s) == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize(
    ("mode", "z_sd", "ratio"),
    [
        (CountMode.EXACT, (0.0, 0.10), (2.55, 2.85)),
        (CountMode.BINOMIAL, (0.90, 1.10), (2.55, 2.85)),
        (CountMode.BERNOULLI, (0.88, 1.06), (2.85, 3.12)),
    ],
)
def test_weighted_modes_at_cdr3_nt1(tokenized_rows, mode, z_sd, ratio):
    collator = WeightedMaskingCollator(
        tokenizer=AblmTokenizerFast(),
        mlm=True,
        mlm_probability=P,
        seed=12345,
        cdr_ratios=3.0,
        nt_ratio=1.0,
        count_mode=mode,
    )
    s = _aggregate(collator, tokenized_rows)
    assert s["mask_rate"] == pytest.approx(0.150, abs=0.004)
    assert z_sd[0] <= s["count_z_sd"] <= z_sd[1]
    assert ratio[0] <= _cdr_fw_ratio(s) <= ratio[1]
    assert set(TIERS) <= {k.removeprefix("rate_") for k in s if k.startswith("rate_")}


def test_exact_at_weight_one_matches_stock_rates(tokenized_rows):
    s = _aggregate(
        WeightedMaskingCollator(
            tokenizer=AblmTokenizerFast(),
            mlm=True,
            mlm_probability=P,
            seed=12345,
            cdr_ratios=1.0,
            count_mode=CountMode.EXACT,
        ),
        tokenized_rows,
    )
    assert s["count_z_sd"] < 0.10
    assert _cdr_fw_ratio(s) == pytest.approx(1.0, abs=0.08)

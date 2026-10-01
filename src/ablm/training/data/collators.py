"""Paired-antibody MLM collators that carry ``region_mask`` / ``seq_mutated`` through collation.

Region code is ``cdr_level + 4 * nt_flag``: 0 framework-templated, 1-3 CDR1-3 templated, 4-7
the same with the position non-templated. ``-1`` marks special tokens, including the interior
``<cls>`` between the chains: ``get_special_tokens_mask`` flags only the outer CLS/EOS, so
without it the separator would land in CDR3's bucket.

Two collators, one contract:

- ``RegionAwareCollator`` is the stock HF Bernoulli collator with the side channels surviving
  dynamic padding. It is the production collator: fixing the per-sequence mask count instead
  costs +0.01447 on held-out donors.
- ``WeightedMaskingCollator`` biases selection toward CDR / non-templated positions, with the
  per-sequence count as a separate ``CountMode``.

Ported from ablm-sweeps ``esm2/training_mods/region_eval.py`` and ``weighted_masking.py``
(``exp/rerun-v2026-09-17`` @ ``9a607d6``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from transformers import DataCollatorForLanguageModeling

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = [
    "IGNORE",
    "TIERS",
    "CountMode",
    "RegionAwareCollator",
    "WeightedMaskingCollator",
    "add_region_mask",
    "pair_mask",
]

IGNORE = -1

TIERS: dict[str, tuple[int, ...]] = {
    "fw_templated": (0,),
    "cdr_templated": (1, 2, 3),
    "fw_shm": (4,),
    "cdr_shm": (5, 6, 7),
}


def pair_mask(
    heavy_region: list[int], light_region: list[int], ignore_index: int = IGNORE
) -> list[int]:
    """Lay per-residue codes out over ``[CLS] heavy <cls> light [EOS]``."""
    return [ignore_index, *heavy_region, ignore_index, *light_region, ignore_index]


def add_region_mask(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    seq_col: str,
    cdr_col: str,
    nt_col: str,
    mutation_col: str | None = None,
    separator: str = "<cls>",
    padding: bool | str = False,
    truncation: bool = True,
    max_length: int = 320,
) -> dict[str, Any]:
    """Tokenize one paired example and attach its region codes (and mutation flag).

    Args:
        example: Row with ``f"{col}:0"`` (heavy) and ``f"{col}:1"`` (light) entries for
            ``seq_col``, ``cdr_col``, ``nt_col`` and, if given, ``mutation_col``.
        tokenizer: Tokenizer that treats ``separator`` as a single special token.
        seq_col: Base name of the sequence column.
        cdr_col: Base name of the per-residue CDR level mask (one digit per residue).
        nt_col: Base name of the per-residue non-templated mask (one digit per residue).
        mutation_col: Base name of the per-chain V-gene mutation count; heavy chain only,
            ``seq_mutated = int(count > 0)``. ``None`` omits the flag.
        separator: Token placed between the chains.
        padding: Passed to the tokenizer.
        truncation: Passed to the tokenizer.
        max_length: Passed to the tokenizer.

    Returns:
        The tokenizer output plus ``region_mask`` (``list[int]``, one code per token) and,
        when ``mutation_col`` is set, ``seq_mutated`` (``int``).

    Raises:
        ValueError: If a mask's length differs from its chain or the paired region mask does
            not line up with the tokens (the tokenizer truncates; ``pair_mask`` does not).
    """
    paired = example[f"{seq_col}:0"] + separator + example[f"{seq_col}:1"]
    tokenized = tokenizer(
        paired,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
        return_special_tokens_mask=True,
    )

    def region(chain: str) -> list[int]:
        cdr = [int(c) for c in example[f"{cdr_col}:{chain}"]]
        nt = [int(c) for c in example[f"{nt_col}:{chain}"]]
        if len(cdr) != len(nt):
            raise ValueError(
                f"{cdr_col}:{chain} has {len(cdr)} positions but {nt_col}:{chain} has {len(nt)}"
            )
        return [c + 4 * n for c, n in zip(cdr, nt, strict=True)]

    region_mask = pair_mask(region("0"), region("1"))
    n_tokens = len(tokenized["input_ids"])
    if len(region_mask) != n_tokens:
        raise ValueError(
            f"region_mask length {len(region_mask)} != input_ids length {n_tokens}; "
            f"check that {cdr_col}/{nt_col} have one character per token"
        )
    tokenized["region_mask"] = region_mask
    if mutation_col is not None:
        tokenized["seq_mutated"] = int(example[f"{mutation_col}:0"] > 0)
    return tokenized


def _pad_paired_batch(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    """Dynamic-pad a batch, keeping the side channels ``tokenizer.pad`` would mangle.

    ``region_mask`` is popped before padding and re-added filled with ``-1``, because
    ``tokenizer.pad`` zero-pads unknown columns and 0 is a real region code. ``seq_mutated``
    is one scalar per example, which ``tokenizer.pad`` cannot handle at all.
    """
    region_masks = [ex["region_mask"] for ex in examples]
    mutated = [ex["seq_mutated"] for ex in examples] if "seq_mutated" in examples[0] else None
    drop = {"region_mask", "seq_mutated"}
    clean = [{k: v for k, v in ex.items() if k not in drop} for ex in examples]

    batch = tokenizer.pad(clean, return_tensors="pt", pad_to_multiple_of=pad_to_multiple_of)
    max_len = batch["input_ids"].shape[1]  # ty: ignore[unresolved-attribute]  # return_tensors="pt"

    padded = torch.full((len(region_masks), max_len), IGNORE, dtype=torch.long)
    for i, m in enumerate(region_masks):
        padded[i, : len(m)] = torch.tensor(m, dtype=torch.long)
    batch["region_mask"] = padded
    if mutated is not None:
        batch["seq_mutated"] = torch.tensor(mutated, dtype=torch.long)
    return batch  # ty: ignore[invalid-return-type]  # BatchEncoding is dict-like at runtime


@dataclass
class RegionAwareCollator(DataCollatorForLanguageModeling):
    """Stock Bernoulli MLM masking, with ``region_mask`` / ``seq_mutated`` surviving collation.

    Masking is the base class's, untouched. ``seed`` is honoured: the base class creates its
    generator lazily inside its own ``torch_call``, so an override has to do the same or the
    seed is silently inert and masking falls back to the global RNG.
    """

    def torch_call(  # ty: ignore[invalid-method-override]  # narrows base's untyped examples
        self, examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.seed and self.generator is None:
            self.create_rng()
        batch = _pad_paired_batch(examples, self.tokenizer, self.pad_to_multiple_of)
        side = {k: batch.pop(k) for k in ("region_mask", "seq_mutated") if k in batch}
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        batch.update(side)
        return batch


class CountMode(StrEnum):
    """How many tokens a sequence masks, given its region weights.

    ``EXACT`` fixes the count at ``round(n_valid * p)`` (the original top-k collator);
    ``BINOMIAL`` draws ``k ~ Binomial(n_valid, p)`` and keeps Gumbel-top-k selection;
    ``BERNOULLI`` draws every position independently at ``p_i = w_i / sum(w) * n_valid * p``.
    """

    EXACT = "exact"
    BINOMIAL = "binomial"
    BERNOULLI = "bernoulli"


@dataclass
class WeightedMaskingCollator(RegionAwareCollator):
    """Region-weighted MLM masking with a selectable per-sequence count.

    Weights: framework 1.0, CDRn ``cdr_ratios[n]``, framework non-templated ``nt_ratio``,
    CDR non-templated ``cdr + nt - 1``. The 80/10/10 corruption and the seeded generator are
    the stock collator's, so only token selection differs from ``RegionAwareCollator``.
    Positions with region ``-1`` (specials, the chain separator) are never selected.
    """

    cdr_ratios: float | list[float] = 1.0
    nt_ratio: float = 1.0
    count_mode: CountMode = CountMode.BERNOULLI

    def __post_init__(self) -> None:
        super().__post_init__()
        cdr = self.cdr_ratios
        ratios = [float(cdr)] * 3 if isinstance(cdr, (int, float)) else [float(r) for r in cdr]
        if len(ratios) != 3:
            raise ValueError("cdr_ratios must be a float or a list of 3 floats [cdr1, cdr2, cdr3]")
        nt = float(self.nt_ratio)
        self.count_mode = CountMode(self.count_mode)
        self.region_weights = torch.tensor([1.0, *ratios, nt, *(c + nt - 1.0 for c in ratios)])

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.seed and self.generator is None:
            self.create_rng()
        batch = _pad_paired_batch(examples, self.tokenizer, self.pad_to_multiple_of)
        side = {k: batch.pop(k) for k in ("region_mask", "seq_mutated") if k in batch}
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self._mask(
            batch["input_ids"], side["region_mask"], special_tokens_mask
        )
        batch.update(side)
        return batch

    def _select(self, weights: torch.Tensor, maskable: torch.Tensor) -> torch.Tensor:
        """Boolean ``(B, L)`` selection under ``count_mode``; weights are 0 where not maskable."""
        valid = maskable.sum(dim=-1)
        p: float = self.mlm_probability  # ty: ignore[invalid-assignment]  # base types it Optional
        if self.count_mode is CountMode.BERNOULLI:
            w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            p_tok = weights / w_sum * (valid.float() * p).unsqueeze(-1)
            return torch.bernoulli(p_tok.clamp(max=1.0), generator=self.generator).bool() & maskable

        if self.count_mode is CountMode.EXACT:
            n_mask = (valid.float() * p).round().long()
        else:
            n_mask = torch.binomial(
                valid.float(),
                torch.full_like(valid, p, dtype=torch.float),
                generator=self.generator,
            ).long()
            n_mask = torch.minimum(n_mask.clamp(min=1), valid)

        u = torch.rand(weights.shape, generator=self.generator).clamp(1e-10, 1 - 1e-10)
        scores = torch.log(weights.clamp(min=1e-10)) - torch.log(-torch.log(u))
        scores.masked_fill_(~maskable, float("-inf"))
        ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
        return ranks < n_mask.unsqueeze(-1)

    def _mask(
        self,
        inputs: torch.Tensor,
        region_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(v, already_has_special_tokens=True)
                    for v in labels.tolist()
                ],
                dtype=torch.bool,
            )
        maskable = (region_mask >= 0) & ~special_tokens_mask.bool()
        weights = self.region_weights[region_mask.clamp(min=0)] * maskable
        selected = self._select(weights, maskable)
        labels[~selected] = -100

        replace = float(self.mask_replace_prob)
        to_mask = torch.bernoulli(
            torch.full(labels.shape, replace), generator=self.generator
        ).bool()
        to_mask &= selected
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[to_mask] = mask_id  # ty: ignore[invalid-assignment]  # single token -> int
        if replace < 1 and self.random_replace_prob > 0:
            scaled = float(self.random_replace_prob) / (1 - replace)
            to_random = torch.bernoulli(
                torch.full(labels.shape, scaled), generator=self.generator
            ).bool()
            to_random &= selected & ~to_mask
            words = torch.randint(
                len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator
            )
            inputs[to_random] = words[to_random]
        return inputs, labels

"""Region-weighted CDR masking collator.

Region = ``cdr_level + 4 * nt_flag`` (0=FW, 1-3=CDRn, +4 if SHM), aligned
token-for-token to ``input_ids``. Ported from
``esm2/12_sota_convergence/training_mods/preferential_masking.py`` in
ablm-sweeps (same lineage as the ``05_preferential_masking_sweep`` origin).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import DataCollatorForLanguageModeling

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["PreferentialMaskingCollator", "add_region_mask", "pair_mask"]


# --- region_mask construction ---


def pair_mask(
    heavy_region: list[int], light_region: list[int], ignore_index: int = -1
) -> list[int]:
    """Build the paired region mask: ``[CLS] heavy <cls> light [EOS]``.

    The interior ``<cls>`` needs the -1: ``get_special_tokens_mask`` flags only
    the outer CLS/EOS, so nothing else keeps it out of the maskable set.
    """
    return (
        [ignore_index] + list(heavy_region) + [ignore_index] + list(light_region) + [ignore_index]
    )


def add_region_mask(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    cdr_col: str,
    nt_col: str,
    seq_col: str,
    separator: str = "<cls>",
    padding: bool | str = False,
    truncation: bool = True,
    max_length: int = 320,
) -> dict[str, Any]:
    """Tokenize a paired (heavy, light) example and attach an aligned region_mask.

    region = cdr_level + 4*nt_flag (0=FW, 1-3=CDRn, +4 if SHM), aligned to input_ids.
    """
    paired = example[f"{seq_col}:0"] + separator + example[f"{seq_col}:1"]
    tokenized = tokenizer(
        paired,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
        return_special_tokens_mask=True,
    )

    # contiguous digit strings, one char per residue
    heavy_cdr = [int(c) for c in example[f"{cdr_col}:0"]]
    heavy_nt = [int(c) for c in example[f"{nt_col}:0"]]
    light_cdr = [int(c) for c in example[f"{cdr_col}:1"]]
    light_nt = [int(c) for c in example[f"{nt_col}:1"]]

    heavy_region = [c + 4 * n for c, n in zip(heavy_cdr, heavy_nt, strict=True)]
    light_region = [c + 4 * n for c, n in zip(light_cdr, light_nt, strict=True)]

    tokenized["region_mask"] = pair_mask(heavy_region, light_region)

    # pair_mask does not truncate but the tokenizer does: a mismatch would silently misalign
    n_tokens = len(tokenized["input_ids"])
    assert len(tokenized["region_mask"]) == n_tokens, (
        f"region_mask length {len(tokenized['region_mask'])} != input_ids length {n_tokens}"
    )
    return tokenized


# --- collator ---


def _pad_paired_batch(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    """Dynamic-pad a paired batch. region_mask is popped so tokenizer.pad cannot zero-pad it,
    then re-added padded to -1."""
    region_masks = [ex["region_mask"] for ex in examples]
    clean_examples = [{k: v for k, v in ex.items() if k != "region_mask"} for ex in examples]

    batch = tokenizer.pad(
        clean_examples,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of,
    )

    max_len = batch["input_ids"].shape[1]  # ty: ignore[unresolved-attribute]  # return_tensors="pt"

    def pad_with(masks: list[list[int]], fill: int) -> torch.Tensor:
        padded = torch.full((len(masks), max_len), fill, dtype=torch.long)
        for i, m in enumerate(masks):
            padded[i, : len(m)] = torch.tensor(m, dtype=torch.long)
        return padded

    batch["region_mask"] = pad_with(region_masks, fill=-1)
    return batch  # ty: ignore[invalid-return-type]  # BatchEncoding is dict-like at runtime


@dataclass
class PreferentialMaskingCollator(DataCollatorForLanguageModeling):
    """Preferential masked-LM collator: Gumbel-top-k selection masks exactly
    round(n_valid * mlm_probability) tokens per sequence, biased toward higher-weight regions."""

    # float = same ratio for CDR1/2/3; list = [cdr1, cdr2, cdr3]
    cdr_ratios: float | list[float] = 2.0
    nt_ratio: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()

        if isinstance(self.cdr_ratios, (int, float)):
            self.cdr_ratios = [float(self.cdr_ratios)] * 3
        if len(self.cdr_ratios) != 3:
            raise ValueError("cdr_ratios must be a float or a list of 3 floats [cdr1, cdr2, cdr3]")

        cdr1_r, cdr2_r, cdr3_r = [float(r) for r in self.cdr_ratios]
        nt_r = float(self.nt_ratio)

        # CDR+SHM is additive: cdr + nt - 1, so the 1.0 baseline counts once, not twice
        self.region_multipliers = torch.tensor(
            [
                1.0,  # 0: FW, templated
                cdr1_r,  # 1: CDR1, templated
                cdr2_r,  # 2: CDR2, templated
                cdr3_r,  # 3: CDR3, templated
                nt_r,  # 4: FW, SHM
                cdr1_r + nt_r - 1.0,  # 5: CDR1, SHM
                cdr2_r + nt_r - 1.0,  # 6: CDR2, SHM
                cdr3_r + nt_r - 1.0,  # 7: CDR3, SHM
            ]
        )

    def torch_call(  # ty: ignore[invalid-method-override]  # narrows base's untyped Any examples
        self, examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        batch = _pad_paired_batch(examples, self.tokenizer, self.pad_to_multiple_of)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch["region_mask"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def torch_mask_tokens(  # ty: ignore[invalid-method-override]  # region-weighted signature
        self,
        inputs: torch.Tensor,
        region_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-top-k selection weighted by region, at an exact per-sequence count."""
        labels = inputs.clone()
        batch_size, seq_len = inputs.shape

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in labels.tolist()
                ],
                dtype=torch.bool,
                device=inputs.device,
            )
        else:
            special_tokens_mask = special_tokens_mask.bool().to(inputs.device)

        region_mask = region_mask.long()

        maskable = (region_mask >= 0) & ~special_tokens_mask

        multipliers = self.region_multipliers.to(inputs.device)
        weights = multipliers[region_mask.clamp(min=0)]  # clamp: -1 would index from the end
        weights[~maskable] = 0.0

        valid_counts = maskable.sum(dim=-1)
        # __post_init__ coerces mlm_probability to float, so this is Tensor * float at runtime
        mlm_probability: float = self.mlm_probability  # ty: ignore[invalid-assignment]
        n_mask = (valid_counts.float() * mlm_probability).round().long().clamp(min=0)

        # weighted sampling without replacement; fresh noise each call
        eps = 1e-10
        uniform = torch.rand(weights.shape, device=inputs.device).clamp(min=eps, max=1 - eps)
        gumbel_noise = -torch.log(-torch.log(uniform))
        scores = torch.log(weights.clamp(min=eps)) + gumbel_noise
        scores.masked_fill_(~maskable, float("-inf"))

        # rank every position, then take the top n_mask -- which varies per sequence
        _, sorted_indices = scores.sort(dim=-1, descending=True)
        position_ranks = torch.zeros(batch_size, seq_len, dtype=torch.long, device=inputs.device)
        position_ranks.scatter_(
            dim=-1,
            index=sorted_indices,
            src=torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1),
        )
        masked_indices = position_ranks < n_mask.unsqueeze(-1)

        labels[~masked_indices] = -100

        # 80% → <mask>
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool()
            & masked_indices
        )
        # mask_token is a single str -> scalar int id, broadcasts fine at runtime
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[indices_replaced] = mask_id  # ty: ignore[invalid-assignment]

        # 10% → random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device
        )
        inputs[indices_random] = random_words[indices_random]

        # Remaining 10%: keep original token unchanged
        return inputs, labels

"""Per-region eval metrics (CDR-level CE / accuracy) for region-weighted MLM.

Pairs with `RegionAwareCollator` / `WeightedMaskingCollator` (`ablm.training.data.collators`),
which write `region_mask` and `seq_mutated` onto each batch. `RegionEvalMixin` swaps in a
uniform eval collator and reduces logits to per-token CE/hits in
`prediction_step`, `compute_metrics` aggregates those by region. Ported from
`esm2/12_sota_convergence/training_mods/preferential_masking.py` in
ablm-sweeps (eval-metrics half of the region-weighted-masking subsystem;
originally from `esm2/05_preferential_masking_sweep/weighted_masking.py`'s
`WeightedMaskingTrainer`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .collators import TIERS

if TYPE_CHECKING:
    from transformers import EvalPrediction

__all__ = [
    "MaskingStatsMixin",
    "RegionEvalMixin",
    "compute_metrics",
    "masking_stats",
    "per_token_ce_and_hits",
]


def per_token_ce_and_hits(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce (B, L, V) logits to per-token CE (fp32) and top-1 hits (int8), both (B, L).

    Done in the eval step so logits are never accumulated: for eval-50k that is ~845 MB of bf16
    on GPU, which nested_numpify then upcasts to 1.7 GB on the host. Ignored positions come out 0.
    """
    logits = logits.float()
    token_ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none"
    ).view(labels.shape)
    token_hit = (logits.argmax(dim=-1) == labels).to(torch.int8)
    return token_ce, token_hit


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Per-region CE and top-1 accuracy over the eval-masked positions, off RegionEvalMixin's
    numpy arrays. A CDR level is regions {n, n+4}. CE_overall should equal HF's eval_loss, and the
    four levels partition every scored token -- so they stop summing to it if a position with
    region_mask < 0 was ever masked."""
    # RegionEvalMixin.prediction_step hands Trainer a dict, not the ndarray/tuple
    # EvalPrediction.predictions is typed for; Any here reflects that runtime shape.
    predictions: Any = eval_pred.predictions
    label_ids: Any = eval_pred.label_ids
    token_ce = predictions["ce"].ravel()
    region_mask = predictions["region"].ravel()
    token_hit = predictions["hit"].ravel()
    masked = label_ids.ravel() != -100

    def level(
        cdr: int,
    ) -> Any:  # exact match, as in 05: `region_mask % 4` aliases the -1 sentinel to CDR3
        return (region_mask == cdr) | (region_mask == cdr + 4)

    def region_stats(sel: Any = None) -> tuple[float, float]:
        active = masked if sel is None else masked & sel
        n = int(active.sum())
        if not n:
            return float("nan"), float("nan")
        return float(token_ce[active].sum()) / n, float(token_hit[active].sum()) / n

    slices: list[tuple[str, Any]] = [
        ("overall", None),
        ("non_cdr", level(0)),
        ("cdr1", level(1)),
        ("cdr2", level(2)),
        ("cdr3", level(3)),
        ("templated", (region_mask >= 0) & (region_mask < 4)),
        ("non_templated", region_mask >= 4),
    ]
    if "mutated" in predictions:
        mutated = predictions["mutated"].ravel()
        slices += [("seq_mutated", mutated == 1), ("seq_unmutated", mutated == 0)]

    metrics = {}
    for name, sel in slices:
        metrics[f"CE_{name}"], metrics[f"ACC_{name}"] = region_stats(sel)
    return metrics


class RegionEvalMixin:
    """Region-aware training + evaluation, for a Trainer paired with
    `RegionAwareCollator` / `WeightedMaskingCollator`. The collator carries `region_mask` on
    every batch; this mixin strips it before `model.forward` in training (`compute_loss`) and eval
    (`prediction_step`) -- the model does not accept it -- and additionally, in eval,
    swaps in `eval_data_collator` so masking is uniform for every arm however it trained
    (else eval/loss is not comparable) and reduces each eval step's logits to per-token
    CE + hits carried beside region_mask for compute_metrics.

    region_mask comes off the batch, not the model output, so model.py's arch class-swaps
    are untouched. Reducing in the step also puts it ahead of the cross-rank gather."""

    def __init__(self, *args: Any, eval_data_collator: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def get_eval_dataloader(self, eval_dataset: Any = None) -> Any:
        if getattr(self, "eval_data_collator", None) is None:
            return super().get_eval_dataloader(  # ty: ignore[unresolved-attribute]
                eval_dataset
            )  # mixin is always composed with Trainer
        orig = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(  # ty: ignore[unresolved-attribute]
                eval_dataset
            )  # mixin is always composed with Trainer
        finally:
            self.data_collator = orig

    def compute_loss(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        # The collators carry region_mask / seq_mutated on every batch, but only eval
        # (prediction_step) consumes it. Drop it in training so it never reaches
        # model.forward, which does not accept it. `None` default: eval's prediction_step
        # already popped it before compute_loss runs, so this is a no-op there.
        inputs.pop("region_mask", None)
        inputs.pop("seq_mutated", None)
        return super().compute_loss(model, inputs, *args, **kwargs)  # ty: ignore[unresolved-attribute]

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> Any:
        # no default: a KeyError beats silently returning raw logits
        region_mask = inputs.pop("region_mask")
        seq_mutated = inputs.pop("seq_mutated", None)
        loss, logits, labels = super().prediction_step(  # ty: ignore[unresolved-attribute]
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )  # mixin is always composed with Trainer
        if prediction_loss_only:
            return loss, logits, labels
        token_ce, token_hit = per_token_ce_and_hits(logits, labels)
        # int8 is safe and 8x smaller: codes are -1..7, and nested_concat pads with -100
        out = {
            "ce": token_ce,
            "region": region_mask.to(token_ce.device, torch.int8),
            "hit": token_hit,
        }
        if seq_mutated is not None:
            out["mutated"] = (
                seq_mutated.view(-1, 1).expand_as(token_hit).to(token_ce.device, torch.int8)
            )
        return loss, out, labels


def masking_stats(labels: torch.Tensor, region_mask: torch.Tensor, p: float) -> dict[str, float]:
    """Realised masking of one batch: the statistics that tell the count modes apart.

    Args:
        labels: ``(B, L)`` MLM labels, ``-100`` where not scored.
        region_mask: ``(B, L)`` region codes, ``-1`` where not maskable.
        p: Nominal masking probability.

    Returns:
        ``mask_rate`` (masked / maskable), ``count_z_sd`` (sd over sequences of
        ``(k - n p) / sqrt(n p (1 - p))``: ~1 for a binomial count, ~0 for exact-count) and
        ``rate_<tier>`` for each ``TIERS`` entry, whose ratio to ``rate_fw_templated`` equals the
        weight ratio under ``CountMode.BERNOULLI``.
    """
    maskable = region_mask >= 0
    selected = (labels != -100) & maskable
    n = maskable.sum(dim=-1).float()
    k = selected.sum(dim=-1).float()
    keep = n > 0
    z = (k[keep] - n[keep] * p) / torch.sqrt(n[keep] * p * (1 - p))
    stats = {
        "mask_rate": (k.sum() / n.sum().clamp(min=1)).item(),
        "count_z_sd": z.std().item() if z.numel() > 1 else float("nan"),
    }
    for name, codes in TIERS.items():
        in_tier = torch.isin(region_mask, torch.tensor(codes, device=region_mask.device))
        stats[f"rate_{name}"] = ((selected & in_tier).sum() / in_tier.sum().clamp(min=1)).item()
    return stats


class MaskingStatsMixin:
    """Log ``masking_stats`` of the latest training micro-batch under ``mask/*``.

    Under gradient accumulation only the last micro-batch of each step is stashed.

    Computed in the main process from the batch itself: the collator runs in DataLoader
    workers, so anything it stashes on itself never reaches a callback. Mix in ahead of
    ``Trainer`` (and ahead of ``RegionEvalMixin``, which strips the side channels later).
    """

    def training_step(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if "region_mask" in inputs and "labels" in inputs:
            self._mask_batch = (inputs["labels"].detach(), inputs["region_mask"].detach())
        return super().training_step(model, inputs, *args, **kwargs)  # ty: ignore[unresolved-attribute]

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> Any:
        batch = getattr(self, "_mask_batch", None)
        if batch is not None and "loss" in logs:
            p = float(self.data_collator.mlm_probability)  # ty: ignore[unresolved-attribute]
            logs = {
                **logs,
                **{f"mask/{k}": v for k, v in masking_stats(batch[0], batch[1], p).items()},
            }
        return super().log(logs, *args, **kwargs)  # ty: ignore[unresolved-attribute]

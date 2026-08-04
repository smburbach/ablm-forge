"""Per-region eval metrics (CDR-level CE / accuracy) for region-weighted MLM.

Pairs with `ablm.data.PreferentialMaskingCollator`: `RegionEvalMixin` swaps in a
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

if TYPE_CHECKING:
    from transformers import EvalPrediction

__all__ = ["RegionEvalMixin", "compute_metrics", "per_token_ce_and_hits"]


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

    metrics = {}
    for name, sel in (
        ("overall", None),
        ("non_cdr", level(0)),
        ("cdr1", level(1)),
        ("cdr2", level(2)),
        ("cdr3", level(3)),
    ):
        metrics[f"CE_{name}"], metrics[f"ACC_{name}"] = region_stats(sel)
    return metrics


class RegionEvalMixin:
    """Region-aware evaluation, for a Trainer: swap in `eval_data_collator` so masking is uniform
    for every arm however it trained (else eval/loss is not comparable), and reduce each eval
    step's logits to per-token CE + hits, carried beside region_mask for compute_metrics.

    region_mask comes off the eval batch, not the model output, so model.py's arch class-swaps are
    untouched. Reducing in the step also puts it ahead of the cross-rank gather."""

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

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> Any:
        # no default: a KeyError beats silently returning raw logits
        region_mask = inputs.pop("region_mask")
        loss, logits, labels = super().prediction_step(  # ty: ignore[unresolved-attribute]
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )  # mixin is always composed with Trainer
        if prediction_loss_only:
            return loss, logits, labels
        token_ce, token_hit = per_token_ce_and_hits(logits, labels)
        # int8 is safe and 8x smaller: codes are -1..7, and nested_concat pads with -100
        return (
            loss,
            {
                "ce": token_ce,
                "region": region_mask.to(token_ce.device, torch.int8),
                "hit": token_hit,
            },
            labels,
        )

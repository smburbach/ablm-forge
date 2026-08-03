"""Muon optimizer (2D transformer-body weights) + AdamW (everything else).

HF-native optimizers are selected directly via `TrainingArguments.optim`. Muon is
the one HF doesn't ship: `build_muon_optimizer` splits the model by name and wraps
`DistributedMuon` (on the 2D attention/MLP body weights) and `torch.optim.AdamW`
(embeddings, LM head, norms, biases, and the muP-scaled hidden AdamW matrices) in a
single `CombinedOptimizer`. Build it in the training script and hand it to the stock
`Trainer` via `optimizers=(opt, None)` -- no `Trainer` subclass; the LR scheduler
still comes from `TrainingArguments`.

`DistributedMuon` (see `distributed_muon.py`) shards the Newton-Schulz
orthogonalization across the DDP group -- each rank orthogonalizes only its slice
and the results are gathered back -- numerically identical to `torch.optim.Muon` but
without the redundant per-rank compute. It assumes DDP (replicated params), which is
how forge trains (`accelerate --multi_gpu`); it is not FSDP-aware, so AdamW is the
optimizer for any sharded (FSDP) setup.

`CombinedOptimizer` serializes in the standard flat layout, so checkpoint
save/resume round-trips on a single GPU and under DDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .distributed_muon import DistributedMuon

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

__all__ = [
    "MUON_OPTIM",
    "MUON_PARAM_PREFIX",
    "CombinedOptimizer",
    "build_muon_optimizer",
    "split_muon_params",
]

# Sentinel for the `--optimizer` choice; not a valid HF OptimizerNames value.
MUON_OPTIM = "muon"

# torch.optim.Muon takes only 2D hidden weights; this prefix scopes it to the
# transformer body (attention + MLP) on AblmForMaskedLM. Embeddings, the LM head,
# norms and biases fall through to AdamW. (cf. ESM-2's `esm.encoder.layer.`)
MUON_PARAM_PREFIX = "ablm.backbone.layers."


def split_muon_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    """Partition trainable params into ``(muon_params, adam_params, adam_names)``.

    Muon gets the 2D transformer-body weights (name under `MUON_PARAM_PREFIX`);
    everything else (embeddings, LM head, norms, biases) goes to AdamW. `adam_names`
    is positionally aligned with `adam_params` so callers can apply HF's name-based
    weight-decay rule. Asserts the split is exhaustive and that no embedding /
    LM-head weight leaks into Muon, so a module rename fails loudly.
    """
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
    adam_names: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name.startswith(MUON_PARAM_PREFIX):
            assert "embed" not in name and not name.startswith("lm_head"), (
                f"embedding/lm_head weight leaked into the Muon group: {name}"
            )
            muon_params.append(p)
        else:
            adam_params.append(p)
            adam_names.append(name)

    assert muon_params, "Muon group is empty -- did the transformer-body module names change?"
    return muon_params, adam_params, adam_names


class CombinedOptimizer(torch.optim.Optimizer):
    """Makes several sub-optimizers look like one, since HF Trainer expects a single one.

    `param_groups` concatenates the children's real group dicts *by reference* (children in
    order, so the Muon group's params get the first param-ids), so an LR scheduler writing
    `group["lr"]` mutates them in place. `Optimizer.__init__` is deliberately not called (the
    children own the params/state); `isinstance(.., Optimizer)` still holds.

    `state_dict` / `load_state_dict` emit and consume the **standard** flat layout
    (`{"state": {pid: ...}, "param_groups": [...]}`), so both the normal `optimizer.pt` path
    and torch's distributed-checkpoint path (FSDP) round-trip. Load splits the merged dict
    back to each child positionally — param-ids are matched by order, exactly as
    `torch.optim.Optimizer.load_state_dict` does, so FQN-keyed dicts (what FSDP DSD hands
    back) work too.
    """

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)
        assert self.optimizers, "CombinedOptimizer needs at least one sub-optimizer"
        self.defaults = self.optimizers[0].defaults

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    @param_groups.setter
    def param_groups(self, _value: Any) -> None:
        pass  # groups live on the sub-optimizers; ignore replacement writes

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(  # ty: ignore[invalid-method-override]  # facade matches Optimizer.step at runtime
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        # Standard layout: param-ids index the concatenated param_groups (Muon group first),
        # state keyed by those ids. Mirrors torch.optim.Optimizer.state_dict (minus its hook
        # machinery, which our facade doesn't own since it skips Optimizer.__init__).
        pid_of: dict[int, int] = {}
        packed_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            ids = []
            for p in group["params"]:
                pid_of.setdefault(id(p), len(pid_of))
                ids.append(pid_of[id(p)])
            packed["params"] = ids
            packed_groups.append(packed)
        packed_state = {
            (pid_of[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()
        }
        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Slice the merged dict per child by param-group count and let each child's own
        # load_state_dict match params positionally (works for int- or FQN-keyed dicts).
        saved_state = state_dict["state"]
        saved_groups = state_dict["param_groups"]
        g_start = 0
        for opt in self.optimizers:
            n_groups = len(opt.param_groups)
            child_groups = saved_groups[g_start : g_start + n_groups]
            keys = [pid for g in child_groups for pid in g["params"]]
            child_state = {k: saved_state[k] for k in keys if k in saved_state}
            opt.load_state_dict({"state": child_state, "param_groups": child_groups})
            g_start += n_groups


def build_muon_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.95,
    adjust_lr_fn: str | None = "match_rms_adamw",
    muon_weight_decay: float | None = None,
    decay_parameter_names: set[str] | None = None,
) -> CombinedOptimizer:
    """Build Muon (2D body weights) + AdamW (everything else) as one optimizer.

    `adjust_lr_fn="match_rms_adamw"` rescales the Muon update to AdamW's RMS
    (Moonshot 2025), so AdamW's tuned `lr` / `weight_decay` transfer to Muon --
    which is why `weight_decay` applies to BOTH children by default. Pass
    `muon_weight_decay` to decouple them.

    `decay_parameter_names` should be `Trainer.get_decay_parameter_names(model)`, so
    the decay / no-decay split matches HF's convention. When it is `None` the fallback
    is `p.ndim >= 2`, which is exact for this architecture (every 2D param is a weight,
    every 1D param a norm/bias) -- so callers can leave it `None`. `betas` / `eps`
    default to HF `TrainingArguments` defaults.

    Requires torch >= 2.11 (`DistributedMuon` reuses `torch.optim._muon` internals).
    """
    muon_params, adam_params, adam_names = split_muon_params(model)
    if decay_parameter_names is None:
        decay = [p for p in adam_params if p.ndim >= 2]
        no_decay = [p for p in adam_params if p.ndim < 2]
    else:
        pairs = list(zip(adam_names, adam_params, strict=True))
        decay = [p for n, p in pairs if n in decay_parameter_names]
        no_decay = [p for n, p in pairs if n not in decay_parameter_names]

    cfg = getattr(model, "config", None)
    mup_group = None
    if cfg is not None and getattr(cfg, "mup_enabled", False):
        # muP: every hidden AdamW 2D weight (readout, MLM-head dense, classifier, ...)
        # has fan_in proportional to width and needs Adam LR ~ 1/d, scaled here by
        # `d0/d` (`mup_adamw_lr_mult`). The input embedding is Theta(1) LR and is
        # excluded. `get_input_embeddings` (not `get_output_embeddings`, which is
        # `None` on classification/token heads) exists on every Ablm* head.
        # ty: model is typed nn.Module; get_input_embeddings is a PreTrainedModel method.
        input_emb = model.get_input_embeddings().weight  # ty: ignore[call-non-callable]
        mup_params = [p for p in decay if p.ndim == 2 and p is not input_emb]
        decay = [p for p in decay if not (p.ndim == 2 and p is not input_emb)]
        mup_group = {
            "params": mup_params,
            "weight_decay": weight_decay,
            "lr": lr * cfg.mup_adamw_lr_mult,
        }

    adamw_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if mup_group is not None:
        adamw_groups.append(mup_group)
    return CombinedOptimizer(
        [
            DistributedMuon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay if muon_weight_decay is None else muon_weight_decay,
                momentum=momentum,
                adjust_lr_fn=adjust_lr_fn,
            ),
            torch.optim.AdamW(adamw_groups, lr=lr, betas=betas, eps=eps),
        ]
    )

"""Muon optimizer (2D transformer-body weights) + AdamW (everything else).

HF-native optimizers are selected directly via `TrainingArguments.optim`. Muon is
the one HF doesn't ship: `build_muon_optimizer` wraps `torch.optim.Muon` (on the
2D attention/MLP weights) and `torch.optim.AdamW` (embeddings, LM head, norms,
biases) in a single `CombinedOptimizer`.

`OptimizerTrainer` is a thin `Trainer` subclass that overrides only
`create_optimizer` to build Muon. The subclass is required, not a preference: HF
*forbids* passing a pre-built `optimizers=` tuple once FSDP is enabled
("Passing `optimizers` is not allowed if PyTorch FSDP is enabled. You should
subclass `Trainer` and override ...") and `optimizer_cls_and_kwargs` can't express
the name-based Muon/AdamW split (HF hands it grouped tensors, not names). The
override builds the optimizer *after* the model is FSDP-sharded. With Muon off it
is exactly the stock Trainer; there is no custom training loop.

`CombinedOptimizer` serializes in the standard flat layout, so checkpoint
save/resume round-trips on a single GPU / DDP *and* through torch's
distributed-checkpoint path under FSDP2. The remaining FSDP caveat is numerical,
not serialization: `torch.optim.Muon` runs Newton-Schulz on whatever tensor it is
handed, so on a sharded 2D weight the orthogonalization is per-shard and only
correct if DTensor redistributes it. Validate Muon on multiple GPUs before
trusting it; AdamW is the safe FSDP default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from transformers import Trainer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

__all__ = [
    "MUON_OPTIM",
    "MUON_PARAM_PREFIX",
    "CombinedOptimizer",
    "OptimizerTrainer",
    "build_muon_optimizer",
    "split_muon_params",
]

# Sentinel for the `--optimizer` choice; not a valid HF OptimizerNames value.
MUON_OPTIM = "muon"

# torch.optim.Muon takes only 2D hidden weights; this prefix scopes it to the
# transformer body (attention + MLP) on AblmForMaskedLM. Embeddings, the LM head,
# norms and biases fall through to AdamW. (cf. ESM-2's `esm.encoder.layer.`)
MUON_PARAM_PREFIX = "ablm.backbone.layers."


def split_muon_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable params into ``(muon_params, adam_params)``.

    Muon gets the 2D transformer-body weights (name under `MUON_PARAM_PREFIX`);
    everything else (embeddings, LM head, norms, biases) goes to AdamW. Asserts the
    split is exhaustive and that no embedding / LM-head weight leaks into Muon, so a
    module rename fails loudly.
    """
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
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

    assert muon_params, "Muon group is empty -- did the transformer-body module names change?"
    return muon_params, adam_params


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
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
    momentum: float = 0.95,
    adjust_lr_fn: str | None = "match_rms_adamw",
) -> CombinedOptimizer:
    """Build Muon (2D body weights) + AdamW (everything else) as one optimizer.

    `adjust_lr_fn="match_rms_adamw"` rescales the Muon update to AdamW's RMS
    (Moonshot 2025), so AdamW's tuned `lr` / `weight_decay` transfer to Muon. AdamW
    decays only the 2D weights (embeddings, LM head); norms and biases are excluded,
    matching the HF/ESM convention. Requires torch >= 2.11 (`torch.optim.Muon`).
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is unavailable; 'muon' requires torch >= 2.11.")

    muon_params, adam_params = split_muon_params(model)
    decay = [p for p in adam_params if p.ndim >= 2]
    no_decay = [p for p in adam_params if p.ndim < 2]
    return CombinedOptimizer(
        [
            torch.optim.Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                adjust_lr_fn=adjust_lr_fn,
            ),
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=betas,
                eps=eps,
            ),
        ]
    )


class OptimizerTrainer(Trainer):
    """Stock `transformers.Trainer` plus an optional Muon optimizer.

    With `use_muon=False` (the default) this is exactly the stock Trainer — every
    HF-native optimizer flows through `TrainingArguments.optim` untouched. With
    `use_muon=True`, `create_optimizer` builds Muon (+ aux AdamW) from the model after
    it is placed / FSDP-sharded. Overriding `create_optimizer` is mandatory for FSDP:
    HF rejects a pre-built `optimizers=` tuple once FSDP is on. No training loop is
    overridden — `lr`/`weight_decay`/scheduler all still come from `TrainingArguments`.
    """

    def __init__(self, *args: Any, use_muon: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._use_muon = use_muon

    def create_optimizer(self, model: nn.Module | None = None) -> torch.optim.Optimizer:
        if self.optimizer is not None or not self._use_muon:
            return super().create_optimizer(model)
        opt_model = model if model is not None else self.model
        assert opt_model is not None, "OptimizerTrainer.create_optimizer needs a model"
        self.optimizer = build_muon_optimizer(
            opt_model, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return self.optimizer

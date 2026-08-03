"""Muon with Newton-Schulz orthogonalization sharded across the data-parallel group.

`torch.optim.Muon` runs the (expensive) Newton-Schulz orthogonalization on every
DDP rank for every 2D parameter -- so on N GPUs the work is done N times over.
`DistributedMuon` assigns each parameter to one owner rank, each rank orthogonalizes
only its slice, and the updated parameters are gathered back so all ranks stay in
sync. The per-parameter update reuses torch's own NS kernel and LR adjustment, so the
result is numerically identical to `torch.optim.Muon`; only the redundant compute is
removed. With no distributed group it degrades to plain Muon.

This assumes **DDP semantics**: every rank holds the full 2D parameter and gradients
are already averaged across ranks (DDP does this before `step()`), so every rank
computes the same update for any given parameter. It is *not* FSDP-aware -- under
FSDP each rank holds only a shard, which Newton-Schulz cannot orthogonalize. forge
trains under DDP (`accelerate --multi_gpu`); AdamW is the optimizer for any sharded
(FSDP) setup.

Communication follows the chunked `all_gather` pattern from the `Muon.step` of Keller
Jordan's reference implementation (MIT License):
https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py
Parameters are processed in chunks of `world_size`, each rank computes one slot, and a
single async `all_gather` per chunk writes results straight into the param tensors --
far fewer, batched collectives than a per-parameter broadcast, with comm overlapping
the next chunk's compute. Chunks are formed within same-shape groups so every
`all_gather` is uniform. The per-parameter update math is unchanged from
`torch.optim.Muon` (not from the reference, whose LR scaling differs).

Requires torch >= 2.11 (reuses `torch.optim._muon` internals).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
from torch.optim._muon import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
    _adjust_lr,
    _zeropower_via_newtonschulz,
)

__all__ = ["DistributedMuon"]


class DistributedMuon(torch.optim.Optimizer):
    """Drop-in replacement for `torch.optim.Muon` that shards NS across DDP ranks.

    Gradients are assumed already averaged across ranks (DDP does this before
    `step()`), so every rank computes the same update for any given parameter --
    which is what lets one rank own each parameter and the rest receive the result.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Unsupported adjust_lr_fn: {adjust_lr_fn}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    def _apply_update(self, p, group):
        """In-place Muon update for one param -- identical to torch.optim.Muon."""
        grad = p.grad
        if grad.ndim != 2:
            raise ValueError("Muon only supports 2D parameters")
        state = self.state[p]
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = torch.zeros_like(grad, memory_format=torch.preserve_format)
            state["momentum_buffer"] = buf
        momentum = group["momentum"]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if group["nesterov"] else buf
        update = _zeropower_via_newtonschulz(
            update, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        adjusted_lr = _adjust_lr(group["lr"], group["adjust_lr_fn"], p.shape)
        p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update, alpha=-adjusted_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]

            if world_size == 1:
                for p in params:
                    self._apply_update(p, group)
                continue

            # Same-shape groups so each all_gather chunk is uniform; within a group,
            # chunk by world_size and pad the tail with throwaway buffers.
            by_shape = defaultdict(list)
            for p in params:
                by_shape[tuple(p.shape)].append(p)

            # chunked all_gather, adapted from Keller Jordan's Muon.step (see module docstring)
            handles = []
            for shape, plist in by_shape.items():
                pad = (-len(plist)) % world_size
                padded = plist + [plist[-1].new_empty(shape) for _ in range(pad)]
                for base in range(0, len(padded), world_size):
                    chunk = padded[base : base + world_size]
                    if base + rank < len(plist):
                        self._apply_update(chunk[rank], group)
                    # each rank contributes its slot; output overwrites the whole chunk
                    handles.append(dist.all_gather(chunk, chunk[rank], async_op=True))
            for h in handles:
                h.wait()

        return loss

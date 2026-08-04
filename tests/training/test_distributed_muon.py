"""DistributedMuon: single-process updates are bit-identical to torch.optim.Muon.

The distributed path (sharded Newton-Schulz across ranks) can't run in a single
pytest process; what we pin here is the numerical contract that makes the port
safe -- with no process group, DistributedMuon degrades to plain Muon and reuses
torch's own NS kernel / LR adjust, so every update matches `torch.optim.Muon`.
"""

from __future__ import annotations

import pytest
import torch

from ablm.training.distributed_muon import DistributedMuon

pytestmark = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="DistributedMuon reuses torch.optim._muon internals (torch >= 2.11)",
)


@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_single_process_matches_torch_muon(adjust_lr_fn):
    torch.manual_seed(0)
    w0 = torch.randn(8, 16)
    grads = [torch.randn(8, 16) for _ in range(3)]

    ref_p = w0.clone().requires_grad_(True)
    dist_p = w0.clone().requires_grad_(True)
    kw = dict(lr=1e-2, weight_decay=0.1, momentum=0.95, adjust_lr_fn=adjust_lr_fn)
    ref = torch.optim.Muon([ref_p], **kw)
    dist = DistributedMuon([dist_p], **kw)

    for g in grads:  # multiple steps exercise the momentum buffer
        ref_p.grad = g.clone()
        dist_p.grad = g.clone()
        ref.step()
        dist.step()
        assert torch.equal(ref_p, dist_p), (ref_p - dist_p).abs().max().item()


def test_rejects_non_2d_parameter():
    p = torch.zeros(4, requires_grad=True)
    p.grad = torch.ones(4)
    opt = DistributedMuon([p], lr=1e-3)
    with pytest.raises(ValueError, match="2D"):
        opt.step()


def test_rejects_unknown_adjust_lr_fn():
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        DistributedMuon([torch.zeros(2, 2, requires_grad=True)], adjust_lr_fn="bogus")

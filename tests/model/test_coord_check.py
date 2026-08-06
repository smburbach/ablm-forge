"""muP coordinate check: activation RMS is ~width-invariant.

Two levels of CI proof that the preset ladder is muP-correct, both CPU-cheap:

* **at init** -- fan-in init keeps activation RMS flat across width (validates the
  init side of muP);
* **under optimizer steps** -- after a few Muon+AdamW steps, activation RMS *still*
  doesn't grow with width. This is the level that actually exercises the Muon LR
  rule: `build_muon_optimizer` uses `original` under muP, and the contrast test
  shows that swapping in `match_rms_adamw` (~sqrt(width)) makes it grow. An init-only
  check cannot see this class of bug.
"""

from __future__ import annotations

import pytest
import torch

from ablm import AblmConfig, AblmForMaskedLM

_WIDTHS = (128, 256, 512)  # 4x range
_BASE = 128
_HAS_MUON = hasattr(torch.optim, "Muon")


def _hidden_rms(hidden: int, *, mup: bool) -> float:
    kw = dict(mup_enabled=True, mup_base_hidden_size=_BASE) if mup else {}
    cfg = AblmConfig(
        hidden_size=hidden, num_attention_heads=hidden // 64, num_hidden_layers=2, **kw
    )
    torch.manual_seed(0)
    model = AblmForMaskedLM(cfg).eval()
    ids = torch.randint(4, 33, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    with torch.no_grad():
        out = model.ablm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-1]  # last block output, before final_norm
    return h.float().pow(2).mean().sqrt().item()


@pytest.mark.slow
def test_mup_activation_rms_flat_across_width():
    rms = [_hidden_rms(w, mup=True) for w in _WIDTHS]
    assert min(rms) > 0
    spread = max(rms) / min(rms)
    assert spread < 2.5, (
        f"muP activation RMS not width-stable: {dict(zip(_WIDTHS, rms, strict=True))}"
    )


@pytest.mark.slow
def test_mup_is_flatter_than_fixed_init_baseline():
    mup = [_hidden_rms(w, mup=True) for w in _WIDTHS]
    base = [_hidden_rms(w, mup=False) for w in _WIDTHS]
    assert (max(mup) / min(mup)) <= (max(base) / min(base)) + 1e-6


def _hidden_rms_after_steps(hidden: int, *, muon_rule: str | None = None, steps: int = 6) -> float:
    """Train a tiny muP model for `steps` Muon+AdamW steps, then measure last-block
    pre-final-norm hidden RMS. `muon_rule` overrides the Muon child's adjust_lr_fn on
    the (correct, muP fan-in) init, isolating the LR rule from init scaling."""
    from ablm.training.optim import build_muon_optimizer

    cfg = AblmConfig(
        hidden_size=hidden,
        num_attention_heads=hidden // 64,
        num_hidden_layers=2,
        mup_enabled=True,
        mup_base_hidden_size=_BASE,
    )
    torch.manual_seed(0)
    model = AblmForMaskedLM(cfg).train()
    opt = build_muon_optimizer(model, lr=1e-2)  # muP -> Muon uses "original"
    if muon_rule is not None:
        opt.optimizers[0].param_groups[0]["adjust_lr_fn"] = muon_rule

    torch.manual_seed(1)
    ids = torch.randint(4, 33, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = model.ablm(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[
            -1
        ]
    return h.float().pow(2).mean().sqrt().item()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MUON, reason="needs torch.optim.Muon (torch >= 2.11)")
def test_mup_activation_rms_flat_after_optimizer_steps():
    """With muP + Muon's 'original' rule, activation RMS stays width-stable after a few
    training steps -- the actual LR-transfer property. Regressing build_muon_optimizer to
    match_rms_adamw under muP would make this grow and fail."""
    rms = [_hidden_rms_after_steps(w) for w in _WIDTHS]
    spread = max(rms) / min(rms)
    assert spread < 2.5, (
        f"muP post-step RMS not width-stable: {dict(zip(_WIDTHS, rms, strict=True))}"
    )


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MUON, reason="needs torch.optim.Muon (torch >= 2.11)")
def test_match_rms_adamw_muon_rule_grows_with_width_under_steps():
    """Teeth: on the *same* muP init, swapping the Muon rule to match_rms_adamw (which
    scales ~sqrt(width)) makes post-step RMS grow with width more than 'original' does."""
    original = [_hidden_rms_after_steps(w) for w in _WIDTHS]
    match_rms = [_hidden_rms_after_steps(w, muon_rule="match_rms_adamw") for w in _WIDTHS]
    assert (max(match_rms) / min(match_rms)) > (max(original) / min(original)), (
        f"match_rms_adamw should be less width-stable: original={original}, match_rms={match_rms}"
    )

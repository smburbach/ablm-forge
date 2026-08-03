"""muP forward multipliers on embedding output and MLM logits."""

from __future__ import annotations

import torch

from ablm import AblmConfig, AblmForMaskedLM


def _model(**mup) -> AblmForMaskedLM:
    cfg = AblmConfig(hidden_size=256, num_attention_heads=4, num_hidden_layers=2, **mup)
    return AblmForMaskedLM(cfg).eval()


def test_embedding_multiplier_scales_embedding_output():
    ids = torch.randint(4, 33, (2, 8))
    base = _model()
    scaled = _model(mup_enabled=True, mup_base_hidden_size=256, mup_emb_mult=3.0)
    scaled.load_state_dict(base.state_dict())  # identical weights
    with torch.no_grad():
        e_base = base.ablm.backbone.embed_tokens(ids)
        e_scaled = scaled.ablm.backbone.embed_tokens(ids)
    assert torch.allclose(e_scaled, 3.0 * e_base, atol=1e-5)


def test_output_multiplier_scales_logits():
    # hidden 512, base 256 -> output_mult = 0.5
    ids = torch.randint(4, 33, (2, 8))
    base = AblmForMaskedLM(
        AblmConfig(hidden_size=512, num_attention_heads=8, num_hidden_layers=2)
    ).eval()
    mup = AblmForMaskedLM(
        AblmConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=2,
            mup_enabled=True,
            mup_base_hidden_size=256,
        )
    ).eval()
    mup.load_state_dict(base.state_dict())
    mask = torch.ones(2, 8, dtype=torch.long)
    with torch.no_grad():
        lb = base(input_ids=ids, attention_mask=mask).logits
        lm = mup(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(lm, 0.5 * lb, atol=1e-4)

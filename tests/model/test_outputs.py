"""Tests for `ablm.model.outputs` — LogitsConfig / LogitsOutput dataclasses."""

from __future__ import annotations

import torch

from ablm import AblmConfig, AblmForMaskedLM
from ablm.model import LogitsConfig, LogitsOutput


def test_logits_config_defaults_match_spec():
    cfg = LogitsConfig()
    assert cfg.sequence is True
    assert cfg.return_embeddings is False
    assert cfg.return_hidden_states is False
    assert cfg.return_attentions is False


def test_logits_config_is_mutable():
    cfg = LogitsConfig()
    cfg.return_embeddings = True
    assert cfg.return_embeddings is True


def test_logits_config_keyword_overrides():
    cfg = LogitsConfig(sequence=False, return_attentions=True)
    assert cfg.sequence is False
    assert cfg.return_attentions is True
    # Untouched fields keep their defaults.
    assert cfg.return_embeddings is False
    assert cfg.return_hidden_states is False


def test_logits_output_accepts_all_none():
    out = LogitsOutput(
        sequence_logits=None,
        embeddings=None,
        hidden_states=None,
        attentions=None,
    )
    assert out.sequence_logits is None
    assert out.embeddings is None
    assert out.hidden_states is None
    assert out.attentions is None


def test_logits_output_preserves_tensor_identity():
    logits = torch.zeros(2, 4, 33)
    hidden = torch.zeros(2, 4, 8)
    attn = torch.zeros(2, 2, 4, 4)
    out = LogitsOutput(
        sequence_logits=logits,
        embeddings=hidden,
        hidden_states=(hidden, hidden),
        attentions=(attn,),
    )
    assert out.sequence_logits is logits
    assert out.embeddings is hidden
    assert out.hidden_states[0] is hidden
    assert out.attentions[0] is attn


def test_mlm_head_activation_is_an_nn_module():
    m = AblmForMaskedLM(AblmConfig(hidden_size=16, num_attention_heads=2, num_hidden_layers=1))
    assert isinstance(m.lm_head.act, torch.nn.Module)


def test_mlm_head_activation_changes_the_logits():
    ids = torch.randint(4, 30, (1, 12))
    kw = dict(vocab_size=33, hidden_size=32, num_attention_heads=4, num_hidden_layers=2)
    outs = {}
    for act in ("gelu", "silu", "relu"):
        torch.manual_seed(0)
        model = AblmForMaskedLM(AblmConfig(**kw, mlm_head_activation=act)).eval()
        with torch.no_grad():
            outs[act] = model(input_ids=ids).logits
    assert not torch.allclose(outs["gelu"], outs["silu"])
    assert not torch.allclose(outs["gelu"], outs["relu"])

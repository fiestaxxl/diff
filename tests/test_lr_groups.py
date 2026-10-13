"""Per-group learning rates: a multiplier that the training loop actually honours.

The loop writes the scheduled learning rate onto every parameter group at every optimizer
step. Before this, it wrote the same number to all of them, so a group created with its own
rate was reset on the first step and every step after, and a config asking for a fine-tuned
encoder at a tenth of the rate got the full rate in silence. These tests pin both halves:
the groups carry the multiplier, and the loop applies it rather than overwriting it.
"""
from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

from dimol.training.optim import param_groups


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.cross_attn = nn.Linear(4, 4)
        self.token_embedding = nn.Embedding(8, 4)


def test_without_a_multiplier_the_grouping_is_unchanged():
    groups = param_groups(_Model(), weight_decay=0.1,
                          no_decay_patterns=["token_embedding"], verbose=False)
    assert len(groups) == 2
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    assert all(g["lr_mult"] == 1.0 for g in groups)
    # the embedding is excluded from decay even though it is two-dimensional
    assert any(p.shape == (8, 4) for p in groups[1]["params"])


def test_a_pattern_splits_the_groups_and_carries_its_multiplier():
    groups = param_groups(_Model(), weight_decay=0.1,
                          no_decay_patterns=["token_embedding"],
                          lr_mult_patterns={"text_encoder": 0.1}, verbose=False)
    by_mult = {}
    for g in groups:
        by_mult.setdefault(g["lr_mult"], []).extend(g["params"])
    assert set(by_mult) == {1.0, 0.1}
    encoder = _Model().text_encoder
    assert sum(p.numel() for p in by_mult[0.1]) == sum(p.numel()
                                                       for p in encoder.parameters())


def test_every_parameter_lands_in_exactly_one_group():
    model = _Model()
    groups = param_groups(model, weight_decay=0.1,
                          lr_mult_patterns={"text_encoder": 0.1, "cross_attn": 0.5},
                          verbose=False)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_the_first_matching_pattern_wins():
    groups = param_groups(_Model(), weight_decay=0.0,
                          lr_mult_patterns={"text_encoder.0": 0.01, "text_encoder": 0.1},
                          verbose=False)
    mults = sorted({g["lr_mult"] for g in groups})
    assert mults == [0.01, 0.1, 1.0]


def test_the_training_loop_multiplies_rather_than_overwrites():
    """The regression itself, read from the source: the loop must not write a bare lr."""
    import dimol.training.trainer as trainer_mod

    src = pathlib.Path(trainer_mod.__file__).read_text().splitlines()
    assigns = [ln.strip() for ln in src if ln.strip().startswith('param_group["lr"]')]
    assert assigns, "the optimizer step no longer sets a learning rate at all"
    for line in assigns:
        assert "lr_mult" in line, f"this assignment drops the multiplier: {line}"

    # and the arithmetic it is meant to do
    opt = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(2))], "lr_mult": 1.0},
        {"params": [nn.Parameter(torch.zeros(2))], "lr_mult": 0.1},
    ], lr=1.0)
    for g in opt.param_groups:
        g["lr"] = 3e-4 * float(g.get("lr_mult", 1.0))
    got = [g["lr"] for g in opt.param_groups]
    assert got[0] == pytest.approx(3e-4) and got[1] == pytest.approx(3e-5)


# ------------------------------------- the same mechanism against real parameter names

HIDDEN = 32


def _real_denoiser(tmp_path, layers: int = 12, blocks: int = -1):
    """The actual model, not a toy nn.Sequential: the patterns are matched by substring."""
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertModel

    from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig

    path = tmp_path / "enc"
    BertModel(BertConfig(
        vocab_size=48, hidden_size=HIDDEN, num_hidden_layers=layers,
        num_attention_heads=4, intermediate_size=2 * HIDDEN,
        max_position_embeddings=32), add_pooling_layer=False).save_pretrained(path)
    return DiffusionTransformer(TransformerConfig(
        model_dim=HIDDEN, emb_dim=8, time_dim=HIDDEN, num_heads=4, num_text_blocks=2,
        vocab_size=64, pad_idx=0, max_pos=16, text_dim=HIDDEN, text_global=True,
        text_encoder=str(path), text_encoder_trainable_blocks=blocks))


def _names_by_mult(model, groups) -> dict:
    name_of = {id(p): n for n, p in model.named_parameters()}
    out: dict = {}
    for group in groups:
        out.setdefault(group["lr_mult"], set()).update(
            name_of[id(p)] for p in group["params"])
    return out


def test_the_encoder_pattern_slows_the_encoder_and_nothing_else(tmp_path):
    """The denoiser's own text modules — text_transformer_blocks, text_rope_embeddings,
    text_global_proj — all contain "text", so a looser pattern would put the entire
    backbone on a fifteenth of the rate without a word in any log."""
    model = _real_denoiser(tmp_path)
    buckets = _names_by_mult(model, param_groups(
        model, weight_decay=0.1, no_decay_patterns=["token_embedding"],
        lr_mult_patterns={"text_encoder.": 0.067}, verbose=False))

    assert set(buckets) == {1.0, 0.067}
    assert buckets[0.067] and all(n.startswith("text_encoder.") for n in buckets[0.067])
    assert any(n.startswith("text_transformer_blocks.") for n in buckets[1.0])
    assert "text_global_proj.weight" in buckets[1.0]
    assert not any(n.startswith("text_encoder.") for n in buckets[1.0])


def test_a_frozen_encoder_block_never_reaches_the_optimizer(tmp_path):
    """Freezing happens in the constructor so that this holds: a parameter the optimizer
    never sees is also a parameter DDP never waits for a gradient on."""
    model = _real_denoiser(tmp_path, layers=4, blocks=2)
    buckets = _names_by_mult(model, param_groups(
        model, weight_decay=0.1, lr_mult_patterns={"text_encoder.": 0.067}, verbose=False))
    assert {n.split(".")[3] for n in buckets[0.067]} == {"2", "3"}
    assert not any("embeddings" in n for n in buckets[0.067])


def test_a_layer_pattern_without_its_trailing_dot_catches_layer_eleven(tmp_path):
    """Substring matching, so "layer.1" is also a prefix of "layer.11", "layer.12"... A
    layer-wise rate map written without the trailing dot silently slows four extra blocks."""
    model = _real_denoiser(tmp_path, layers=12)
    loose = _names_by_mult(model, param_groups(
        model, weight_decay=0.0,
        lr_mult_patterns={"text_encoder.encoder.layer.1": 0.5}, verbose=False))
    tight = _names_by_mult(model, param_groups(
        model, weight_decay=0.0,
        lr_mult_patterns={"text_encoder.encoder.layer.1.": 0.5}, verbose=False))

    assert any(".layer.11." in n for n in loose[0.5])
    assert all(".layer.1." in n for n in tight[0.5])
    assert not any(".layer.11." in n for n in tight[0.5])

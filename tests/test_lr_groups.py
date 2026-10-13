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

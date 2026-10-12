"""AR-clamped sampling: the autoregressive model revises the draft, discretely."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from dimol.builders import build_path
from dimol.models.denoiser import ARClampedDenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.gpt import SmilesAR, SmilesARConfig

SEQ, VOCAB, EMB, TEXT = 12, 64, 8, 32


def _diffusion():
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4, text_dim=TEXT,
        self_conditioning=False, length_conditioning=False))


def _ar():
    return SmilesAR(SmilesARConfig(
        vocab_size=VOCAB, model_dim=32, n_heads=4, n_layers=2, max_pos=SEQ + 4,
        pad_idx=0, text_dim=TEXT))


def _path():
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    return build_path(cfg, seq_len=SEQ, emb_dim=EMB, device="cpu")


def _wrapper(weight: float, from_alpha: float = 0.0):
    w = ARClampedDenoiserModel(_diffusion(), _path(), _ar(), weight=weight,
                               from_alpha=from_alpha, pad_idx=0)
    w.text = torch.randn(3, 5, TEXT)
    w.text_mask = torch.ones(3, 5, dtype=torch.bool)
    return w


def test_weight_zero_is_ordinary_clamping_free_sampling():
    torch.manual_seed(0)
    w = _wrapper(weight=0.0)
    out = w(torch.randn(3, SEQ, EMB), torch.full((3, 1, 1), 0.9))
    assert out.shape == (3, SEQ, EMB)
    assert w.seen == 0, "with weight 0 the AR model should never be consulted"


def test_the_autoregressive_model_is_frozen():
    w = _wrapper(weight=1.0)
    assert all(not p.requires_grad for p in w.ar_model.parameters())


def test_it_revises_some_positions_once_the_weight_is_real():
    torch.manual_seed(0)
    w = _wrapper(weight=50.0)
    w(torch.randn(3, SEQ, EMB), torch.full((3, 1, 1), 0.9))
    assert w.seen > 0, "the AR model should have been consulted"
    assert w.revised > 0, "a large weight should change at least one position"


def test_padding_positions_are_never_revised():
    """The length is supplied separately; the canvas tail is not the AR model's to fill."""
    torch.manual_seed(0)
    model = _diffusion()
    w = ARClampedDenoiserModel(model, _path(), _ar(), weight=1e6, from_alpha=0.0,
                               pad_idx=0)
    w.text = torch.randn(2, 5, TEXT)
    w.text_mask = torch.ones(2, 5, dtype=torch.bool)
    # Drive the draft to padding everywhere by zeroing the readout except the pad row.
    with torch.no_grad():
        model.out_proj.weight.zero_()
        model.out_proj.bias.fill_(-10.0)
        model.out_proj.bias[0] = 10.0
    w(torch.randn(2, SEQ, EMB), torch.full((2, 1, 1), 0.9))
    assert w.revised == 0, "padding positions must survive the revision untouched"


def test_guidance_stays_off_below_the_alpha_floor():
    torch.manual_seed(0)
    w = _wrapper(weight=50.0, from_alpha=0.9)
    w(torch.randn(3, SEQ, EMB), torch.full((3, 1, 1), 0.05))  # alpha ~ 0.08
    assert w.seen == 0


def test_selective_revision_touches_far_fewer_positions():
    """Speculative decoding accepts the draft unless the verifier confidently objects.

    Rewriting every disagreement edits about a fifth of the molecule per solver step --
    measured agreement between these two models is 80.8% -- and the sample collapses.
    """
    torch.manual_seed(0)
    blanket = _wrapper(weight=50.0)
    torch.manual_seed(1)
    x, t = torch.randn(3, SEQ, EMB), torch.full((3, 1, 1), 0.9)
    blanket(x, t)

    torch.manual_seed(0)
    selective = ARClampedDenoiserModel(
        _diffusion(), _path(), _ar(), weight=50.0, from_alpha=0.0, pad_idx=0,
        ar_min_prob=0.9, draft_max_prob=0.5)
    selective.text, selective.text_mask = blanket.text, blanket.text_mask
    torch.manual_seed(1)
    selective(x, t)

    assert selective.revised <= blanket.revised, (
        f"thresholds should not increase edits: {selective.revised} vs {blanket.revised}")


def test_impossible_thresholds_revise_nothing():
    torch.manual_seed(0)
    w = ARClampedDenoiserModel(_diffusion(), _path(), _ar(), weight=50.0, from_alpha=0.0,
                               pad_idx=0, ar_min_prob=1.01, draft_max_prob=1.0)
    w.text = torch.randn(3, 5, TEXT)
    w.text_mask = torch.ones(3, 5, dtype=torch.bool)
    w(torch.randn(3, SEQ, EMB), torch.full((3, 1, 1), 0.9))
    assert w.revised == 0

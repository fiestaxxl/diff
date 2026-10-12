"""Particle filtering over the reverse trajectory, with the AR model as the potential.

Three separate measurements say the verifier cannot edit a position: forcing its token
destroys the sample, its gradient moves nothing, and re-noising a disputed position gives
the same token back. What it does well is rank finished molecules. These tests pin the
properties that make ranking usable during the solve rather than only after it: groups
never mix, a resample only ever permutes rows within a caption's own group, and every
piece of per-row state travels with its particle.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from dimol.builders import build_path
from dimol.eval.sampling import SamplingParams, _ar_mean_logp, _smc_on, _smc_simulate
from dimol.models.denoiser import DenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.gpt import SmilesAR, SmilesARConfig

SEQ, VOCAB, EMB, TEXT, PAD = 12, 64, 8, 32, 0


def _path():
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    return build_path(cfg, seq_len=SEQ, emb_dim=EMB, device="cpu")


def _raw():
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=PAD, max_pos=SEQ + 4, text_dim=TEXT,
        self_conditioning=False, length_conditioning=False))


def _ar():
    return SmilesAR(SmilesARConfig(vocab_size=VOCAB, model_dim=32, n_heads=4, n_layers=2,
                                   max_pos=SEQ + 4, pad_idx=PAD, text_dim=TEXT))


class _Simulator:
    """Records the segments it is asked to integrate and adds a row-dependent drift."""

    def __init__(self):
        self.segments = []

    def simulate(self, x, ts, use_bar=False, on_step=None):
        self.segments.append((float(ts[0, 0]), float(ts[0, -1]), ts.shape[1]))
        out = x + 1.0
        return on_step(out, ts[:, -1]) if on_step is not None else out


def _setup(group=4, ncap=3, alphas=(0.5, 0.8), lam=1.0, ess=1.0):
    torch.manual_seed(0)
    raw, path = _raw(), _path()
    score_model = DenoiserModel(raw, path, regime="epsilon")
    b = ncap * group
    score_model.text = torch.randn(b, 5, TEXT)
    score_model.text_mask = torch.ones(b, 5, dtype=torch.bool)
    params = SamplingParams(num_samples=b, ar_model=_ar(), smc_group=group,
                            smc_alphas=alphas, smc_lambda=lam, smc_ess=ess)
    grid = torch.linspace(0.0, 1.0, 11)
    ts = grid.view(1, -1, 1, 1).expand(b, -1, -1, -1)
    x0 = torch.randn(b, SEQ, EMB)
    return raw, path, score_model, params, grid, ts, x0, b


def test_smc_is_off_without_a_verifier_or_checkpoints():
    assert not _smc_on(SamplingParams(num_samples=1, smc_group=4, smc_alphas=(0.5,)))
    assert not _smc_on(SamplingParams(num_samples=1, ar_model=_ar(), smc_group=1,
                                      smc_alphas=(0.5,)))
    assert not _smc_on(SamplingParams(num_samples=1, ar_model=_ar(), smc_group=4,
                                      smc_alphas=()))
    assert _smc_on(SamplingParams(num_samples=1, ar_model=_ar(), smc_group=4,
                                  smc_alphas=(0.5,)))


def test_trajectory_is_split_at_the_requested_alphas_and_covers_the_whole_grid():
    raw, path, score_model, params, grid, ts, x0, _ = _setup()
    sim = _Simulator()
    _smc_simulate(sim, x0, ts, grid, path, None, score_model, raw, raw.out_proj,
                  PAD, params, "cpu", None, True, False)
    assert len(sim.segments) == 3                       # two checkpoints, three segments
    assert sim.segments[0][0] == pytest.approx(float(grid[0]))
    assert sim.segments[-1][1] == pytest.approx(float(grid[-1]))
    for before, after in zip(sim.segments[:-1], sim.segments[1:]):
        assert before[1] == pytest.approx(after[0])     # segments join, nothing skipped
    for lo, hi, n in sim.segments:
        assert n >= 2                                   # a segment needs two endpoints


def test_a_resample_never_moves_a_particle_between_captions():
    group, ncap = 4, 3
    raw, path, score_model, params, grid, ts, x0, b = _setup(group=group, ncap=ncap)
    # Tag every row with its caption so a leak across groups is visible in the output.
    x0 = torch.arange(b, dtype=torch.float32).view(b, 1, 1).expand(b, SEQ, EMB).clone()
    sim = _Simulator()
    out = _smc_simulate(sim, x0, ts, grid, path, None, score_model, raw, raw.out_proj,
                        PAD, params, "cpu", None, True, False)
    tags = (out[:, 0, 0] - len(sim.segments)).round().long()   # undo the drift
    for cap in range(ncap):
        owned = set(range(cap * group, (cap + 1) * group))
        assert set(tags[cap * group : (cap + 1) * group].tolist()) <= owned


def test_every_piece_of_per_row_state_travels_with_its_particle():
    group, ncap = 4, 3
    raw, path, score_model, params, grid, ts, x0, b = _setup(group=group, ncap=ncap)
    score_model.length = torch.arange(b)
    score_model._x0_self = torch.arange(b, dtype=torch.float32).view(b, 1, 1) \
        .expand(b, SEQ, EMB).clone()
    mask_box = [torch.arange(b).view(b, 1, 1).expand(b, SEQ, 1) % 2 == 0]
    text_tag = score_model.text[:, 0, 0].clone()
    _smc_simulate(_Simulator(), x0, ts, grid, path, None, score_model, raw, raw.out_proj,
                  PAD, params, "cpu", mask_box, True, True)
    carry = score_model._x0_self[:, 0, 0].round().long()
    assert torch.equal(score_model.length, carry)
    assert torch.equal(mask_box[0][:, 0, 0], carry % 2 == 0)
    assert torch.equal(score_model.text[:, 0, 0], text_tag[carry])


def test_a_diverse_group_is_left_alone_when_the_effective_size_is_high():
    raw, path, score_model, params, grid, ts, x0, b = _setup(lam=0.0, ess=0.5)
    # lambda 0 makes every weight equal, so the effective sample size is the group size
    # and no particle should be replaced.
    x0 = torch.arange(b, dtype=torch.float32).view(b, 1, 1).expand(b, SEQ, EMB).clone()
    sim = _Simulator()
    out = _smc_simulate(sim, x0, ts, grid, path, None, score_model, raw, raw.out_proj,
                        PAD, params, "cpu", None, True, False)
    assert torch.equal((out[:, 0, 0] - len(sim.segments)).round().long(),
                       torch.arange(b))


def test_the_batch_has_to_divide_into_whole_groups():
    raw, path, score_model, params, grid, ts, x0, _ = _setup(group=4, ncap=3)
    with pytest.raises(ValueError, match="multiple of the particle count"):
        _smc_simulate(_Simulator(), x0[:-1], ts[:-1], grid, path, None, score_model, raw,
                      raw.out_proj, PAD, params, "cpu", None, True, False)


def test_the_score_ignores_how_much_padding_follows_the_molecule():
    """Particles differ in length by construction, so the score has to be comparable.

    The same molecule written on a longer canvas must score the same: a summed
    log-likelihood would be unchanged here too, but it would then prefer short molecules
    among the particles, which is exactly the wrong bias when length is what the particles
    disagree about. The mean is what the post-hoc selection rule uses.
    """
    torch.manual_seed(0)
    ar = _ar()
    score_model = DenoiserModel(_raw(), _path(), regime="epsilon")
    score_model.text = torch.randn(2, 5, TEXT)
    score_model.text_mask = torch.ones(2, 5, dtype=torch.bool)
    body = torch.randint(1, VOCAB, (2, 6))
    narrow = torch.cat((body, torch.full((2, 2), PAD)), dim=1)
    wide = torch.cat((body, torch.full((2, SEQ - 6), PAD)), dim=1)
    on_narrow = _ar_mean_logp(narrow, score_model, ar, PAD, True)
    on_wide = _ar_mean_logp(wide, score_model, ar, PAD, True)
    assert on_wide.shape == (2,)
    assert torch.allclose(on_narrow, on_wide, atol=1e-5)


# --- the length as a posterior rather than a decision ---------------------------------

def _pin_params(cdf=None):
    return SamplingParams(num_samples=2, length_cdf=cdf)


def test_a_hard_mask_still_overwrites_the_tail_exactly():
    """The boolean path must be untouched: it is what every existing number was measured
    with, and a weighted blend that quietly replaced it would move them all."""
    path = _path()
    state = torch.randn(2, SEQ, EMB)
    pad = torch.randn(1, 1, EMB)
    mask = torch.zeros(2, SEQ, 1, dtype=torch.bool)
    mask[:, 8:] = True
    t = torch.full((2, 1, 1), 0.5)
    torch.manual_seed(0)
    alpha, beta = path.alpha(t), path.beta(t)
    noise = torch.randn_like(state)
    out = torch.where(mask, alpha * pad + beta * noise, state)
    assert torch.equal(out[:, :8], state[:, :8])
    assert not torch.allclose(out[:, 8:], state[:, 8:])


def test_the_posterior_is_a_cumulative_distribution_over_the_canvas():
    import numpy as np

    cand = np.array([[5, 7, 9]], dtype=np.int64)
    probs = np.array([[0.5, 0.3, 0.2]], dtype=np.float32)
    positions = np.arange(12).reshape(1, 12, 1)
    below = cand.reshape(1, 1, -1) <= positions
    cdf = (below * probs.reshape(1, 1, -1)).sum(-1)
    assert cdf[0, 4] == pytest.approx(0.0)      # before any candidate ends
    assert cdf[0, 5] == pytest.approx(0.5)      # the first candidate has ended here
    assert cdf[0, 7] == pytest.approx(0.8)
    assert cdf[0, 9] == pytest.approx(1.0)      # past every candidate
    assert all(cdf[0, i] <= cdf[0, i + 1] for i in range(11))


def test_a_weighted_mask_leaves_the_certain_ends_exactly_where_a_hard_one_would():
    """Only the uncertain band may differ; a position the posterior is sure about has to
    behave exactly as the hard mask behaved, or the two are not comparable."""
    path = _path()
    state = torch.randn(2, SEQ, EMB)
    pad = torch.randn(1, 1, EMB)
    weight = torch.zeros(2, SEQ, 1)
    weight[:, 6:9] = torch.tensor([0.25, 0.6, 0.9]).view(1, 3, 1)
    weight[:, 9:] = 1.0
    t = torch.full((2, 1, 1), 0.5)
    torch.manual_seed(0)
    alpha, beta = path.alpha(t), path.beta(t)
    noise = torch.randn_like(state)
    conditional = alpha * pad + beta * noise
    soft = weight * conditional + (1.0 - weight) * state
    assert torch.allclose(soft[:, :6], state[:, :6])
    assert torch.allclose(soft[:, 9:], conditional.expand_as(state)[:, 9:])
    middle = soft[:, 6:9]
    assert not torch.allclose(middle, state[:, 6:9])
    assert not torch.allclose(middle, conditional.expand_as(state)[:, 6:9])

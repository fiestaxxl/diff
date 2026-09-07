"""alpha/beta schedules and the denoiser: boundaries, derivatives, consistency.

These tests pin the current formulas so that refactors and optimizations cannot
shift them.
"""
from __future__ import annotations

import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta, LinearAlpha, SquareRootBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath


def test_cosine_boundaries() -> None:
    alpha, beta = CosineAlpha("cpu"), CosineBeta("cpu")
    zeros, ones = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    assert torch.allclose(alpha(zeros), torch.zeros(1, 1, 1), atol=1e-6)
    assert torch.allclose(alpha(ones), torch.ones(1, 1, 1), atol=1e-6)
    assert torch.allclose(beta(zeros), torch.ones(1, 1, 1), atol=1e-6)
    assert torch.allclose(beta(ones), torch.zeros(1, 1, 1), atol=1e-6)


def test_cosine_variance_preserving() -> None:
    """alpha^2 + beta^2 = 1 for the cosine schedule."""
    alpha, beta = CosineAlpha("cpu"), CosineBeta("cpu")
    t = torch.rand(16, 1, 1)
    assert torch.allclose(alpha(t) ** 2 + beta(t) ** 2, torch.ones_like(t), atol=1e-5)


def test_analytic_dt_matches_autograd() -> None:
    """Analytic derivatives match autograd (they also feed the SDE drift)."""
    for fn in (CosineAlpha("cpu"), CosineBeta("cpu"), LinearAlpha()):
        t = torch.rand(8, 1, 1, requires_grad=True)
        y = fn(t)
        (grad,) = torch.autograd.grad(y.sum(), t)
        assert torch.allclose(fn.dt(t.detach()), grad, atol=1e-4), type(fn).__name__


def test_sqrt_beta_dt_is_regularized() -> None:
    """SquareRootBeta has +1e-4 in the dt denominator, so the gap to autograd is
    small but non-zero. This test pins the current behaviour."""
    fn = SquareRootBeta()
    t = torch.rand(8, 1, 1, requires_grad=True) * 0.8
    (grad,) = torch.autograd.grad(fn(t).sum(), t)
    assert torch.allclose(fn.dt(t.detach()), grad, atol=1e-3)


def test_conditional_path_statistics() -> None:
    """x_t = alpha*z + beta*eps: mean and std follow the schedule."""
    torch.manual_seed(0)
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[4, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    z = torch.ones(4096, 4, 8)
    t = torch.full((4096, 1, 1), 0.3)
    x = path.sample_conditional_path(z, t)
    a, b = float(path.alpha(t)[0]), float(path.beta(t)[0])
    assert abs(float(x.mean()) - a) < 0.05
    assert abs(float(x.std()) - b) < 0.05


def test_conditional_score_matches_gaussian() -> None:
    """score = (alpha*z - x) / beta^2, which is what the denoiser learns."""
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[2, 3], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    z = torch.randn(5, 2, 3)
    t = torch.rand(5, 1, 1) * 0.8 + 0.1
    x = path.sample_conditional_path(z, t)
    expected = (z * path.alpha(t) - x) / path.beta(t) ** 2
    assert torch.allclose(path.conditional_score(x, z, t), expected, atol=1e-5)


def test_p_simple_sampling_is_seeded() -> None:
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[3, 4], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    a = path.p_simple.sample(2, seed=1)
    b = path.p_simple.sample(2, seed=1)
    c = path.p_simple.sample(2, seed=2)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_time_grids_keep_the_endpoints_and_the_step_count():
    from dimol.eval.sampling import SamplingParams, _time_grid

    for grid in ("uniform", "data_dense", "noise_dense", "mid_dense", "ends_dense"):
        ts = _time_grid(SamplingParams(num_timesteps=50, t_start=1e-3, t_end=0.999,
                                       time_grid=grid))
        assert ts.numel() == 50, grid
        assert abs(float(ts[0]) - 1e-3) < 1e-6, grid
        assert abs(float(ts[-1]) - 0.999) < 1e-6, grid
        assert bool((ts[1:] - ts[:-1] > 0).all()), grid


def test_dense_grids_lean_the_way_they_say():
    import torch

    from dimol.eval.sampling import SamplingParams, _time_grid

    def spacing(grid):
        ts = _time_grid(SamplingParams(num_timesteps=101, time_grid=grid))
        d = ts[1:] - ts[:-1]
        return float(d[:50].mean()), float(d[50:].mean())

    near_noise, near_data = spacing("data_dense")
    assert near_noise > near_data  # small steps at the data end
    near_noise, near_data = spacing("noise_dense")
    assert near_noise < near_data
    mid = _time_grid(SamplingParams(num_timesteps=101, time_grid="mid_dense"))
    d = mid[1:] - mid[:-1]
    assert float(d[45:55].mean()) < float(d[:10].mean())  # dense in the middle
    ends = _time_grid(SamplingParams(num_timesteps=101, time_grid="ends_dense"))
    d = ends[1:] - ends[:-1]
    assert float(d[45:55].mean()) > float(d[:10].mean())  # dense at both ends


def test_diversity_survives_a_smiles_that_will_not_reparse():
    """rdkit can write a canonical string it cannot read back; that must not crash."""
    from dimol.eval.report import evaluate_smiles

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "", "C1CC1", "n1cccc1", "CCN(CC)CC"]
    m = evaluate_smiles(smiles)
    assert 0.0 <= m["validity"] <= 1.0
    assert m["diversity"] == m["diversity"] or m["n_valid"] < 2  # not NaN when it can be

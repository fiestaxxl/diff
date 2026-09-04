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

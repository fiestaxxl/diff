"""Length conditioning: the drawn length is respected and the prior is followed."""
from __future__ import annotations

import numpy as np
import torch

from dimol.eval.sampling import SamplingParams, _draw_lengths


def test_lengths_come_from_the_prior_and_stay_on_the_canvas():
    prior = np.array([5, 5, 5, 40, 40, 99])
    generator = torch.Generator().manual_seed(0)
    drawn = _draw_lengths(prior, 500, canvas=64, generator=generator)
    assert drawn.shape == (500,)
    assert set(drawn.tolist()) <= {5, 40, 64}  # 99 is clipped to the canvas
    assert drawn.min() >= 1


def test_the_drawn_distribution_matches_the_prior():
    prior = np.array([10] * 90 + [30] * 10)
    generator = torch.Generator().manual_seed(0)
    drawn = _draw_lengths(prior, 4000, canvas=64, generator=generator)
    share_short = float((drawn == 10).float().mean())
    assert 0.85 < share_short < 0.95


def test_drawing_is_reproducible_for_a_seed():
    prior = np.arange(1, 50)
    a = _draw_lengths(prior, 32, 64, torch.Generator().manual_seed(7))
    b = _draw_lengths(prior, 32, 64, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_an_empty_prior_is_reported():
    try:
        _draw_lengths(np.array([]), 4, 64, torch.Generator().manual_seed(0))
    except ValueError as err:
        assert "empty" in str(err)
    else:
        raise AssertionError("an empty length prior must be rejected")


def test_the_prior_is_off_by_default():
    assert SamplingParams().length_prior is None

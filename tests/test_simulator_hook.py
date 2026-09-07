"""The per-step hook: conditional sampling needs to overwrite part of the state."""
from __future__ import annotations

import torch

from dimol.diffusion.simulators import EulerMaruyamaSimulator


class ConstantDriftSDE:
    """dx = 1 dt, no noise: the trajectory is exactly predictable."""

    def drift_coefficient(self, xt, t, **kwargs):
        return torch.ones_like(xt)

    def diffusion_coefficient(self, xt, t, **kwargs):
        return torch.zeros_like(xt)


def _ts(steps: int, batch: int = 2):
    grid = torch.linspace(0.0, 1.0, steps)
    return grid.view(1, steps, 1, 1).expand(batch, -1, -1, -1)


def test_without_a_hook_the_trajectory_is_unchanged():
    sim = EulerMaruyamaSimulator(ConstantDriftSDE())
    x = torch.zeros(2, 4, 3)
    out = sim.simulate(x, _ts(11), use_bar=False)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_the_hook_can_pin_part_of_the_state():
    sim = EulerMaruyamaSimulator(ConstantDriftSDE())
    x = torch.zeros(2, 4, 3)

    def pin(state, t_next):
        state = state.clone()
        state[:, 2:] = -7.0  # the caller knows these positions
        return state

    out = sim.simulate(x, _ts(11), use_bar=False, on_step=pin)
    assert torch.allclose(out[:, :2], torch.ones_like(out[:, :2]), atol=1e-5)
    assert torch.allclose(out[:, 2:], torch.full_like(out[:, 2:], -7.0))


def test_the_hook_sees_the_time_it_is_stepping_to():
    sim = EulerMaruyamaSimulator(ConstantDriftSDE())
    seen = []

    def record(state, t_next):
        seen.append(float(t_next.flatten()[0]))
        return state

    sim.simulate(torch.zeros(2, 4, 3), _ts(5), use_bar=False, on_step=record)
    assert len(seen) == 4
    assert abs(seen[0] - 0.25) < 1e-6 and abs(seen[-1] - 1.0) < 1e-6


def test_the_hook_is_not_passed_on_to_the_model():
    """on_step must be consumed by simulate, not forwarded into the SDE."""

    class StrictSDE(ConstantDriftSDE):
        def drift_coefficient(self, xt, t, **kwargs):
            assert not kwargs, f"unexpected kwargs reached the SDE: {sorted(kwargs)}"
            return torch.ones_like(xt)

    sim = EulerMaruyamaSimulator(StrictSDE())
    sim.simulate(torch.zeros(1, 2, 2), _ts(3, batch=1), use_bar=False,
                 on_step=lambda x, t: x)

"""Score wrapper: eps/x prediction -> score, for use inside the SDE.

The denoiser and score formulas are carried over VERBATIM from
dimol/models/models.py.
"""
import torch
import torch.nn as nn


class DenoiserModel(nn.Module):
    """eps/x prediction turned into a score, plus the self-conditioning carry.

    A self-conditioned model wants its own previous estimate of x0 as a second input. The
    solver calls this wrapper once per step in order, so the estimate is simply kept here
    between calls, starting from zeros. ``reset()`` clears it, which matters when the same
    wrapper is reused for another batch.
    """

    def __init__(self, eps_model, path, regime='epsilon'):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime
        self._x0_self = None

    def reset(self):
        self._x0_self = None

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t  = torch.clamp(self.path.beta(t),  min=1e-3)
        t_in    = t.squeeze(-1)
        if getattr(unwrap_model(self.eps_model), "self_conditioning", False):
            kwargs["x0_self"] = self._x0_self
        pred    = self.eps_model(x, t_in, **kwargs)

        if self.regime == 'epsilon':
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == 'x':
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        self._x0_self = x0_pred.detach()
        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score


class ClampedDenoiserModel(nn.Module):
    """Denoiser whose x0 estimate is snapped towards the nearest token embedding.

    The clamping trick from Diffusion-LM: the reverse process drifts through a
    continuous space, but only the points that are token embeddings decode to anything,
    so pulling the x0 estimate onto that set at every step keeps the trajectory on the
    data manifold. The score formula is unchanged, only the x0 estimate it is built from.

    ``strength`` is how far to pull (1.0 = fully onto the embedding), and clamping is
    applied only once alpha exceeds ``from_alpha``, because early in the reverse process
    the estimate carries no information and snapping it would inject noise.
    """

    def __init__(self, eps_model, path, regime="epsilon", strength=1.0, from_alpha=0.5):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime
        self.strength = float(strength)
        self.from_alpha = float(from_alpha)

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        raw = unwrap_model(self.eps_model)
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t = torch.clamp(self.path.beta(t), min=1e-3)
        t_in = t.squeeze(-1)
        pred = self.eps_model(x, t_in, **kwargs)

        if self.regime == "epsilon":
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == "x":
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        if self.strength > 0:
            ids = raw.out_proj(x0_pred).argmax(-1)
            snapped = raw.token_embedding(ids)
            gate = (alpha_t > self.from_alpha).to(x0_pred.dtype)
            x0_pred = x0_pred + gate * self.strength * (snapped - x0_pred)

        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score

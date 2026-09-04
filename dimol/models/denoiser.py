"""Score wrapper: eps/x prediction -> score, for use inside the SDE.

The denoiser and score formulas are carried over VERBATIM from
dimol/models/models.py.
"""
import torch
import torch.nn as nn


class DenoiserModel(nn.Module):
    def __init__(self, eps_model, path, regime='epsilon'):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime

    def forward(self, x, t, **kwargs):
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t  = torch.clamp(self.path.beta(t),  min=1e-3)
        t_in    = t.squeeze(-1)
        pred    = self.eps_model(x, t_in, **kwargs)

        if self.regime == 'epsilon':
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == 'x':
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score

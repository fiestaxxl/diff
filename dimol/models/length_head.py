"""Predicting a molecule's length from its caption.

The diffusion model takes a length as an input, and measurement says that input is worth
0.11 MACCS on top of a correct caption while carrying almost no structure of its own: with
a wrong caption and the right length the samples sit at 0.296 against a 0.270 floor. So it
is a genuine constraint rather than a shortcut, and the only problem is where the length
comes from at generation time. Drawing it from the corpus wastes it; the caption predicts
it, since captions routinely state the size ("N-nonacosanoyl" is twenty-nine carbons).

A ridge regression on mean-pooled frozen caption states already halves the error, from
18.9 tokens to 9.0 with r = 0.855. This head sees the token-level states instead of a mean
pool and predicts a distribution rather than a point, so generation can sample a length
the way it samples everything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthHead(nn.Module):
    """Attention-pool the caption, then classify the length over the canvas.

    Classification rather than regression: the answer is a small integer, many captions
    are genuinely ambiguous about size, and a distribution lets the sampler draw instead
    of always taking the mode. Predicting a point would quietly collapse that ambiguity
    onto the mean, which is the one length that is never right.
    """

    def __init__(self, text_dim: int = 768, hidden: int = 256, canvas: int = 128):
        super().__init__()
        self.canvas = int(canvas)
        self.query = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
        self.attend = nn.Linear(text_dim, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.canvas + 1),
        )

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, D) and (B, S) -> logits over lengths 0..canvas."""
        scores = self.attend(text + self.query).squeeze(-1)  # (B, S)
        scores = scores.masked_fill(~text_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (text * weights).sum(dim=1)
        return self.net(pooled)

    @torch.no_grad()
    def predict(self, text: torch.Tensor, text_mask: torch.Tensor,
                temperature: float = 0.0) -> torch.Tensor:
        """Lengths for a batch of captions: the mode, or a draw at temperature > 0."""
        logits = self(text, text_mask)
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "config": {"text_dim": self.query.shape[-1],
                               "hidden": self.net[1].out_features,
                               "canvas": self.canvas}}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "LengthHead":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(**payload["config"])
        head.load_state_dict(payload["state_dict"])
        head.eval()
        return head


def length_loss(logits: torch.Tensor, target: torch.Tensor,
                smoothing: float = 0.1) -> torch.Tensor:
    """Cross-entropy with neighbouring lengths given partial credit.

    Being one token out is nearly right and should not be punished like being thirty out,
    so the target is smoothed onto its immediate neighbours rather than being a spike.
    """
    canvas = logits.shape[-1]
    soft = torch.zeros_like(logits)
    soft.scatter_(1, target[:, None], 1.0 - smoothing)
    for offset in (-1, 1):
        shifted = (target + offset).clamp(0, canvas - 1)
        soft.scatter_add_(1, shifted[:, None],
                          torch.full_like(shifted[:, None], smoothing / 2, dtype=soft.dtype))
    return -(soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()

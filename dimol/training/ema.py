"""Exponential moving average of the weights.

Diffusion models are usually sampled from an averaged copy of the weights rather than
from the last step: the average is a cheap variance reduction over the optimization
noise, and in image diffusion it is worth several points of sample quality. The average
is kept on the training device and written next to the checkpoint, so generation can
point at it without any change to the loading path.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from dimol.training.distributed import unwrap_model


class EmaWeights:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_step: int = 0,
        update_every: int = 1,
    ):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.update_every = max(1, int(update_every))
        raw = unwrap_model(model)
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone().float() for name, param in raw.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        if step < self.start_step or step % self.update_every:
            return
        raw = unwrap_model(model)
        # warm up the average so early steps are not dominated by the initialization
        decay = min(self.decay, (1 + step) / (10 + step))
        for name, value in raw.state_dict().items():
            shadow = self.shadow.get(name)
            if shadow is None or not value.is_floating_point():
                self.shadow[name] = value.detach().clone().float()
                continue
            shadow.mul_(decay).add_(value.detach().float(), alpha=1 - decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.detach().clone().float() for name, value in state.items()}

    @torch.no_grad()
    def swap_into(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """Put the average into the model, returning the weights it replaced."""
        raw = unwrap_model(model)
        current = {name: value.detach().clone() for name, value in raw.state_dict().items()}
        raw.load_state_dict(
            {name: value.to(dtype=current[name].dtype) for name, value in self.shadow.items()
             if name in current},
            strict=False,
        )
        return current

    @torch.no_grad()
    def restore(self, model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
        unwrap_model(model).load_state_dict(weights, strict=False)

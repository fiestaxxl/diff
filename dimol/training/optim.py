"""Optimizers and learning rate schedules.

Both the parameter grouping and the warmup+cosine formula are carried over from
the old code:
* the groups come from ``DiffusionTransformer.configure_optimizers``
  (token_embedding without weight decay, tensors with dim >= 2 with decay,
  everything else without);
* the schedule comes from ``ConditionalGaussianDenoiserTrainerLite.get_lr``.
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from dimol import registry
from dimol.config import Duration


# ----------------------------------------------------------------------
# Parameter groups
# ----------------------------------------------------------------------
def param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_patterns: Iterable[str] = (),
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    no_decay_patterns = tuple(no_decay_patterns)
    decay_params: List[nn.Parameter] = []
    nodecay_params: List[nn.Parameter] = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if any(pat in name for pat in no_decay_patterns):
            nodecay_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    if verbose:
        nd = sum(p.numel() for p in decay_params)
        nn_ = sum(p.numel() for p in nodecay_params)
        print(f"decayed: {len(decay_params)} tensors, {nd:,} params")
        print(f"non-decayed: {len(nodecay_params)} tensors, {nn_:,} params")

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


@registry.optimizers.register("adamw")
def build_adamw(
    params: List[Dict[str, Any]],
    lr: float,
    betas: Iterable[float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
    device_type: str = "cpu",
    verbose: bool = True,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    if fused is None:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused = bool(fused_available and device_type == "cuda")
    if verbose:
        print(f"using fused AdamW: {fused}")
    return torch.optim.AdamW(
        params, lr=lr, betas=tuple(betas), eps=eps, fused=fused, **kwargs
    )


# ----------------------------------------------------------------------
# LR schedules
# ----------------------------------------------------------------------
class LRScheduler:
    """Returns the lr for a given optimizer step index."""

    def __call__(self, step: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@registry.schedulers.register("warmup_cosine")
class WarmupCosine(LRScheduler):
    """Linear warmup then cosine decay. The formula is the old get_lr, unchanged."""

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        t_warmup: Any = "500ba",
        t_max: Any = "2500ba",
        steps_per_epoch: Optional[int] = None,
    ):
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = Duration.parse(t_warmup).in_batches(steps_per_epoch)
        self.max_steps = Duration.parse(t_max).in_batches(steps_per_epoch)
        if self.max_steps <= self.warmup_steps:
            raise ValueError(
                f"scheduler.t_max ({self.max_steps}ba) must be greater than t_warmup "
                f"({self.warmup_steps}ba)"
            )

    def __call__(self, it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


@registry.schedulers.register("constant")
class ConstantLR(LRScheduler):
    def __init__(self, max_lr: float, **kwargs: Any):
        self.max_lr = float(max_lr)

    def __call__(self, it: int) -> float:
        return self.max_lr


@registry.schedulers.register("warmup_constant")
class WarmupConstant(LRScheduler):
    def __init__(
        self, max_lr: float, t_warmup: Any = "500ba", steps_per_epoch: Optional[int] = None, **kwargs: Any
    ):
        self.max_lr = float(max_lr)
        self.warmup_steps = Duration.parse(t_warmup).in_batches(steps_per_epoch)

    def __call__(self, it: int) -> float:
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        return self.max_lr

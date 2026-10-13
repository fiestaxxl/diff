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
    lr_mult_patterns: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Split the parameters by weight decay, and optionally by learning rate.

    ``lr_mult_patterns`` maps a substring of a parameter name to a multiplier on the
    scheduled learning rate. It exists for fine-tuning a pretrained submodule beside
    freshly initialised ones: a text encoder that has already converged needs a far
    smaller step than a cross-attention that starts at zero, and one rate cannot serve
    both. The multiplier rides on the group as ``lr_mult`` and the trainer applies it at
    every optimizer step, because the scheduler produces one number for the whole run.

    The first pattern a name matches wins, so order the dictionary from specific to
    general. Parameters matching nothing get a multiplier of 1.0.
    """
    no_decay_patterns = tuple(no_decay_patterns)
    lr_mult_patterns = dict(lr_mult_patterns or {})
    # 1.0 last: it is the fallback bucket, and insertion order decides matching.
    buckets: Dict[float, Dict[str, List[nn.Parameter]]] = {}
    seen = set()

    def bucket_for(name: str) -> float:
        for pattern, mult in lr_mult_patterns.items():
            if pattern in name:
                return float(mult)
        return 1.0

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        mult = bucket_for(name)
        side = buckets.setdefault(mult, {"decay": [], "nodecay": []})
        if any(pat in name for pat in no_decay_patterns):
            side["nodecay"].append(p)
        elif p.dim() >= 2:
            side["decay"].append(p)
        else:
            side["nodecay"].append(p)

    groups: List[Dict[str, Any]] = []
    for mult in sorted(buckets, key=lambda m: (m != 1.0, m)):
        side = buckets[mult]
        if side["decay"]:
            groups.append({"params": side["decay"], "weight_decay": weight_decay,
                           "lr_mult": mult})
        if side["nodecay"]:
            groups.append({"params": side["nodecay"], "weight_decay": 0.0,
                           "lr_mult": mult})

    if verbose:
        for g in groups:
            kind = "decayed" if g["weight_decay"] else "non-decayed"
            total = sum(p.numel() for p in g["params"])
            print(f"{kind} (lr x{g['lr_mult']:g}): {len(g['params'])} tensors, "
                  f"{total:,} params")

    return groups


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

"""DDP plumbing: initialization, ranks, seeding, metric reduction."""
from __future__ import annotations

import datetime
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    ddp: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    device_type: str = "cpu"

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def is_ddp_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def init_distributed(
    backend: str = "nccl",
    timeout_min: int = 30,
    device: Optional[str] = None,
) -> DistEnv:
    """Start the process group when launched through torchrun."""
    if not is_ddp_run():
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return DistEnv(
            ddp=False,
            device=device,
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
        )

    if not torch.cuda.is_available() and backend == "nccl":
        raise RuntimeError(
            "backend=nccl requires CUDA; for a CPU run set ddp_config.backend=gloo"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # set the device BEFORE init_process_group: NCCL requires it
    if torch.cuda.is_available():
        dev = f"cuda:{local_rank}"
        torch.cuda.set_device(dev)
        device_type = "cuda"
    else:
        dev, device_type = "cpu", "cpu"

    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_min))

    return DistEnv(
        ddp=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=dev,
        device_type=device_type,
    )


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_all(seed: int, rank: int = 0, deterministic: bool = False, tf32: bool = False) -> None:
    """Per-process seeding (offset by rank so noise/dropout differ across replicas)."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if tf32:
        torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip the DDP / torch.compile wrappers."""
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def reduce_metrics(metrics: Dict[str, torch.Tensor], ddp: bool) -> Dict[str, torch.Tensor]:
    """
    Average a dict of scalar tensors across all DDP ranks.
    Stacks values into a single tensor for one all-reduce instead of N.

    Carried over from dimol/diffusion/trainers.py::reduce_loss_dict with two fixes:
    keys are sorted (same order on every rank) and SUM followed by a division by
    world_size replaces ReduceOp.AVG, which the gloo backend does not support.
    """
    if not ddp or not metrics:
        return {k: v.detach().float().clone() for k, v in metrics.items()}

    keys = sorted(metrics.keys())
    stacked = torch.stack([metrics[k].detach().float() for k in keys])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    stacked = stacked / dist.get_world_size()
    return {k: stacked[i] for i, k in enumerate(keys)}


def all_reduce_min(value: int, ddp: bool, device: str | torch.device = "cpu") -> int:
    """Minimum of an integer across ranks (used to stay in sync on an empty eval)."""
    if not ddp:
        return value
    t = torch.tensor([int(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def model_size_b(model: torch.nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

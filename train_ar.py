"""Train the AR grammar teacher.

Single-GPU:
    export CUDA_VISIBLE_DEVICES=GPU-xxxx
    python3 train_ar.py

DDP:
    export CUDA_VISIBLE_DEVICES=GPU-32b1f097-451b-c231-70bf-c360468975c6,GPU-dc48e1f4-b5a9-76e3-488f-64573ffede9b,GPU-aeeae707-d2ff-94ba-f8b2-a9692669101c,GPU-c624b4e0-f383-df3a-3cc1-4c3b797558e5,GPU-fee0dc35-8889-3f90-cd80-238315957336,GPU-e55f6ac3-3f61-5ac3-f78b-7f11681f037a
    torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 train_ar.py

Env overrides (mirrors your sweep convention):
    RUN_NAME, NUM_EPOCHS
"""
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dimol.datasets.data import SmilesDataset
from dimol.models.gpt import SmilesAR, SmilesARConfig
from dimol.diffusion.trainers import ARTrainer

import comet_ml

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ARTrainingConfig:
    # LR schedule
    max_lr: float = 3e-3
    min_lr: float = 3e-5
    warmup_steps: int = 500
    max_steps: int = 2500          # cosine decay endpoint
    num_epochs: int = 3000

    # data / batch
    batch_size: int = 1024
    seq_len: int = 208
    grad_accum_steps: int = 1
    pad_idx: int = 0

    # optimisation
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # hardware / DDP
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    device_type: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    ddp: bool = False
    master_process: bool = True
    ddp_world_size: int = 1
    ddp_rank: int = 0
    sampler: Any = None
    val_sampler: Any = None

    use_mp: bool = False
    mixed_dtype: torch.dtype = torch.float16

    # logging / checkpointing
    validation_step: int = 100
    epoch_save_checkpoint: int = 250

    # bookkeeping
    run_name: str = "ar_v2"
    path_to_tokenizer: Path = Path("data/smiles_bpe.json")
    exp: Any = None

    def to_dict(self): 
        return asdict(self)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def is_ddp_run() -> bool: 
    return int(os.environ.get("WORLD_SIZE", 1)) > 1

def set_env(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def main():
    set_env(42)
    ddp = is_ddp_run()

    if ddp:
        init_process_group(backend="nccl")
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = (rank == 0)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master = True

    cfg = ARTrainingConfig(
        device=torch.device(device),
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        ddp=ddp, ddp_world_size=world_size, ddp_rank=rank, master_process=master,
        use_mp = True
    )

    # env overrides
    cfg.run_name   = os.environ.get("RUN_NAME", cfg.run_name)
    cfg.num_epochs = int(os.environ.get("NUM_EPOCHS", cfg.num_epochs))

    model_cfg = SmilesARConfig(
        vocab_size=512,
        model_dim=256,
        n_heads=8,
        n_layers=6,
        max_pos=cfg.seq_len + 4,   # safety pad
        pad_idx=cfg.pad_idx,
        tie_lm_head=True,
    )
    if master:
        print(f"[AR] model config: {model_cfg.to_dict()}")
        print(f"[AR] training config: run_name={cfg.run_name} epochs={cfg.num_epochs} ddp={ddp} world={world_size}")

    model = SmilesAR(model_cfg).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    train_ds = SmilesDataset("/projects/BIAM_Chem/data/tokenized", "train")
    val_ds   = SmilesDataset("/projects/BIAM_Chem/data/tokenized", "val")

    train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=True) if ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=True) if ddp else None
    cfg.sampler, cfg.val_sampler = train_sampler, val_sampler

    train_loader = DataLoader(
        train_ds, sampler=train_sampler, batch_size=cfg.batch_size,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        shuffle=(train_sampler is None),
    )
    val_loader = DataLoader(
        val_ds, sampler=val_sampler, batch_size=cfg.batch_size,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    if master:
        exp = comet_ml.start(project_name="diffusion_ar_model", mode="create")
        exp.log_parameters(cfg.to_dict())
        cfg.exp = exp

    trainer = ARTrainer(model)
    trainer.train(train_loader, cfg, val_loader=val_loader)

    if ddp: 
        destroy_process_group()


if __name__ == "__main__":
    main()
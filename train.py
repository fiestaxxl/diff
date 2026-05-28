#export CUDA_VISIBLE_DEVICES=GPU-06442a8b-7f6a-4196-aaa4-7d1bd86520e3,GPU-aeeae707-d2ff-94ba-f8b2-a9692669101c && python3 train.py
#export CUDA_VISIBLE_DEVICES=GPU-06442a8b-7f6a-4196-aaa4-7d1bd86520e3,GPU-aeeae707-d2ff-94ba-f8b2-a9692669101c && torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py
import torch
from torch.utils.data import DataLoader, DistributedSampler

from dimol.diffusion.trainers import ConditionalGaussianDenoiserTrainerLite
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.diffusion.conditionals import LinearAlpha, SquareRootBeta, CosineAlpha, CosineBeta
from dimol.diffusion.distributions import GaussianMixture
from dimol.models.models import TransformerConfig, DiffusionTransformer
from dimol.datasets.data import SimpleDataset, SmilesDataset
from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer

from dataclasses import dataclass, asdict, field

from pathlib import Path

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import comet_ml

from typing import Any



@dataclass
class LRConfig:
    learning_rate: float = 3e-4
    max_lr: float = 3e-3
    min_lr: float = 3e-3 * 0.01       # max_lr * 0.01
    warmup_steps: int = 500


@dataclass
class ScheduleConfig:
    max_steps: int = 2500              # 19,073 steps ≈ 1 epoch @ 10B tokens / 0.5M batch
    decoder_pretrain_steps: int = 0
    num_epochs: int = 500
    validation_step: int = 25
    epoch_save_checkpoint: int = 100


@dataclass
class DataConfig:
    batch_size: int = 512 #128
    seq_len: int = 208
    grad_accum_steps: int = 1 #4
    pad_idx: int = 0 #0
    sampler: DistributedSampler = None
    val_sampler: DistributedSampler = None
    vocab: dict = None


@dataclass
class OptimiserConfig:
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    lambda_ce: float = 1.0
    lambda_mse: float = 1.0
    lambda_grammar: float = 1e-3
    eps: float = 1e-3
    alpha_threshold: float = 0.80


@dataclass
class HardwareConfig:
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    device_type: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    use_mp: bool = False
    mixed_dtype: torch.dtype = torch.float16
    compile_model: bool = False


@dataclass
class DDPConfig:
    ddp: bool = False
    master_process: bool = True
    ddp_world_size: int = 1
    ddp_rank: int = 0


@dataclass
class SamplingConfig:
    sampling_noise_std: float = 0.25
    sampling_variance: float = 1.0
    num_sampling_timesteps: int = 300  # fixed typo: num_samling_timesteps
    sample_examples: bool = True
    sample_step: int = 500
    num_samples: int = 64


@dataclass
class ClassWeightConfig:
    use_class_weights: bool = False
    path_to_weights: Path = Path("data/class_weights.pt")
    class_weights: torch.Tensor = None  # fixed typo: class_weigths


# ── Master config ──────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # ── Sub-configs ───────────────────────────────────────────────────────────
    lr:           LRConfig          = field(default_factory=LRConfig)
    schedule:     ScheduleConfig    = field(default_factory=ScheduleConfig)
    data:         DataConfig        = field(default_factory=DataConfig)
    optimiser:    OptimiserConfig   = field(default_factory=OptimiserConfig)
    hardware:     HardwareConfig    = field(default_factory=HardwareConfig)
    ddp:          DDPConfig         = field(default_factory=DDPConfig)
    sampling:     SamplingConfig    = field(default_factory=SamplingConfig)
    class_weight: ClassWeightConfig = field(default_factory=ClassWeightConfig)

    # ── Misc ──────────────────────────────────────────────────────────────────
    run_name:            str  = 'v11'
    path_to_tokenizer:   Path = Path("data/smiles_bpe.json")
    exp:                 Any  = None
    regime:              str = 'epsilon'

    def to_dict(self) -> dict:
        return asdict(self)

def is_ddp_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", 1)) > 1

def set_env(seed: int, use_tf32: bool = False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if use_tf32:
        torch.set_float32_matmul_precision('high')

def main():
    set_env(seed=42, use_tf32=False)

    training_config = TrainingConfig(
        hardware=HardwareConfig(use_mp=True, mixed_dtype=torch.float16, compile_model=False)
    )
    model_config = TransformerConfig(vocab_size=512, pad_idx=training_config.data.pad_idx, num_text_blocks=4, max_pos=training_config.data.seq_len+10)

    tokenizer = SmilesTokenizer.load(training_config.path_to_tokenizer)
    training_config.data.vocab = tokenizer.get_vocab()

    ddp = is_ddp_run()

    if ddp:
        training_config.ddp.ddp = True
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank       = int(os.environ["RANK"])           # global rank
        ddp_local_rank = int(os.environ["LOCAL_RANK"])     # rank on this node
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        training_config.hardware.device = device
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0                      # only rank 0 prints/saves
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = training_config.hardware.device

    training_config.ddp.ddp_world_size = ddp_world_size
    training_config.ddp.ddp_rank = ddp_rank
    model = DiffusionTransformer(model_config)
    model.to(device)


    if training_config.hardware.compile_model:
        model = torch.compile(model, backend="aot_eager")

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # dataset = SimpleDataset(max_len=training_config.seq_len, num_samples=20000)
    train_dataset = SmilesDataset('/projects/BIAM_Chem/data/tokenized', 'train')
    val_dataset = SmilesDataset('/projects/BIAM_Chem/data/tokenized', 'val')
    

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if ddp else None

    training_config.data.sampler = train_sampler
    training_config.data.val_sampler = val_sampler
    if not master_process:
        training_config.ddp.master_process = False

    dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=training_config.data.batch_size,            # ↓ from 256
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,   # ⭐ IMPORTANT
            prefetch_factor=2
        )

    val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=training_config.data.batch_size,            # ↓ from 256
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,   # ⭐ IMPORTANT
            prefetch_factor=2
        )

    p_data = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0) 
    path = GaussianConditionalProbabilityPath(
        p_data = p_data.to(training_config.hardware.device), 
        p_simple_shape = [training_config.data.seq_len, model_config.emb_dim],
        alpha = CosineAlpha(device),
        beta = CosineBeta(device)).to(device)

    if master_process:
        exp = comet_ml.start(project_name="diffusion_smiles_model", mode="create")
        exp.log_parameters(training_config.to_dict())
        training_config.exp = exp


    if training_config.class_weight.use_class_weights and training_config.path_to_weights is not None:
        if master_process:
            print(f'Using class weigths: {training_config.path_to_weights}')
        class_weigths = torch.load(training_config.path_to_weights, weights_only=True).to(device)
        training_config.class_weight.class_weigths = class_weigths

    trainer = ConditionalGaussianDenoiserTrainerLite(path=path, 
                                                    model=model)
    trainer.train(dataloader, training_config, val_dataloader=val_dataloader)

    if ddp:
        destroy_process_group()

if __name__=='__main__':
    main()
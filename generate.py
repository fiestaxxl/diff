from dimol.diffusion.diff_eqs import LearnedScoreSDE
from dimol.models.models import  DenoiserModel
from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer
from dimol.diffusion.simulators import EulerMaruyamaSimulator

import torch

from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.diffusion.conditionals import LinearAlpha, SquareRootBeta, CosineAlpha, CosineBeta
from dimol.diffusion.distributions import GaussianMixture
from dimol.models.models import TransformerConfig, DiffusionTransformer

from dataclasses import dataclass, asdict, field

from pathlib import Path
import os
import comet_ml

from typing import Any
from torch.utils.data import DataLoader, DistributedSampler



@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4   
    max_lr: float = 3e-4
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 715
    max_steps: int = 1875 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens 


    num_epochs: int = 500           
    sampler: DistributedSampler = None
    val_sampler: DistributedSampler = None
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_grad_norm: float = 1.0

    batch_size: int = 128
    seq_len: int = 208
    grad_accum_steps: int = 4
    weight_decay: float = 0.01

    sampling_noise_std: float = 0.25

    use_mp: bool = False
    mixed_dtype: torch.dtype = torch.float16

    pad_idx: int = 0
    label_smoothing: float = 0.0
    lambda_ce: int = 0.1

    compile_model: bool = False
    ddp: bool = False
    master_process: bool = True
    ddp_world_size: int = 1
    ddp_rank: int = 0

    validation_step: int = 25

    exp: Any = None

    epoch_save_checkpoint: int = 50

    run_name: str = 'v1'
    sample_examples: bool = True
    sample_step: int = 500
    num_samples: int = 12
    sampling_variance: float = 1.0
    num_samling_timesteps: int = 300
    path_to_tokenizer: str = Path("data/smiles_bpe.json")

    alpha_threshold: float = 0.01
    use_class_weights: bool = False
    path_to_weights: str = Path("data/class_weights.pt")
    class_weigths: torch.tensor = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

training_config = TrainingConfig(use_mp=True, mixed_dtype=torch.float16, compile_model=False)

device = training_config.device
model = DiffusionTransformer.from_pretrained(load_dir='checkpoints/v7/499', map_location=device)

print('loaded model')

p_data = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0) 
path = GaussianConditionalProbabilityPath(
    p_data = p_data.to(training_config.device), 
    p_simple_shape = [training_config.seq_len, 32],
    alpha = CosineAlpha(device),
    beta = CosineBeta(device)).to(device)

tokenizer = SmilesTokenizer.load(training_config.path_to_tokenizer)
score_model = DenoiserModel(model, path)
sde = LearnedScoreSDE(path, score_model, training_config.sampling_variance)
simulator = EulerMaruyamaSimulator(sde)

x0 = path.p_simple.sample(training_config.num_samples, seed=training_config.ddp_rank)

print('generating')
eps = 1e-3
ts = torch.linspace(0.001, 0.900, training_config.num_samling_timesteps).view(1, training_config.num_samling_timesteps, 1, 1).expand(training_config.num_samples, -1, -1, -1).to(training_config.device) # (num_samples, nts, 1)
xts = simulator.simulate(x0, ts, use_bar=True) 

get_logits = model.module.out_proj if hasattr(model, "module") else model.out_proj
probs = get_logits(xts).softmax(-1).detach().cpu()
ids = probs.argmax(-1).tolist()

smiles_list = tokenizer.decode_batch(ids)
smiles_list_raw = tokenizer.decode_batch(ids, skip_special_tokens=False)

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

n_decoded_valid = 0
for s in smiles_list:
    if Chem.MolFromSmiles(s) is not None:
        n_decoded_valid += 1

validity = torch.tensor(n_decoded_valid/len(smiles_list))
print('\n'.join(smiles_list_raw))
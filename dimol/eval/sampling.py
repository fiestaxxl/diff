"""Generate SMILES from noise: SDE integration plus logit decoding.

The sampling scheme is carried over unchanged from the old code
(``generate.py::generate`` and the sampling block in
``ConditionalGaussianDenoiserTrainerLite.evaluate``): p_simple -> Euler-Maruyama
over the ts grid -> out_proj -> argmax -> decode_batch(special_decode=True).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from dimol.diffusion.diff_eqs import LearnedScoreSDE
from dimol.diffusion.simulators import EulerMaruyamaSimulator
from dimol.models.denoiser import DenoiserModel
from dimol.training.distributed import unwrap_model


@dataclass
class SamplingParams:
    num_samples: int = 64
    num_timesteps: int = 300
    variance: float = 1.0
    t_start: float = 1e-4
    t_end: float = 0.999
    seed: int = 0
    batch_size: Optional[int] = None
    regime: str = "epsilon"
    progress: bool = False


@torch.no_grad()
def sample_smiles(
    model: torch.nn.Module,
    path: Any,
    tokenizer: Any,
    params: SamplingParams,
    device: str | torch.device,
) -> List[str]:
    raw = unwrap_model(model)
    get_logits = raw.out_proj

    score_model = DenoiserModel(model, path, regime=params.regime)
    sde = LearnedScoreSDE(path, score_model, params.variance)
    simulator = EulerMaruyamaSimulator(sde)

    batch_size = params.batch_size or params.num_samples
    smiles: List[str] = []
    done = 0
    while done < params.num_samples:
        b = min(batch_size, params.num_samples - done)
        x0 = path.p_simple.sample(b, seed=params.seed + done)
        ts = (
            torch.linspace(params.t_start, params.t_end, params.num_timesteps)
            .view(1, params.num_timesteps, 1, 1)
            .expand(b, -1, -1, -1)
            .to(device)
        )
        xts = simulator.simulate(x0, ts, use_bar=params.progress)
        ids = get_logits(xts).softmax(-1).argmax(-1).detach().cpu().tolist()
        smiles.extend(tokenizer.decode_batch(ids, special_decode=True))
        done += b
    return smiles


def validity(smiles_list: List[str]) -> float:
    """rdkit validity rate: a cheap proxy metric for the training log."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles_list:
        return 0.0
    n_valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return n_valid / len(smiles_list)

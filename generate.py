"""Quality composer for the SMILES diffusion model.

Generates N molecules from pure noise, truncates each at the first <eos>,
and reports a panel of metrics (validity + Wilson CI, uniqueness, novelty,
internal diversity, failure breakdown, mean length).

Run:
    export CUDA_VISIBLE_DEVICES=... && python3 evaluate.py
"""
from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Sequence

import torch

from dimol.diffusion.diff_eqs import LearnedScoreSDE
from dimol.diffusion.distributions import GaussianMixture
from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.diffusion.simulators import EulerMaruyamaSimulator
from dimol.models.models import DenoiserModel, DiffusionTransformer
from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class EvalConfig:
    checkpoint: str = "checkpoints/v12/499"
    regime: str = "epsilon"                 # 'epsilon' or 'x'
    seq_len: int = 208
    num_samples: int = 2000                 # >= 2000 for stable validity at low p
    gen_batch_size: int = 500               # lower if you OOM
    num_sampling_timesteps: int = 300
    sampling_variance: float = 1.0
    t_start: float = 1e-3
    t_end: float = 0.999
    seed: int = 42                          # fixed -> runs are comparable
    path_to_tokenizer: str = "data/smiles_bpe.json"
    # optional: path to a newline-delimited file of training SMILES for novelty
    train_smiles_path: Optional[str] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_env(seed: int, use_tf32: bool = False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if use_tf32:
        torch.set_float32_matmul_precision("high")




# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _canon(smi: Optional[str]) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval — correct for small proportions (validity ~3%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_smiles(
    generated: List[str],
    train_canon: Optional[set] = None,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    div_subsample: int = 1000,
) -> dict:
    """`generated` are already-truncated decoded SMILES strings."""
    n = len(generated)
    canon = [_canon(s) for s in generated]
    valid = [c for c in canon if c is not None]
    nv = len(valid)
    uniq_valid = set(valid)

    m: dict = {"n": n, "n_valid": nv}

    m["validity"] = nv / n if n else 0.0
    m["validity_ci95"] = _wilson_ci(nv, n)

    m["uniqueness"] = len(uniq_valid) / nv if nv else 0.0          # among valid

    if train_canon is not None and uniq_valid:
        novel = sum(1 for c in uniq_valid if c not in train_canon)
        m["novelty"] = novel / len(uniq_valid)
    else:
        m["novelty"] = float("nan")

    # internal diversity = 1 - mean pairwise Tanimoto (Morgan), subsampled
    div = float("nan")
    pool = list(uniq_valid)[:div_subsample]
    if len(pool) > 1:
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), fp_radius, fp_bits)
            for s in pool
        ]
        sims: List[float] = []
        for i in range(len(fps)):
            sims += DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        if sims:
            div = 1.0 - sum(sims) / len(sims)
    m["diversity"] = div

    # failure breakdown — counted on the decoded STRING (BPE-agnostic)
    reasons: Counter = Counter()
    for s, c in zip(generated, canon):
        if c is None:
            s = s or ""
            if s.count("(") != s.count(")"):
                reasons["unbalanced_parens"] += 1
            elif any(s.count(str(d)) % 2 for d in range(1, 10)):
                reasons["odd_ring_digits"] += 1
            elif len(s) == 0:
                reasons["empty"] += 1
            else:
                reasons["other"] += 1
    m["failures"] = dict(reasons)
    m["mean_len"] = sum(len(s or "") for s in generated) / n if n else 0.0
    return m


def format_report(m: dict) -> str:
    lo, hi = m["validity_ci95"]
    return "\n".join([
        "================ QUALITY REPORT ================",
        f"samples:    {m['n']}",
        f"validity:   {m['validity']:.3%}   (95% CI {lo:.2%} - {hi:.2%})",
        f"uniqueness: {m['uniqueness']:.3%}   (among {m['n_valid']} valid)",
        f"novelty:    {m['novelty']:.3%}",
        f"diversity:  {m['diversity']:.3f}",
        f"mean_len:   {m['mean_len']:.1f} chars",
        f"failures:   {m['failures']}",
        "===============================================",
    ])


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: torch.nn.Module,
    path: GaussianConditionalProbabilityPath,
    tokenizer: SmilesTokenizer,
    cfg: EvalConfig,
) -> List[str]:
    """Returns decoded SMILES (truncated at first <eos>)."""
    set_env(cfg.seed)  # reproducible per-step SDE noise across checkpoints

    score_model = DenoiserModel(model, path, cfg.regime)
    sde = LearnedScoreSDE(path, score_model, cfg.sampling_variance)
    simulator = EulerMaruyamaSimulator(sde)
    get_logits = model.module.out_proj if hasattr(model, "module") else model.out_proj

    smiles: List[str] = []
    done = 0
    while done < cfg.num_samples:
        b = min(cfg.gen_batch_size, cfg.num_samples - done)
        # vary the noise seed per batch so samples differ, but stay deterministic
        x0 = path.p_simple.sample(b, seed=cfg.seed + done)
        ts = (
            torch.linspace(cfg.t_start, cfg.t_end, cfg.num_sampling_timesteps)
            .view(1, cfg.num_sampling_timesteps, 1, 1)
            .expand(b, -1, -1, -1)
            .to(cfg.device)
        )
        xT = simulator.simulate(x0, ts, use_bar=True)
        ids_batch = get_logits(xT).softmax(-1).argmax(-1).cpu().tolist()
        smiles_list = tokenizer.decode_batch(ids_batch, special_decode=True)
        smiles.extend(smiles_list)
        done += b
        print(f"  generated {done}/{cfg.num_samples}")
    return smiles


# ----------------------------------------------------------------------
# Optional: training set for novelty
# ----------------------------------------------------------------------
def load_train_canon(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[novelty] train smiles file not found: {p} -> skipping novelty")
        return None
    canon = set()
    with open(p) as f:
        for line in f:
            c = _canon(line.strip())
            if c is not None:
                canon.add(c)
    print(f"[novelty] loaded {len(canon)} canonical training SMILES")
    return canon


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    cfg = EvalConfig()
    cfg.checkpoint     = os.environ.get("EVAL_CHECKPOINT", cfg.checkpoint)
    cfg.regime         = os.environ.get("EVAL_REGIME", cfg.regime)
    cfg.num_samples    = int(os.environ.get("EVAL_NUM_SAMPLES", cfg.num_samples))
    cfg.gen_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", cfg.gen_batch_size))
    
    set_env(cfg.seed)

    model = DiffusionTransformer.from_pretrained(load_dir=cfg.checkpoint, map_location=cfg.device)
    model.eval()
    print(f"loaded model from {cfg.checkpoint} (regime={cfg.regime})")

    tokenizer = SmilesTokenizer.load(cfg.path_to_tokenizer)

    # p_simple lives in [seq_len, emb_dim]; read dim from the loaded config
    emb_dim = model.config.emb_dim
    p_data = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0)
    path = GaussianConditionalProbabilityPath(
        p_data=p_data.to(cfg.device),
        p_simple_shape=[cfg.seq_len, emb_dim],
        alpha=CosineAlpha(cfg.device),
        beta=CosineBeta(cfg.device),
    ).to(cfg.device)

    print("generating...")
    smiles = generate(model, path, tokenizer, cfg)

    train_canon = load_train_canon(cfg.train_smiles_path)
    metrics = evaluate_smiles(smiles, train_canon=train_canon)

    print("\n" + format_report(metrics) + "\n")

    print("First 20 generated (truncated at <eos>):")
    for s in smiles[:20]:
        print(f"  {s!r}")

    lines = "\n" + format_report(metrics) + "\n"
    first_gen = '\n'.join([f"  {s!r}\n" for s in smiles[:20]])
    lines += first_gen

    with open(os.path.join(cfg.checkpoint, 'report.txt'), 'w') as f:
        f.write(lines)


if __name__ == "__main__":
    main()

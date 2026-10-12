#!/usr/bin/env python3
"""Turn generated candidates into preference pairs for Diffusion-DPO.

    python scripts/build_preferences.py --config configs/chebi20_finetune.yaml \
        --dirs scratch/train_cand_k0 ... --split train --out data/chebi20/prefs

Why preferences at all. The training objective is an MSE in latent space, and across
eleven configurations it correlated with sample quality at r = +0.46 -- the wrong sign. So
the thing being minimised is not the thing being measured, which is the situation
preference optimisation exists for.

Why exact match drives the reward. A caption-retrieval baseline that generates nothing
already reaches BLEU 0.786, MACCS 0.785 and FCD 0.48 on this benchmark, above the
published MolT5-base row on each, and scores 0.004 on exact match. Fingerprint similarity
is therefore reachable by lookup and exact match is not, so exact match is where
generative skill actually shows.

The reward is graded rather than binary, because exact match fires on only a few percent
of candidates and a sparse reward would leave almost every pair tied:

    invalid string            0.0
    valid                     0.1 + 0.6 * MACCS(candidate, reference)
    exact (canonical match)   1.0

A pair is kept only when the gap is wide enough to be worth a gradient, which is what
stops the model learning from noise between two near-identical candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RDLogger.DisableLog("rdApp.*")

MIN_HEAVY_ATOMS = 3


def reward(candidate: str, reference_mol, reference_canonical: str) -> float:
    if not candidate or not candidate.strip():
        return 0.0
    mol = Chem.MolFromSmiles(candidate)
    if mol is None or mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return 0.0
    try:
        if Chem.MolToSmiles(mol) == reference_canonical:
            return 1.0
    except Exception:
        return 0.0
    if reference_mol is None:
        return 0.1
    try:
        similarity = TanimotoSimilarity(MACCSkeys.GenMACCSKeys(mol),
                                        MACCSkeys.GenMACCSKeys(reference_mol))
    except Exception:
        similarity = 0.0
    return 0.1 + 0.6 * float(similarity)


def read_generations(directory: Path) -> list[str]:
    files = sorted(directory.glob("guidance_*.txt"))
    if not files:
        raise FileNotFoundError(f"no guidance_*.txt in {directory}")
    return files[0].read_text().splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-gap", type=float, default=0.15,
                        help="smallest reward difference worth a gradient")
    args = parser.parse_args()

    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    data_dir = Path(cfg.variables.data_dir)
    seq_len = int(cfg.variables.seq_len)
    pad_idx = int(cfg.variables.pad_idx)
    tokenizer = SmilesTokenizer.load(cfg.tokenizer["path"])

    tokens = np.load(data_dir / f"{args.split}_tokens_00000.npy")
    masks = np.load(data_dir / f"{args.split}_attn_mask_00000.npy")
    candidates = [read_generations(d) for d in args.dirs]
    n = min(len(tokens), min(len(c) for c in candidates))
    k = len(candidates)
    print(f"{n} {args.split} captions, {k} candidates each")

    winners, losers, keep_index = [], [], []
    stats = {"exact_found": 0, "pairs": 0, "all_tied": 0, "all_invalid": 0,
             "mean_best": 0.0, "mean_worst": 0.0}
    for i in range(n):
        ids = [int(t) for t, m in zip(tokens[i], masks[i]) if m]
        reference = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
        reference_mol = Chem.MolFromSmiles(reference)
        try:
            reference_canonical = (Chem.MolToSmiles(reference_mol)
                                   if reference_mol is not None else None)
        except Exception:
            reference_canonical = None

        scores = [reward(candidates[r][i], reference_mol, reference_canonical)
                  for r in range(k)]
        best, worst = int(np.argmax(scores)), int(np.argmin(scores))
        if scores[best] >= 1.0:
            stats["exact_found"] += 1
        if max(scores) <= 0.0:
            stats["all_invalid"] += 1
            continue
        if scores[best] - scores[worst] < args.min_gap:
            stats["all_tied"] += 1
            continue
        stats["mean_best"] += scores[best]
        stats["mean_worst"] += scores[worst]
        winners.append(candidates[best][i])
        losers.append(candidates[worst][i])
        keep_index.append(i)
    stats["pairs"] = len(keep_index)
    for key in ("mean_best", "mean_worst"):
        stats[key] /= max(stats["pairs"], 1)

    def encode_all(strings):
        out = np.full((len(strings), seq_len), pad_idx, dtype=np.int64)
        mask = np.zeros((len(strings), seq_len), dtype=np.uint8)
        for j, s in enumerate(strings):
            encoded = tokenizer.encode(s)[:seq_len] if s.strip() else []
            out[j, : len(encoded)] = encoded
            mask[j, : len(encoded)] = 1
        return out, mask

    args.out.mkdir(parents=True, exist_ok=True)
    w_tokens, w_mask = encode_all(winners)
    l_tokens, l_mask = encode_all(losers)
    np.save(args.out / f"{args.split}_tokens_00000.npy", w_tokens)
    np.save(args.out / f"{args.split}_attn_mask_00000.npy", w_mask)
    np.save(args.out / f"{args.split}_rejected_tokens.npy", l_tokens)
    np.save(args.out / f"{args.split}_rejected_attn_mask.npy", l_mask)
    np.save(args.out / f"{args.split}_text_index.npy",
            np.asarray(keep_index, dtype=np.int64))

    # The caption states are the originals, indexed; copying them would be 5 GB.
    import os
    for name in (f"{args.split}_text.npy", f"{args.split}_text_mask.npy"):
        link = args.out / name
        if not link.exists():
            os.symlink((data_dir / name).resolve(), link)

    (args.out / "meta.json").write_text(json.dumps(
        {"source_dirs": [str(d) for d in args.dirs], "split": args.split,
         "candidates": k, "min_gap": args.min_gap, **stats}, indent=2))

    print(f"  captions with an exact candidate: {stats['exact_found']} "
          f"({stats['exact_found'] / n * 100:.1f}%)")
    print(f"  dropped, every candidate invalid: {stats['all_invalid']}")
    print(f"  dropped, reward gap below {args.min_gap}: {stats['all_tied']}")
    print(f"  pairs kept: {stats['pairs']} ({stats['pairs'] / n * 100:.1f}%)")
    print(f"  mean reward, winner {stats['mean_best']:.3f} vs "
          f"loser {stats['mean_worst']:.3f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score candidate molecules with a caption-conditioned autoregressive model.

    python scripts/score_with_ar.py --config configs/chebi20_ar.yaml \
        --checkpoint runs/ar_text/ep600-ba60000 --split test \
        --dirs scratch/test/cand_k0 ... --out scratch/test/ar_scores.npy

Generating one candidate per candidate length lifted validity from 0.764 to 0.949, but it
cost fingerprint similarity (MACCS 0.754 to 0.731) because the selection rule knew only
whether a string parses and how likely its length was. It had no idea what the caption
asked for, so among several valid candidates it could not prefer the right molecule.

This provides that missing term: log p(SMILES | caption) under an autoregressive decoder
trained on the same data, tokenizer and split as the diffusion model. The reference is
never consulted, so selection stays a decoding procedure rather than an oracle.

Two scores are written. The summed log-likelihood prefers short strings mechanically,
which is exactly the wrong bias when the candidates differ in length by construction, so
the per-token mean is written beside it and is the one to select on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_tokenizer  # noqa: E402
from dimol.models.gpt import SmilesAR  # noqa: E402


def read_generations(directory: Path) -> list[str]:
    files = sorted(directory.glob("guidance_*.txt"))
    if not files:
        raise FileNotFoundError(f"no guidance_*.txt in {directory}")
    return files[0].read_text().splitlines()


@torch.no_grad()
def score_batch(model, tokenizer, smiles, text, text_mask, seq_len, pad_idx, device):
    """Summed and mean log p(token) over the real tokens of each string."""
    ids = np.full((len(smiles), seq_len), pad_idx, dtype=np.int64)
    lengths = np.zeros(len(smiles), dtype=np.int64)
    for i, s in enumerate(smiles):
        encoded = tokenizer.encode(s) if s.strip() else []
        encoded = encoded[:seq_len]
        ids[i, : len(encoded)] = encoded
        lengths[i] = len(encoded)

    token_ids = torch.from_numpy(ids).to(device)
    inp, tgt = token_ids[:, :-1], token_ids[:, 1:]
    kwargs = {}
    if int(getattr(model.config, "text_dim", 0) or 0):
        kwargs = {"text": text, "text_mask": text_mask}
    logits = model(inp, **kwargs)
    logp = F.log_softmax(logits.float(), dim=-1)
    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # (B, L-1)
    real = (tgt != pad_idx).float()
    total = (picked * real).sum(-1)
    count = real.sum(-1).clamp(min=1)
    return total.cpu().numpy(), (total / count).cpu().numpy(), lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer(cfg)
    seq_len = int(cfg.variables.seq_len)
    pad_idx = int(cfg.variables.pad_idx)

    model = SmilesAR.from_pretrained(load_dir=str(args.checkpoint),
                                     map_location=device).to(device).eval()

    data_dir = Path(cfg.variables.data_dir)
    text_all = np.load(data_dir / f"{args.split}_text.npy", mmap_mode="r")
    mask_all = np.load(data_dir / f"{args.split}_text_mask.npy", mmap_mode="r")

    candidates = [read_generations(d) for d in args.dirs]
    n = min(len(text_all), min(len(c) for c in candidates))
    k = len(candidates)
    print(f"{n} captions, {k} candidates each, scoring with {args.checkpoint}")

    totals = np.zeros((n, k), dtype=np.float32)
    means = np.zeros((n, k), dtype=np.float32)
    for rank in range(k):
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            text = torch.from_numpy(
                np.asarray(text_all[start:stop], dtype=np.float32)).to(device)
            mask = torch.from_numpy(
                np.asarray(mask_all[start:stop]).astype(bool)).to(device)
            total, mean, _ = score_batch(
                model, tokenizer, candidates[rank][start:stop], text, mask,
                seq_len, pad_idx, device)
            totals[start:stop, rank] = total
            means[start:stop, rank] = mean
        print(f"  rank {rank}: mean per-token logp {means[:, rank].mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, means)
    np.save(args.out.with_name(args.out.stem + "_total.npy"), totals)
    print(f"  written to {args.out.name} (per-token mean) and "
          f"{args.out.stem}_total.npy (summed)")


if __name__ == "__main__":
    main()

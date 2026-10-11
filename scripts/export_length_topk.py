#!/usr/bin/env python3
"""Export the top-K lengths per caption, not just the mode.

    python scripts/export_length_topk.py --encoder data/chebi20/lenc_5e5 \
        --config configs/chebi20_finetune.yaml --split test --k 5

Exact match needs the length to be exactly right, and the length encoder gets it exactly
right only about 19% of the time even at a mean absolute error of 3.3 tokens. Halving that
error moved exact match not at all (0.047 to 0.048), because closeness is not what the
metric rewards. The oracle length gives 0.142.

The encoder's argmax throws away the rest of a distribution that is often nearly right.
This writes the top-K lengths and their probabilities so a sampler can generate a
candidate at each and choose between them, which is the only way left to convert the
encoder's near-misses into exact hits without a reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_lenc", Path(__file__).resolve().parent / "train_length_encoder.py")
_lenc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lenc)
LengthRegressor, encode, load_split = _lenc.LengthRegressor, _lenc.encode, _lenc.load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path,
                        help="the directory train_length_encoder.py wrote")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-caption-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    corpus = args.corpus or Path(cfg.variables.data_dir)
    max_length = int(cfg.model.max_pos)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(args.encoder / "encoder"))
    encoder = AutoModel.from_pretrained(str(args.encoder / "encoder"))
    model = LengthRegressor(encoder, encoder.config.hidden_size, max_length).to(device)
    model.load_state_dict(torch.load(args.encoder / "heads.pt", map_location=device),
                          strict=False)
    model.eval()

    captions, true_lengths = load_split(corpus, args.split)
    tops, probs = [], []
    with torch.no_grad():
        for start in range(0, len(captions), args.batch_size):
            chunk = captions[start : start + args.batch_size]
            ids, mask = encode(tokenizer, chunk, args.max_caption_tokens, device)
            logits, _ = model(ids, mask)
            p = logits.softmax(-1)
            top = p.topk(args.k, dim=-1)
            tops.append(top.indices.cpu().numpy())
            probs.append(top.values.cpu().numpy())
    top_lengths = np.concatenate(tops)
    top_probs = np.concatenate(probs)

    out_lengths = args.encoder / f"{args.split}_top{args.k}_lengths.npy"
    out_probs = args.encoder / f"{args.split}_top{args.k}_probs.npy"
    np.save(out_lengths, top_lengths)
    np.save(out_probs, top_probs)

    # How much headroom does K candidates actually buy? If the true length is not in the
    # top K, no amount of choosing between them can recover it, so this number bounds
    # everything a candidate sampler could achieve.
    print(f"{args.split}: {len(captions)} captions, top-{args.k}")
    for k in range(1, args.k + 1):
        hit = (top_lengths[:, :k] == true_lengths[:, None]).any(1).mean()
        print(f"  true length within top-{k}: {hit * 100:5.1f}%")
    print(f"  mean probability mass on the top-{args.k}: "
          f"{top_probs.sum(1).mean() * 100:.1f}%")
    print(f"  written to {out_lengths.name} and {out_probs.name}")


if __name__ == "__main__":
    main()

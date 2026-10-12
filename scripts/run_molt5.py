#!/usr/bin/env python3
"""Generate ChEBI-20 test predictions from a published MolT5 checkpoint.

    python scripts/run_molt5.py --config configs/chebi20_finetune.yaml \
        --model laituan245/molt5-base-caption2smiles --out scratch/test/molt5_base

A caption-retrieval baseline that generates nothing already reaches BLEU 0.786, MACCS
0.785 and FCD 0.48 on this benchmark -- above the published MolT5-base row on each of
them. That is a claim about our scorer and our split until it is checked against a
published system's own outputs on the same rows, which is what this produces.

It also makes the proximity stratification a statement about the field rather than about
one model: if a published system's aggregate is carried by the near-duplicate third in the
same way ours is, the benchmark is the thing being characterised.

The captions come from our curated split, so the numbers here are directly comparable to
every other row this project reports, and not to the paper's table, which uses the
uncurated 3,300-pair split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default="laituan245/molt5-base-caption2smiles")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1,
                        help="1 is greedy, which is what the MolT5 paper reports")
    args = parser.parse_args()

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    data_dir = Path(cfg.variables.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions = (data_dir / f"{args.split}_captions.txt").read_text().splitlines()

    # References decoded exactly as every other row in this project decodes them, with the
    # tokenizer the corpus was built with.
    chem_tokenizer = SmilesTokenizer.load(cfg.tokenizer["path"])
    tokens = np.load(data_dir / f"{args.split}_tokens_00000.npy")
    masks = np.load(data_dir / f"{args.split}_attn_mask_00000.npy")
    references = []
    for row, mask in zip(tokens, masks):
        ids = [int(t) for t, m in zip(row, mask) if m]
        references.append(
            chem_tokenizer.decode(ids, skip_special_tokens=True).replace(" ", ""))

    n = min(len(captions), len(references))
    print(f"{args.model}: {n} {args.split} captions, on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    generated = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            chunk = captions[start : start + args.batch_size]
            batch = tokenizer(chunk, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            out = model.generate(**batch, num_beams=args.num_beams,
                                 max_length=args.max_new_tokens)
            generated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (start // args.batch_size) % 20 == 0:
                print(f"  {start + len(chunk)}/{n}")

    generated = [s.replace(" ", "") for s in generated[:n]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guidance_0.0_molt5.txt").write_text("\n".join(generated) + "\n")
    (args.out / "reference.txt").write_text("\n".join(references[:n]) + "\n")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

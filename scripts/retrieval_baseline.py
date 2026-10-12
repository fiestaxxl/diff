#!/usr/bin/env python3
"""A caption-retrieval baseline that generates nothing.

    python scripts/retrieval_baseline.py --config configs/chebi20_finetune.yaml \
        --out scratch/test/retrieval

For every test caption, find the most similar training caption by TF-IDF cosine and return
that training molecule unchanged. No model, no decoding, no chemistry: pure lookup.

It exists to put a floor under the benchmark. ChEBI-20's test split is largely a
neighbourhood of its training split -- 86.9% of test molecules share a Bemis-Murcko
scaffold with a training molecule and the median nearest-neighbour Morgan similarity is
0.787 -- so a system that retrieves well should already score well on fingerprint metrics
without generating anything. If it does, aggregate fingerprint numbers on this benchmark
cannot by themselves demonstrate generative skill, and every published row needs reading
against this floor.

The floor is deliberately weak: TF-IDF over caption text, no embeddings, no reranking, no
tuning. A stronger retriever would only strengthen the argument.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def decode_split(data_dir: Path, split: str, tokenizer) -> list[str]:
    tokens = np.load(data_dir / f"{split}_tokens_00000.npy")
    masks = np.load(data_dir / f"{split}_attn_mask_00000.npy")
    out = []
    for row, mask in zip(tokens, masks):
        ids = [int(t) for t, m in zip(row, mask) if m]
        out.append(tokenizer.decode(ids, skip_special_tokens=True).replace(" ", ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    data_dir = Path(cfg.variables.data_dir)
    tokenizer = SmilesTokenizer.load(cfg.tokenizer["path"])

    train_captions = (data_dir / "train_captions.txt").read_text().splitlines()
    test_captions = (data_dir / f"{args.split}_captions.txt").read_text().splitlines()
    train_molecules = decode_split(data_dir, "train", tokenizer)
    references = decode_split(data_dir, args.split, tokenizer)

    n = min(len(test_captions), len(references))
    print(f"{len(train_captions)} training captions, {n} {args.split} captions")

    vec = TfidfVectorizer(max_features=args.max_features, sublinear_tf=True,
                          stop_words="english", ngram_range=(1, 2))
    train_matrix = vec.fit_transform(train_captions)
    test_matrix = vec.transform(test_captions[:n])

    retrieved, similarity = [], []
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        scores = linear_kernel(test_matrix[start:stop], train_matrix)
        best = scores.argmax(1)
        retrieved.extend(train_molecules[int(b)] for b in best)
        similarity.extend(float(scores[i, b]) for i, b in enumerate(best))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guidance_0.0_retrieval.txt").write_text("\n".join(retrieved) + "\n")

    # Token ids too, so the retrieved molecules can seed a diffusion trajectory.
    seq_len = int(cfg.variables.seq_len)
    pad_idx = int(cfg.variables.pad_idx)
    ids = np.full((len(retrieved), seq_len), pad_idx, dtype=np.int64)
    for i, smiles in enumerate(retrieved):
        encoded = tokenizer.encode(smiles)[:seq_len]
        ids[i, : len(encoded)] = encoded
    np.save(args.out / "retrieved_tokens.npy", ids)
    (args.out / "reference.txt").write_text("\n".join(references[:n]) + "\n")
    np.save(args.out / "caption_similarity.npy", np.asarray(similarity))

    exact_copies = sum(1 for r, g in zip(references[:n], retrieved) if r == g)
    print(f"  mean caption cosine to the retrieved neighbour: {np.mean(similarity):.3f}")
    print(f"  retrieved molecule is literally the reference: {exact_copies} "
          f"({exact_copies / n * 100:.1f}%)")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

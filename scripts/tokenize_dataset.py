#!/usr/bin/env python3
"""Tokenize a corpus into padded arrays (train/val/test) driven by config.

    python scripts/tokenize_dataset.py configs/tokenizer_chebi.yaml

Same logic as the old tokenize_dataset.py; the data source, the length and the
number of worker processes come from the ``prepare`` config section.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402

tok: SmilesTokenizer | None = None


def encode_one(args):
    smiles, max_length = args
    return tok.encode_padded(smiles, max_length=max_length, add_special_tokens=True)


def init_worker(tokenizer_path: str):
    """multiprocessing worker init: load the tokenizer once per process."""
    global tok
    tok = SmilesTokenizer.load(tokenizer_path)


def process_split(
    split_name: str,
    smiles_list: list[str],
    tokenizer_path: str,
    out_dir: Path,
    max_length: int,
    nprocs: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(smiles_list)
    token_arr = np.zeros((n, max_length), dtype=np.uint16)
    mask_arr = np.zeros((n, max_length), dtype=np.uint8)
    keep = np.zeros(n, dtype=bool)

    args_iter = ((s, max_length) for s in smiles_list)

    if nprocs > 1:
        import multiprocessing as mp

        with mp.Pool(nprocs, initializer=init_worker, initargs=(tokenizer_path,)) as pool:
            it = pool.imap(encode_one, args_iter, chunksize=64)
            for i, (toks, mask) in enumerate(tqdm(it, total=n, desc=f"tokenize {split_name}")):
                if toks is None:
                    continue
                token_arr[i] = toks
                mask_arr[i] = mask
                keep[i] = True
    else:
        init_worker(tokenizer_path)
        for i, item in enumerate(tqdm(args_iter, total=n, desc=f"tokenize {split_name}")):
            toks, mask = encode_one(item)
            if toks is None:
                continue
            token_arr[i] = toks
            mask_arr[i] = mask
            keep[i] = True

    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"  dropped {n_dropped} molecules longer than max_length={max_length}")
    token_arr = token_arr[keep]
    mask_arr = mask_arr[keep]

    np.save(out_dir / f"{split_name}_tokens.npy", token_arr)
    np.save(out_dir / f"{split_name}_attn_mask.npy", mask_arr)
    print(f"  saved {token_arr.shape[0]} molecules to {out_dir}/{split_name}_*.npy")


def main(cfg: DictConfig) -> None:
    log_config(cfg)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    source = dict(prep.get("source") or {})
    splits = dict(source.get("splits") or {"train": "train", "val": "validation", "test": "test"})

    tokenizer_path = str(
        prep.get("tokenizer", {}).get("out") or cfg.tokenizer.get("path", None)
    )
    out_dir = Path(prep.get("out_dir") or "data/tokenized")
    max_length = int(prep.get("max_length", 208))
    nprocs = int(prep.get("nprocs") or max(1, (os.cpu_count() or 2) // 2))
    canonicalize = bool(prep.get("canonicalize", True))

    # Sanity: ensure the tokenizer's vocab fits in uint16
    tokenizer = SmilesTokenizer.load(tokenizer_path)
    assert tokenizer.vocab_size <= 65535, (
        f"vocab size {tokenizer.vocab_size} doesn't fit in uint16; switch to uint32"
    )

    for out_name, split_name in splits.items():
        smiles_list = list(iter_smiles(source, split_name, canonicalize=canonicalize))
        print(f"[{out_name}] {len(smiles_list)} molecules from split {split_name!r}")
        process_split(
            split_name=out_name,
            smiles_list=smiles_list,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            max_length=max_length,
            nprocs=nprocs,
        )


if __name__ == "__main__":
    main(load_from_argv())

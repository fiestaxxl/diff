"""Tokenize ChEBI-20-MM (train/val/test) and save padded arrays per split."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from smiles_tokenizer import SmilesTokenizer

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/smiles_bpe.json"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/tokenized"))
    ap.add_argument("--max_length", type=int, default=208)
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    # Sanity: ensure the tokenizer's vocab fits in uint16
    tok = SmilesTokenizer.load(str(args.tokenizer))
    assert tok.vocab_size <= 65535, (
        f"vocab size {tok.vocab_size} doesn't fit in uint16; switch to uint32"
    )

    ds = load_dataset("liupf/ChEBI-20-MM")
    splits = {"train": "train", "val": "validation", "test": "test"}

    for out_name, ds_name in splits.items():
        smiles_list = [ex["SMILES"] for ex in ds[ds_name] if ex["SMILES"]]
        process_split(
            split_name=out_name,
            smiles_list=smiles_list,
            tokenizer_path=str(args.tokenizer),
            out_dir=args.out_dir,
            max_length=args.max_length,
            nprocs=args.nprocs,
        )


if __name__ == "__main__":
    main()
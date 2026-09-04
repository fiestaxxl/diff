#!/usr/bin/env python3
"""Tokenize a curated corpus into sharded, memory-mappable arrays.

    python scripts/tokenize_dataset.py configs/zinc250k.yaml

Molecules stream from the corpus shards through a process pool and are written into
<out_dir>/<split>_tokens_XXXXX.npy plus the matching attention masks. Peak memory is
one shard, so the corpus size does not matter. Training reads these files with mmap.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.data.shards import ArrayShardWriter  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402

tok: SmilesTokenizer | None = None
MAX_LENGTH = 0


def init_worker(tokenizer_path: str, max_length: int) -> None:
    """multiprocessing worker init: load the tokenizer once per process."""
    global tok, MAX_LENGTH
    tok = SmilesTokenizer.load(tokenizer_path)
    MAX_LENGTH = max_length


def encode_one(smiles: str):
    return tok.encode_padded(smiles, max_length=MAX_LENGTH, add_special_tokens=True)


def corpus_source(prep: dict) -> dict:
    """Where the curated corpus lives; falls back to the raw source."""
    return dict(prep.get("corpus") or prep.get("source") or {})


def process_split(
    split_name: str,
    source: dict,
    upstream_split: str,
    tokenizer_path: str,
    out_dir: Path,
    max_length: int,
    shard_size: int,
    nprocs: int,
) -> dict:
    writer = ArrayShardWriter(out_dir, split_name, max_length=max_length, shard_size=shard_size)
    stream = iter_smiles(source, upstream_split, canonicalize=False)
    dropped = 0
    lengths: list[int] = []

    bar = tqdm(desc=f"tokenize {split_name}", unit="mol")
    if nprocs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        pool = ctx.Pool(nprocs, initializer=init_worker, initargs=(tokenizer_path, max_length))
        results = pool.imap(encode_one, stream, chunksize=256)
    else:
        init_worker(tokenizer_path, max_length)
        pool = None
        results = (encode_one(s) for s in stream)

    for tokens, mask in results:
        bar.update(1)
        if tokens is None:  # longer than max_length
            dropped += 1
            continue
        writer.write(tokens, mask)
        lengths.append(int(sum(mask)))
    bar.close()
    if pool is not None:
        pool.close()
        pool.join()

    paths = writer.close()
    arr = np.asarray(lengths, dtype=np.int32)
    stats = {
        "kept": writer.count,
        "dropped_too_long": dropped,
        "shards": len(paths) // 2,
        "tokens_per_molecule": {
            "mean": round(float(arr.mean()), 2) if arr.size else 0,
            "p50": int(np.percentile(arr, 50)) if arr.size else 0,
            "p95": int(np.percentile(arr, 95)) if arr.size else 0,
            "p99": int(np.percentile(arr, 99)) if arr.size else 0,
            "max": int(arr.max()) if arr.size else 0,
        },
        "padding_share": round(1 - float(arr.mean()) / max_length, 4) if arr.size else 0,
    }
    print(f"  {split_name}: {writer.count} molecules in {stats['shards']} shard(s), "
          f"dropped {dropped}, tokens p50/p99/max "
          f"{stats['tokens_per_molecule']['p50']}/{stats['tokens_per_molecule']['p99']}/"
          f"{stats['tokens_per_molecule']['max']}, padding {stats['padding_share']:.1%}")
    return stats


def main(cfg: DictConfig) -> None:
    log_config(cfg)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    source = corpus_source(prep)
    splits = dict(source.get("splits") or {"train": "train", "val": "val", "test": "test"})

    tokenizer_path = str(prep.get("tokenizer", {}).get("out") or cfg.tokenizer.get("path", None))
    out_dir = Path(prep.get("tokenized_dir") or "data/tokenized")
    max_length = int(prep.get("max_length", 96))
    shard_size = int(prep.get("array_shard_size", 250_000))
    nprocs = int(prep.get("nprocs") or max(1, (os.cpu_count() or 2) // 2))

    tokenizer = SmilesTokenizer.load(tokenizer_path)
    assert tokenizer.vocab_size <= 65535, (
        f"vocab size {tokenizer.vocab_size} doesn't fit in uint16; switch to uint32"
    )
    print(f"[tokenize] {tokenizer_path} (vocab {tokenizer.vocab_size}), max_length {max_length}, "
          f"{nprocs} processes, shard {shard_size}")

    started = time.time()
    report = {"tokenizer": tokenizer_path, "vocab_size": tokenizer.vocab_size,
              "max_length": max_length, "splits": {}}
    for out_name, upstream_split in splits.items():
        report["splits"][out_name] = process_split(
            out_name, source, upstream_split, tokenizer_path, out_dir,
            max_length, shard_size, nprocs,
        )
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "meta.json").write_text(json.dumps(report, indent=2))
    print(f"meta written to {out_dir / 'meta.json'} ({report['seconds']}s)")


if __name__ == "__main__":
    main(load_from_argv())

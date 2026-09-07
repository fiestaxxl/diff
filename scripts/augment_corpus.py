#!/usr/bin/env python3
"""Multiply a curated corpus by random SMILES traversals.

    python scripts/augment_corpus.py configs/zinc250k.yaml

A molecule has many valid SMILES strings; the corpus holds one canonical traversal of
each. With 224,568 molecules and a budget of 160 tokens per parameter the model makes
over a hundred passes over the same strings, so the grammar is learned from a single
walk per molecule. Emitting K random traversals per molecule keeps the chemistry and
multiplies the number of distinct strings, which is the cheapest way to attack that
repetition.

Only the train split is augmented: validation and test stay canonical so that the
numbers remain comparable.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.data.shards import TextShardWriter  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402

_K = 1


def _init(k: int) -> None:
    global _K
    _K = int(k)


def _variants(smiles: str) -> list[str]:
    """The canonical string plus K-1 random traversals of the same molecule."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = {smiles}
    for _ in range(_K * 3):  # oversample: random traversals repeat
        if len(out) >= _K:
            break
        out.add(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
    return list(out)


def main(cfg: DictConfig) -> None:
    log_config(cfg)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    source = dict(prep.get("corpus") or {})
    aug = dict(prep.get("augment") or {})
    k = int(aug.get("variants", 4))
    out_dir = Path(aug.get("out_dir") or "data/corpus_aug")
    shard_size = prep.get("shard_size")
    nprocs = int(prep.get("nprocs") or max(1, (os.cpu_count() or 2) // 2))

    print(f"[augment] {k} traversals per molecule, {nprocs} processes -> {out_dir}")
    report = {"variants": k, "splits": {}}
    started = time.time()

    for split in ("train", "val", "test"):
        stream = iter_smiles(source, split, canonicalize=False)
        writer = TextShardWriter(out_dir / split, shard_size=shard_size)
        seen = 0
        if split == "train" and k > 1:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(nprocs, initializer=_init, initargs=(k,)) as pool:
                for variants in tqdm(
                    pool.imap(_variants, stream, chunksize=256), desc=f"augment {split}"
                ):
                    for smi in variants:
                        writer.write(smi)
                    seen += 1
        else:
            for smi in stream:
                writer.write(smi)
                seen += 1
        writer.close()
        report["splits"][split] = {"molecules": seen, "strings": writer.count}
        print(f"  {split}: {seen} molecules -> {writer.count} strings")

    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2))
    print(f"done in {report['seconds']}s")


if __name__ == "__main__":
    main(load_from_argv())

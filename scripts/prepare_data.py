#!/usr/bin/env python3
"""Turn a raw SMILES source into a curated, sharded corpus.

    python scripts/prepare_data.py configs/zinc250k.yaml

Molecules stream from the source through a process pool (canonicalization is the
expensive part) straight into shard files, so memory does not grow with the corpus.
Writes <out_dir>/<split>/shard_XXXXX.txt plus a stats.json recording what every
curation step removed and the length distribution of what survived.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.data.curate import Curator, stable_bucket  # noqa: E402
from dimol.data.shards import TextShardWriter  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402


def _percentiles(histogram: np.ndarray) -> dict:
    """Length statistics from a histogram over integer lengths.

    A list of lengths costs 28 bytes a molecule as Python ints, which is six gigabytes for
    a 218M-molecule corpus. Lengths are small integers, so one counter per length is
    exact and costs nothing.
    """
    total = int(histogram.sum())
    if total == 0:
        return {}
    lengths = np.arange(histogram.size)
    cumulative = np.cumsum(histogram)

    def quantile(q: float) -> int:
        return int(np.searchsorted(cumulative, q * total, side="left"))

    return {
        "count": total,
        "mean": round(float((lengths * histogram).sum() / total), 2),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": int(np.nonzero(histogram)[0].max()),
    }


def main(cfg: DictConfig) -> None:
    log_config(cfg)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    source = dict(prep.get("source") or {})
    splits = dict(source.get("splits") or {"train": "train"})
    out_dir = Path(prep.get("out_dir") or "data/corpus")
    shard_size = prep.get("shard_size")
    seed = int(prep.get("seed", 42))
    test_from_val = float(prep.get("test_from_val", 0.0))
    # A corpus with a single split needs val and test carved out of it. The decision is
    # per molecule and depends only on the molecule and the seed, so the splits stay
    # disjoint and reproducible without holding anything in memory.
    carve = dict(prep.get("carve") or {})
    carve_from = carve.get("from")
    carve_val = float(carve.get("val", 0.0))
    carve_test = float(carve.get("test", 0.0))
    if carve_val + carve_test >= 1.0:
        raise ValueError("prepare.carve: val + test must leave something for train")
    nprocs = prep.get("nprocs")
    max_len = int(prep.get("max_smiles_len") or 0) or 512

    curator = Curator(
        canonicalize_smiles=bool(prep.get("canonicalize", True)),
        max_smiles_len=prep.get("max_smiles_len"),
        allowed_elements=prep.get("allowed_elements"),
        deduplicate=bool(prep.get("deduplicate", True)),
        nprocs=int(nprocs) if nprocs else None,
    )
    print(f"[curate] processes: {curator.nprocs}")

    report: dict = {
        "source": source,
        "seed": seed,
        "shard_size": shard_size,
        "splits": {},
        "length_chars": {},
    }
    writers: dict = {}
    lengths: dict = {}

    def writer_for(name: str) -> TextShardWriter:
        if name not in writers:
            writers[name] = TextShardWriter(out_dir / name, shard_size=shard_size)
            lengths[name] = np.zeros(max_len + 2, dtype=np.int64)
        return writers[name]

    started = time.time()
    for out_name, upstream_split in splits.items():
        before = dict(vars(curator.stats))
        stream = iter_smiles(source, upstream_split, canonicalize=False)
        n = 0
        for smi in curator.stream(stream):
            # Route without holding the split in memory: the target depends only on the
            # molecule and the seed, so the same molecule always lands in the same split.
            target = out_name
            if out_name == "val" and test_from_val > 0:
                if stable_bucket(smi, seed=seed, buckets=10_000) < int(test_from_val * 10_000):
                    target = "test"
            elif out_name == carve_from:
                bucket = stable_bucket(smi, seed=seed, buckets=1_000_000)
                if bucket < carve_val * 1_000_000:
                    target = "val"
                elif bucket < (carve_val + carve_test) * 1_000_000:
                    target = "test"
            writer = writer_for(target)
            writer.write(smi)
            lengths[target][min(len(smi), max_len + 1)] += 1
            n += 1
        delta = {k: getattr(curator.stats, k) - before.get(k, 0)
                 for k in ("total", "unparsable", "too_long", "forbidden_elements",
                           "duplicates", "kept")}
        report["splits"][out_name] = delta
        print(f"[{out_name}] from {upstream_split!r}: {delta['total']} -> {delta['kept']} kept "
              f"(unparsable {delta['unparsable']}, too long {delta['too_long']}, "
              f"forbidden elements {delta['forbidden_elements']}, duplicates {delta['duplicates']})")

    for name, writer in writers.items():
        paths = writer.close()
        report["length_chars"][name] = _percentiles(lengths[name])
        print(f"wrote {writer.count:>7} molecules to {out_dir / name} in {len(paths)} shard(s)")

    report["element_counts"] = curator.stats.as_dict()["element_counts"]
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"stats written to {out_dir / 'stats.json'} ({report['seconds']}s)")


if __name__ == "__main__":
    main(load_from_argv())

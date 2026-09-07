#!/usr/bin/env python3
"""Group sampled runs by configuration, average over seeds, and say what is resolvable.

    python scripts/compare_configs.py --reference r_base80 \
        /workspace/samples_w7/metrics.json /workspace/samples_w9/metrics.json

The input files are the metrics.json that scripts/evaluate_samples.py writes next to a
directory of sample files. Runs are grouped by name with a trailing seed suffix removed
(r_tl0_ce3_s43 and r_tl0_ce3 land in the same group), and each group is reported as a
mean over its seeds with the observed range.

Pass only metrics files whose generations share their sampler settings: the same run
decoded at 100 and at 300 solver steps is two different numbers, and this script has no
way to know that. A run that appears in several files is counted once, from the first file
that holds it. Grouping relies on the `_s<seed>` naming convention for seeds.

The verdict column exists because of what this study measured: at 5M parameters the
binomial interval on 10,000 samples is half a point while the spread between seeds of one
configuration reaches ten, so a single run cannot separate configurations that differ by
less than that. A group is only called an effect when its worst seed beats the
reference's best one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

SEED_SUFFIX = re.compile(r"_s\d+$")


def group_name(run: str) -> str:
    return SEED_SUFFIX.sub("", run.replace(".txt", ""))


def load(paths: List[Path]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for path in paths:
        data = json.loads(path.read_text())
        for run, metrics in data.items():
            name = run.replace(".txt", "")
            if name in seen:
                duplicates.append(f"{name} (kept from {seen[name]})")
                continue
            seen[name] = path.parent.name
            metrics = dict(metrics)
            metrics["run"] = name
            metrics["source"] = path.parent.name
            groups.setdefault(group_name(run), []).append(metrics)
    if duplicates:
        print(f"counted once, listed in several files: {', '.join(duplicates)}\n")
    return groups


def summarize(runs: List[dict]) -> dict:
    validity = sorted(m["validity"] * 100 for m in runs)
    unique = [m["validity"] * m.get("uniqueness", 1.0) * 10000 for m in runs]
    return {
        "n": len(runs),
        "mean": sum(validity) / len(validity),
        "low": validity[0],
        "high": validity[-1],
        "unique": sum(unique) / len(unique),
    }


def verdict(group: dict, ref: dict) -> str:
    if ref is None:
        return ""
    if group["n"] < 2:
        return "one seed, unresolved"
    if group["low"] > ref["high"]:
        return "clears the reference"
    if group["high"] < ref["low"]:
        return "below the reference"
    return "inside the spread"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--reference", default=None,
                        help="group name to compare against, e.g. r_base80")
    parser.add_argument("--sort", choices=("mean", "unique"), default="mean")
    args = parser.parse_args()

    missing = [p for p in args.metrics if not p.exists()]
    if missing:
        raise SystemExit(f"no such metrics file: {missing[0]}")

    groups = {name: summarize(runs) for name, runs in load(args.metrics).items()}
    if not groups:
        raise SystemExit("the metrics files held no runs")
    ref = groups.get(args.reference) if args.reference else None
    if args.reference and ref is None:
        raise SystemExit(f"reference group {args.reference!r} is not in these files; "
                         f"have: {', '.join(sorted(groups))}")

    print(f"{'configuration':<26}{'seeds':>6}{'mean':>8}{'range':>16}"
          f"{'unique valid':>14}{'delta':>8}  verdict")
    order = sorted(groups.items(), key=lambda kv: -kv[1][args.sort])
    for name, g in order:
        delta = "" if ref is None else "%+.2f" % (g["mean"] - ref["mean"])
        span = "%.2f-%.2f" % (g["low"], g["high"])
        print("%-26s%6d%6.2f%%%16s%14.0f%8s  %s"
              % (name, g["n"], g["mean"], span, g["unique"], delta, verdict(g, ref)))


if __name__ == "__main__":
    main()

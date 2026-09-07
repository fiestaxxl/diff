#!/usr/bin/env python3
"""Compare runs by validation loss against the token budget they consumed.

    python scripts/analyze_runs.py runs/probe_5m_s42 runs/probe_17m_s42

Everything is read from the run directory: config.resolved.yaml gives the parameter
count and the batch size, metrics.jsonl gives the evaluation curve, and the tokenized
corpus meta.json gives the mean tokens per molecule. The output is validation loss as a
function of tokens per parameter, plus the smallest budget that already reaches within
1% and 2% of the best value the run ever saw. That budget is the number to carry over to
a larger model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = [0.5, 1, 2, 5, 10, 20, 40, 80]
KEY = "loss/eval/total"


def mean_tokens_per_molecule(data_dir: Path, split: str = "train") -> float:
    meta = data_dir / "meta.json"
    if meta.exists():
        report = json.loads(meta.read_text())
        stats = report.get("splits", {}).get(split, {})
        value = stats.get("tokens_per_molecule", {}).get("mean")
        if value:
            return float(value)
    return 22.5


def read_run(run_dir: Path) -> Optional[Dict[str, object]]:
    cfg_path, metrics_path = run_dir / "config.resolved.yaml", run_dir / "metrics.jsonl"
    if not cfg_path.exists() or not metrics_path.exists():
        print(f"[skip] {run_dir}: no config.resolved.yaml or metrics.jsonl")
        return None
    cfg = OmegaConf.load(cfg_path)
    params = int(cfg.n_params or 0)
    batch = int(cfg.global_train_batch_size or 0)
    tokens_per_mol = mean_tokens_per_molecule(Path(cfg.variables["data_dir"]))

    curve: List[Tuple[int, float, float]] = []
    with metrics_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") != "metrics" or KEY not in row:
                continue
            step = int(row["step"])
            tokens = step * batch * tokens_per_mol
            curve.append((step, tokens / max(params, 1), float(row[KEY])))
    if not curve:
        print(f"[skip] {run_dir}: no evaluation points yet")
        return None
    return {
        "name": str(cfg.run_name),
        "params": params,
        "batch": batch,
        "tokens_per_mol": tokens_per_mol,
        "curve": curve,
        "steps": curve[-1][0],
    }


def at_budget(curve: List[Tuple[int, float, float]], target: float) -> Optional[float]:
    """Loss at the last evaluation that did not exceed the target budget."""
    seen = [loss for _, tpp, loss in curve if tpp <= target]
    return seen[-1] if seen else None


def first_within(curve: List[Tuple[int, float, float]], best: float, tol: float) -> Optional[float]:
    for _, tpp, loss in curve:
        if loss <= best * (1 + tol):
            return tpp
    return None


def main() -> None:
    run_dirs = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    if not run_dirs:
        raise SystemExit("usage: python scripts/analyze_runs.py <run_dir> [<run_dir> ...]")

    runs = [r for r in (read_run(d) for d in run_dirs) if r]
    if not runs:
        return

    header = f"{'run':<16}{'params':>12}{'steps':>8}" + "".join(f"{g:>8}" for g in GRID)
    print("\nvalidation loss at N tokens per parameter")
    print(header)
    print("-" * len(header))
    for run in runs:
        cells = []
        for g in GRID:
            value = at_budget(run["curve"], g)
            cells.append(f"{value:>8.3f}" if value is not None else f"{'-':>8}")
        print(f"{run['name']:<16}{run['params']:>12,}{run['steps']:>8}" + "".join(cells))

    print("\nwhere the curve flattens")
    print(f"{'run':<16}{'best loss':>11}{'at t/p':>9}{'within 1%':>11}{'within 2%':>11}")
    for run in runs:
        losses = [loss for _, _, loss in run["curve"]]
        best = min(losses)
        best_tpp = [tpp for _, tpp, loss in run["curve"] if loss == best][0]
        w1 = first_within(run["curve"], best, 0.01)
        w2 = first_within(run["curve"], best, 0.02)
        print(f"{run['name']:<16}{best:>11.3f}{best_tpp:>9.1f}"
              f"{(f'{w1:.1f}' if w1 else '-'):>11}{(f'{w2:.1f}' if w2 else '-'):>11}")

    print("\nsteps needed for a given budget, at this batch size")
    print(f"{'run':<16}{'batch':>7}{'tok/mol':>9}" + "".join(f"{g:>8}" for g in GRID))
    for run in runs:
        cells = []
        for g in GRID:
            steps = g * run["params"] / (run["batch"] * run["tokens_per_mol"])
            cells.append(f"{steps:>8.0f}")
        print(f"{run['name']:<16}{run['batch']:>7}{run['tokens_per_mol']:>9.1f}" + "".join(cells))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score already generated SMILES: validity, uniqueness, novelty, diversity.

    python scripts/evaluate_samples.py configs/generate_zinc250k.yaml \
        evaluate.samples=checkpoints/zinc_baseline/samples/ba7200.txt

Sampling needs a GPU, scoring needs rdkit. When they are not in the same place, run
scripts/generate.py (or training with sampling on) where the GPU is, and this script
where rdkit is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv  # noqa: E402
from dimol.eval.report import evaluate_smiles, format_report, load_train_canon  # noqa: E402


def main(cfg: DictConfig) -> None:
    gen = OmegaConf.to_container(cfg.generate, resolve=True) or {}
    samples_path = gen.get("samples")
    if not samples_path:
        raise KeyError("set generate.samples to a file with one SMILES per line")

    path = Path(samples_path)
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no sample files under {path}")

    train_canon = load_train_canon(gen.get("train_smiles_path"))
    reports = {}
    for file in files:
        smiles = [line.strip() for line in file.read_text().splitlines() if line.strip()]
        metrics = evaluate_smiles(smiles, train_canon=train_canon)
        reports[file.name] = metrics
        print(f"\n=== {file}")
        print(format_report(metrics))

    if len(reports) > 1:
        print(f"\n{'run':<24}{'valid':>8}{'95% CI':>16}{'uniq':>8}{'novel':>8}{'div':>7}"
              f"{'parens':>8}{'rings':>7}{'other':>7}")
        for name, m in sorted(reports.items(), key=lambda kv: -kv[1]["validity"]):
            lo, hi = m["validity_ci95"]
            fails = m.get("failures", {})
            total_fail = max(sum(fails.values()), 1)
            print(f"{name.replace('.txt', ''):<24}{m['validity'] * 100:>7.2f}%"
                  f"{f'{lo * 100:.2f}-{hi * 100:.2f}':>16}"
                  f"{m['uniqueness'] * 100:>7.1f}%{m['novelty'] * 100:>7.1f}%"
                  f"{m['diversity']:>7.3f}"
                  f"{fails.get('unbalanced_parens', 0) / total_fail * 100:>7.0f}%"
                  f"{fails.get('odd_ring_digits', 0) / total_fail * 100:>6.0f}%"
                  f"{fails.get('other', 0) / total_fail * 100:>6.0f}%")

    out = (path if path.is_dir() else path.parent) / "metrics.json"
    out.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\nmetrics written to {out}")


if __name__ == "__main__":
    main(load_from_argv())
